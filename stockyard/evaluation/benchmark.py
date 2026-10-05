"""Run all methods on identical public scenarios; static-only SA/ACO/MIP are explicit."""

import argparse
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
import random
import numpy as np
import pandas as pd
import torch
from stockyard.data.sample import load, to_plates, validate
from stockyard.environment.yard import Locating
from stockyard.baselines import heuristic as h
from stockyard.evaluation.policy import rollout
from stockyard.training.ppo import make_model


def make_env(df, max_stack):
    df = validate(df)
    if (
        df.inbound.nunique() == 1
        and df.pileno.nunique() > 1
        and df.groupby("pileno").size().max() > max_stack
    ):
        raise ValueError("Static source stack exceeds max_stack; increase --max-stack")
    return Locating(
        inbound_plates=to_plates(df),
        max_stack=max_stack,
        min_obstacles=0,
        max_obstacles=0,
        max_steps=max(200, len(df) * 4),
    )


def run_rule(df, max_stack, source, dest, seed):
    random.seed(seed)
    np.random.seed(seed)
    env = make_env(df, max_stack)
    env.reset()
    agent = h.ConfigurableHeuristicAgent(
        source, dest, {"max_tier": max_stack, "max_bay": len(env.to_keys)}
    )
    started = perf_counter()
    info = {}
    for _ in range(env.max_steps):
        action = agent.get_action(env)
        if action is None:
            action = (0, 0)  # environment advances time while no source is active
        _, _, done, info = env.step(action)
        if done:
            break
    complete = info.get(
        "episode_end_reason"
    ) == "from_cleared" and env.crane_move == len(df)
    metric = (
        info.get("final_blocking_metric", float("inf")) if complete else float("inf")
    )
    return (
        metric,
        env.crane_move,
        perf_counter() - started,
        info.get("episode_end_reason") or "incomplete",
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/sample/scenario.csv")
    p.add_argument(
        "--methods",
        nargs="+",
        choices=["random", "edd-mod", "sop-mfb", "sa", "aco", "gurobi", "ppo"],
        default=["random", "edd-mod", "sop-mfb", "sa", "aco"],
    )
    p.add_argument("--checkpoint", default="outputs/policy.pt")
    p.add_argument("--output", default="outputs/benchmark.csv")
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--max-stack", type=int, default=12)
    p.add_argument("--budget-seconds", type=float, default=2.0)
    a = p.parse_args()
    if a.max_stack < 1 or a.budget_seconds <= 0:
        p.error("Positive stack height and time budget required")
    torch.set_num_threads(1)
    df = load(a.data)
    if len(df) > df.topile.nunique() * a.max_stack:
        raise ValueError("Insufficient destination capacity")
    static = df.inbound.nunique() == 1
    make_env(df, a.max_stack)  # Apply the same scenario checks before any solver runs.
    records = []
    cfg = SimpleNamespace(
        max_stack=a.max_stack,
        OBSERVED_TOP_N_PLATES=10,
        NUM_SUMMARY_STATS_DEEPER=4,
        gamma=0.995,
        max_steps=max(200, len(df) * 4),
    )
    for method in a.methods:
        random.seed(a.seed)
        np.random.seed(a.seed)
        torch.manual_seed(a.seed)
        if method in ["sa", "aco", "gurobi"] and not static:
            records.append(
                dict(
                    method=method,
                    blocking_pairs=None,
                    moves=None,
                    runtime_seconds=None,
                    status="skipped: static-only",
                    completed=False,
                )
            )
            continue
        if method in ["random", "edd-mod", "sop-mfb"]:
            source, dest = {
                "random": (h.source_random_selection, h.dest_random_stacking),
                "edd-mod": (h.source_P1_EDD, h.dest_PH1_MOD),
                "sop-mfb": (
                    h.source_SOP_ShortestOutbound,
                    h.dest_OH1_minimize_final_blocking,
                ),
            }[method]
            metric, moves, duration, status = run_rule(
                df, a.max_stack, source, dest, a.seed
            )
        elif method == "sa":
            from stockyard.baselines.sa import run_sa_one_scenario

            metric, duration, moves = run_sa_one_scenario(
                df, cfg, time_limit_s=a.budget_seconds, max_steps=1000, seed=a.seed
            )
            status = "from_cleared"
        elif method == "aco":
            from stockyard.baselines import aco

            aco.PRINT_PROGRESS = False
            env = make_env(df, a.max_stack)
            env.reset()
            result = aco.WuStyleStaticACO(
                n_ants=4, n_iterations=10, time_limit_sec=a.budget_seconds, seed=a.seed
            ).solve(env)
            metric, moves, duration, status = (
                result.final_blocking,
                result.move_count,
                result.runtime_sec,
                result.best_reason,
            )
            if status != "from_cleared" or moves != len(df):
                metric = float("inf")
        elif method == "gurobi":
            try:
                from stockyard.baselines.gurobi import solve_scenario_sequential_gurobi
            except ImportError as e:
                raise RuntimeError(
                    "Install requirements-gurobi.txt and configure a Gurobi license"
                ) from e
            metric, duration, status, gap = solve_scenario_sequential_gurobi(
                df, "synthetic", "sample", a.max_stack, a.budget_seconds, seed=a.seed
            )
            moves = len(df) if metric >= 0 else None
            if metric < 0:
                metric = None
            status = f"{status}; gap={gap}"
        else:
            checkpoint = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
            if checkpoint["max_stack"] != a.max_stack:
                raise ValueError("Use checkpoint max_stack for PPO evaluation")
            env = make_env(df, a.max_stack)
            model = make_model(env, checkpoint["embed_dim"])
            model.load_state_dict(checkpoint["model_state_dict"])
            _, metric, moves, duration = rollout(model, env, torch.device("cpu"))
            status = "from_cleared" if np.isfinite(metric) else "incomplete"
        records.append(
            dict(
                method=method,
                blocking_pairs=metric,
                moves=moves,
                runtime_seconds=duration,
                status=status,
            )
        )
        records[-1]["completed"] = (
            metric is not None and np.isfinite(metric) and moves == len(df)
        )
    dest = Path(a.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(records)
    result["seed"] = a.seed
    result["max_stack"] = a.max_stack
    result["plate_count"] = len(df)
    result.to_csv(dest, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
