import os
import math
import copy
import random
import numpy as np
import pandas as pd
from time import time

from stockyard.config import get_cfg
from stockyard.environment.yard import Locating, Plate
from stockyard.baselines.heuristic import _calculate_total_blocking


def build_schedule_from_df(df):
    schedule = []
    for _, r in df.iterrows():
        p = Plate(
            id=r["markno"],
            inbound=r["inbound"],
            outbound=r["outbound"],
            unitw=r.get("unitw", 10.0),
        )
        p.from_pile = str(r["pileno"])
        p.topile = str(r["topile"])
        schedule.append(p)
    return schedule


def make_env(df, cfg):
    scenario_obs = int(df.get("obs_max", pd.Series([0])).iloc[0])
    scenario_max_stack = int(df.get("max_stack", pd.Series([cfg.max_stack])).iloc[0])

    schedule = build_schedule_from_df(df)

    env = Locating(
        max_stack=scenario_max_stack,
        min_obstacles=scenario_obs,
        max_obstacles=scenario_obs,
        inbound_plates=schedule,
        observed_top_n_plates=cfg.OBSERVED_TOP_N_PLATES,
        num_summary_stats_deeper=cfg.NUM_SUMMARY_STATS_DEEPER,
        gamma=cfg.gamma,
        max_steps=cfg.max_steps,
    )
    return env


def get_random_valid_action(env):
    source_mask, dest_mask = env.get_masks()

    valid_sources = [
        i for i in range(len(env.from_keys))
        if bool(source_mask[i])
    ]
    valid_dests = [
        j for j in range(len(env.to_keys))
        if bool(dest_mask[j])
    ]

    if not valid_sources or not valid_dests:
        return None

    return random.choice(valid_sources), random.choice(valid_dests)


def evaluate_action_sequence(df, cfg, action_sequence):
    env = make_env(df, cfg)
    env.reset(shuffle_schedule=False)

    max_eval_steps = max(cfg.max_steps * 5, len(df) * 5)

    seq_idx = 0
    step = 0

    while True:
        step += 1

        if len(env.pending_inbound_events) == 0 and all(not env.plates.get(k) for k in env.from_keys):
            break

        if step > max_eval_steps:
            break

        source_mask, dest_mask = env.get_masks()
        valid_source_exists = bool(source_mask[:len(env.from_keys)].any())
        valid_dest_exists = bool(dest_mask[:len(env.to_keys)].any())

        if not valid_source_exists:
            _, _, done, _ = env.step((0, 0))
            if done:
                break
            continue

        if not valid_dest_exists:
            break

        if seq_idx < len(action_sequence):
            action = action_sequence[seq_idx]
            seq_idx += 1
        else:
            action = get_random_valid_action(env)

        if action is None:
            break

        from_idx, to_idx = action

        if not (
            0 <= from_idx < len(env.from_keys)
            and 0 <= to_idx < len(env.to_keys)
            and bool(source_mask[from_idx])
            and bool(dest_mask[to_idx])
        ):
            action = get_random_valid_action(env)

        if action is None:
            break

        _, _, done, _ = env.step(action)
        if done:
            break

    final_reversal = _calculate_total_blocking(env.plates, env)
    return final_reversal


def make_initial_sequence(df, cfg):
    env = make_env(df, cfg)
    env.reset(shuffle_schedule=False)

    sequence = []
    max_len = len(df) + int(df.get("obs_max", pd.Series([0])).iloc[0]) + 100

    for _ in range(max_len):
        action = get_random_valid_action(env)

        if action is None:
            if len(env.pending_inbound_events) > 0:
                _, _, done, _ = env.step((0, 0))
                if done:
                    break
                continue
            break

        sequence.append(action)

        _, _, done, _ = env.step(action)
        if done:
            break

    return sequence


def mutate_sequence(sequence, n_source, n_dest):
    new_seq = copy.deepcopy(sequence)

    if not new_seq:
        return new_seq

    mutation_type = random.choice(["change", "swap"])

    if mutation_type == "change":
        idx = random.randrange(len(new_seq))
        new_seq[idx] = (
            random.randrange(n_source),
            random.randrange(n_dest),
        )

    else:
        if len(new_seq) >= 2:
            i, j = random.sample(range(len(new_seq)), 2)
            new_seq[i], new_seq[j] = new_seq[j], new_seq[i]

    return new_seq


def run_sa_one_scenario(
    df,
    cfg,
    time_limit_s=3600,
    max_steps=10_000_000,
    tmax=100.0,
    tmin=0.00026,
    seed=233423
):
    random.seed(seed)
    np.random.seed(seed)

    base_env = make_env(df, cfg)
    base_env.reset(shuffle_schedule=False)

    n_source = len(base_env.from_keys)
    n_dest = len(base_env.to_keys)

    current_seq = make_initial_sequence(df, cfg)
    current_energy = evaluate_action_sequence(df, cfg, current_seq)

    best_seq = copy.deepcopy(current_seq)
    best_energy = current_energy

    start_time = time()
    step = 0

    while True:
        step += 1
        elapsed = time() - start_time

        if elapsed >= time_limit_s:
            break

        if step > max_steps:
            break

        progress = min(elapsed / time_limit_s, 1.0)
        temperature = tmax * ((tmin / tmax) ** progress)

        candidate_seq = mutate_sequence(current_seq, n_source, n_dest)
        candidate_energy = evaluate_action_sequence(df, cfg, candidate_seq)

        dE = candidate_energy - current_energy

        if dE < 0 or math.exp(-dE / max(temperature, 1e-12)) > random.random():
            current_seq = candidate_seq
            current_energy = candidate_energy

            if current_energy < best_energy:
                best_seq = copy.deepcopy(current_seq)
                best_energy = current_energy

        if step % 5000 == 0:
            print(
                f"    step={step:>7}, "
                f"elapsed={elapsed:.1f}s, "
                f"current={current_energy:.1f}, "
                f"best={best_energy:.1f}"
            )

    elapsed = time() - start_time

    return best_energy, elapsed, len(best_seq)
