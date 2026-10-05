"""Simulated annealing over feasible, replayable static transfer sequences."""

from dataclasses import dataclass
import math
import random
from time import perf_counter

from stockyard.data.sample import to_plates, validate
from stockyard.environment.yard import Locating


@dataclass
class SequenceResult:
    blocking_pairs: float
    moves: int
    status: str
    sequence: list[tuple[int, int]]


def make_env(df, cfg):
    return Locating(
        inbound_plates=to_plates(validate(df)),
        max_stack=cfg.max_stack,
        min_obstacles=0,
        max_obstacles=0,
        observed_top_n_plates=cfg.OBSERVED_TOP_N_PLATES,
        num_summary_stats_deeper=cfg.NUM_SUMMARY_STATS_DEEPER,
        gamma=cfg.gamma,
        max_steps=cfg.max_steps,
    )


def get_random_valid_action(env, rng=None):
    rng = rng or random
    source_mask, dest_mask = env.get_masks()
    sources = source_mask.nonzero().flatten().tolist()
    destinations = dest_mask.nonzero().flatten().tolist()
    if not sources or not destinations:
        return None
    return rng.choice(sources), rng.choice(destinations)


def replay_action_sequence(df, cfg, action_sequence):
    """Repair invalid genes deterministically and retain the actual executed plan.

    Incomplete transfers receive infinite cost, so clearing only part of the yard
    cannot appear to improve the blocking objective.
    """
    env = make_env(df, cfg)
    env.reset(shuffle_schedule=False)
    executed = []
    position = 0
    info = {}
    for _ in range(env.max_steps):
        src, dst = env.get_masks()
        if not src.any():
            action = (0, 0)  # Let the environment advance an idle arrival period.
        elif not dst.any() or env.overflowed:
            action = (0, 0)  # Let the environment report capacity failure.
        else:
            if position < len(action_sequence):
                action = tuple(action_sequence[position])
            else:
                action = (int(src.nonzero()[0]), int(dst.nonzero()[0]))
            position += 1
            i, j = action
            if not (0 <= i < len(src) and src[i] and 0 <= j < len(dst) and dst[j]):
                action = (int(src.nonzero()[0]), int(dst.nonzero()[0]))
            executed.append(action)
        _, _, done, info = env.step(action)
        if done:
            break
    status = info.get("episode_end_reason") or "incomplete"
    complete = status == "from_cleared" and env.crane_move == len(df)
    metric = info.get("final_blocking_metric", math.inf) if complete else math.inf
    return SequenceResult(metric, env.crane_move, status, executed)


def evaluate_action_sequence(df, cfg, action_sequence):
    return replay_action_sequence(df, cfg, action_sequence).blocking_pairs


def make_initial_sequence(df, cfg, rng=None):
    env = make_env(df, cfg)
    env.reset(shuffle_schedule=False)
    sequence = []
    for _ in range(env.max_steps):
        action = get_random_valid_action(env, rng)
        if action is not None:
            sequence.append(action)
        _, _, done, _ = env.step(action or (0, 0))
        if done:
            break
    return sequence


def mutate_sequence(sequence, n_source, n_dest, rng=None):
    rng = rng or random
    candidate = list(sequence)
    if not candidate:
        return candidate
    if rng.choice(["change", "swap"]) == "change":
        index = rng.randrange(len(candidate))
        candidate[index] = rng.randrange(n_source), rng.randrange(n_dest)
    elif len(candidate) >= 2:
        i, j = rng.sample(range(len(candidate)), 2)
        candidate[i], candidate[j] = candidate[j], candidate[i]
    return candidate


def run_sa_one_scenario(
    df,
    cfg,
    time_limit_s=3600,
    max_steps=10_000_000,
    tmax=100.0,
    tmin=0.00026,
    seed=42,
):
    """Return verified objective, search runtime and actual move count."""
    if time_limit_s <= 0 or tmin <= 0 or tmax <= 0 or max_steps < 1:
        raise ValueError("Search budget, iterations and temperatures must be positive")
    df = validate(df)
    if df.inbound.nunique() != 1:
        raise ValueError("The public SA baseline supports static scenarios only")
    started = perf_counter()
    rng = random.Random(seed)
    base_env = make_env(df, cfg)
    initial = make_initial_sequence(df, cfg, rng)
    current = replay_action_sequence(df, cfg, initial)
    if not math.isfinite(current.blocking_pairs):
        raise ValueError(f"SA could not construct a complete plan: {current.status}")
    best = current
    for _ in range(max_steps):
        elapsed = perf_counter() - started
        if elapsed >= time_limit_s:
            break
        progress = min(elapsed / time_limit_s, 1.0)
        temperature = tmax * (tmin / tmax) ** progress
        candidate_sequence = mutate_sequence(
            current.sequence,
            len(base_env.from_keys),
            len(base_env.to_keys),
            rng,
        )
        candidate = replay_action_sequence(df, cfg, candidate_sequence)
        difference = candidate.blocking_pairs - current.blocking_pairs
        if difference <= 0 or rng.random() < math.exp(-difference / temperature):
            current = candidate
            if current.blocking_pairs < best.blocking_pairs:
                best = current
    # Confirm that the saved plan reproduces its reported objective.
    verified = replay_action_sequence(df, cfg, best.sequence)
    if verified.blocking_pairs != best.blocking_pairs or verified.moves != len(df):
        raise RuntimeError("SA best plan failed replay verification")
    return verified.blocking_pairs, perf_counter() - started, verified.moves
