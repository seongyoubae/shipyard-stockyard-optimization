"""Regression checks for public input, feasible search and evaluation reporting."""

import math
import random
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from stockyard.baselines import heuristic as h
from stockyard.baselines.legacy.annealer import Annealer
from stockyard.baselines.legacy.sa_move import make_SA_sample
from stockyard.baselines.sa import replay_action_sequence, run_sa_one_scenario
from stockyard.config import get_cfg
from stockyard.constants import ENCODER_TYPES, ENCODER_ALIASES
from stockyard.data.sample import generate, load, validate
from stockyard.environment.yard import Locating
from stockyard.evaluation.benchmark import make_env, run_rule
from stockyard.models.network import SteelPlateConditionalMLPModel


def config(max_stack=12, max_steps=200):
    return SimpleNamespace(
        max_stack=max_stack,
        max_steps=max_steps,
        OBSERVED_TOP_N_PLATES=10,
        NUM_SUMMARY_STATS_DEEPER=4,
        gamma=0.995,
    )


@pytest.mark.parametrize(
    "case",
    [
        "missing",
        "duplicate_id",
        "duplicate_sequence",
        "blank_id",
        "infinite",
        "fractional_day",
        "negative_weight",
        "outbound_before_inbound",
        "confirmation",
    ],
)
def test_reject_invalid_csv(tmp_path, case):
    df = generate(sources=2, destinations=2, plates_per_source=2)
    if case == "missing":
        df = df.drop(columns="pileseq")
    elif case == "duplicate_id":
        df.loc[1, "markno"] = df.loc[0, "markno"]
    elif case == "duplicate_sequence":
        df.loc[1, "pileseq"] = df.loc[0, "pileseq"]
    elif case == "blank_id":
        df.loc[0, "pileno"] = "   "
    elif case == "infinite":
        df["unitw"] = np.inf
    elif case == "fractional_day":
        df["inbound"] = 1.5
    elif case == "negative_weight":
        df.loc[0, "unitw"] = -1.0
    elif case == "outbound_before_inbound":
        df.loc[0, "outbound"] = 0
    else:
        df["confirm_time"] = 3
    path = tmp_path / "invalid.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load(path)


def test_csv_preserves_string_ids_and_orders_rows(tmp_path):
    df = generate(sources=2, destinations=2, plates_per_source=2)
    df["markno"] = ["001", "002", "003", "004"]
    df["max_stack"] = 999
    df["obs_max"] = 999
    path = tmp_path / "scenario.csv"
    df.iloc[::-1].to_csv(path, index=False)
    loaded = load(path)
    assert loaded.markno.tolist() == ["001", "002", "003", "004"]
    assert "max_stack" not in loaded and "obs_max" not in loaded
    pd.testing.assert_frame_equal(loaded, validate(df))


def test_static_source_capacity_is_checked():
    df = generate(sources=2, destinations=3, plates_per_source=3)
    with pytest.raises(ValueError, match="source stack"):
        make_env(df, 2)


def test_initial_overflow_cannot_be_reported_as_success():
    df = generate(sources=2, destinations=2, plates_per_source=3)
    from stockyard.data.sample import to_plates

    env = Locating(
        inbound_plates=to_plates(df), max_stack=2, min_obstacles=0, max_obstacles=0
    )
    env.reset()
    assert env.overflowed
    _, _, done, info = env.step((0, 0))
    assert done and info["episode_end_reason"] == "overflow"
    assert "final_blocking_metric" not in info and env.crane_move == 0


def test_sa_repaired_sequence_replays_identically():
    df = generate(sources=2, destinations=2, plates_per_source=3)
    cfg = config()
    first = replay_action_sequence(df, cfg, [(29, 29)] * len(df))
    random.seed(999)
    second = replay_action_sequence(df, cfg, [(29, 29)] * len(df))
    verified = replay_action_sequence(df, cfg, first.sequence)
    assert first == second == verified
    assert first.moves == len(df) and first.status == "from_cleared"
    assert math.isfinite(first.blocking_pairs)
    metric, _, moves = run_sa_one_scenario(df, cfg, time_limit_s=1, max_steps=3)
    assert math.isfinite(metric) and moves == len(df)


def test_sa_does_not_score_partial_transfer_as_success():
    df = generate(sources=2, destinations=2, plates_per_source=3)
    result = replay_action_sequence(df, config(max_steps=1), [(0, 0)])
    assert result.status == "max_steps" and math.isinf(result.blocking_pairs)


@pytest.mark.parametrize("encoder", [*ENCODER_TYPES, *ENCODER_ALIASES])
def test_supported_encoder_executes(encoder):
    torch.set_num_threads(1)
    env = make_env(generate(sources=2, destinations=2, plates_per_source=2), 12)
    state = env.reset()
    src, dst = env.get_masks()
    model = SteelPlateConditionalMLPModel(
        embed_dim=16,
        num_actor_layers=2,
        num_critic_layers=2,
        actor_init_std=0.01,
        critic_init_std=1.0,
        pile_feature_dim=env.actual_pile_feature_dim,
        num_heads=4,
        encoder_type=encoder,
    )
    with torch.no_grad():
        action, log, value, _ = model.act_batch(state[None], src[None], dst[None])
        evaluated_log, evaluated_value, entropy = model.evaluate(
            state[None],
            src[None],
            dst[None],
            action,
        )
    assert src[action[0, 0]] and dst[action[0, 1]]
    assert torch.isfinite(value).all() and torch.isfinite(entropy).all()
    assert torch.allclose(log, evaluated_log.flatten(), atol=1e-5)
    assert torch.allclose(value, evaluated_value.flatten(), atol=1e-5)
    assert model.encoder_type == ENCODER_ALIASES.get(encoder, encoder)


def test_bad_encoder_fails_explicitly():
    with pytest.raises(ValueError, match="Unknown encoder"):
        SteelPlateConditionalMLPModel(
            16,
            2,
            2,
            0.01,
            1.0,
            pile_feature_dim=21,
            num_heads=4,
            encoder_type="misspelled-gru",
        )


def test_research_checkpoint_boolean_flags(monkeypatch):
    monkeypatch.setattr("sys.argv", ["research_train", "--load_model"])
    assert get_cfg().load_model is True
    monkeypatch.setattr("sys.argv", ["research_train", "--no-load_model"])
    assert get_cfg().load_model is False


def test_original_heuristic_evaluation_helper():
    df = generate(sources=2, destinations=2, plates_per_source=2)
    df["scenario_id"] = "synthetic"
    agent = h.ConfigurableHeuristicAgent(
        h.source_P1_EDD,
        h.dest_PH1_MOD,
        {"max_tier": 12, "max_bay": 2},
    )
    detail, summary = h.run_evaluation_for_strategy(
        "edd-mod",
        agent,
        agent.env_config,
        ["synthetic"],
        df,
        config(),
        "synthetic",
    )
    assert detail.iloc[0].strategy_display_name == "edd-mod"
    assert summary["Avg_Total_Moves"] == len(df)
    detail, _ = h.run_evaluation_for_strategy(
        "edd-mod",
        agent,
        agent.env_config,
        ["synthetic"],
        df,
        config(max_steps=1),
        "synthetic",
    )
    assert not detail.iloc[0].completed and math.isinf(detail.iloc[0].final_reversals)


def test_legacy_annealer_runtime_tracking():
    random.seed(42)
    sequence, sources, destinations = make_SA_sample(1, 1)
    solver = Annealer(sequence, sources, destinations)
    solver.steps = 2
    _, energy = solver.anneal()
    assert math.isfinite(energy) and len(solver.time_list) == 2


def test_barge_source_can_exceed_destination_height():
    df = generate(sources=1, destinations=4, plates_per_source=8)
    metric, moves, _, status = run_rule(
        df,
        2,
        h.source_P1_EDD,
        h.dest_PH1_MOD,
        42,
    )
    assert status == "from_cleared" and moves == len(df) and math.isfinite(metric)
