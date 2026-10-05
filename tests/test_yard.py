import copy
import math
import torch
from stockyard.data.sample import generate, to_plates
from stockyard.environment.yard import Locating
from stockyard.evaluation.benchmark import make_env, run_rule
from stockyard.baselines import heuristic as h
from stockyard.training.ppo import advantages, make_model
from stockyard.evaluation.policy import evaluate_policy


def test_pairs_ties_and_order():
    env = make_env(generate(), 12)
    p = to_plates(generate(sources=1, destinations=1, plates_per_source=3))
    p[0].outbound = 1
    p[1].outbound = 3
    p[2].outbound = 3
    assert env._get_total_blocking_pairs(p) == 2
    assert env._get_total_blocking_pairs(list(reversed(p))) == 0


def test_masks_top_only_and_capacity():
    df = generate(sources=2, destinations=2, plates_per_source=2)
    env = make_env(df, 2)
    env.reset()
    before = copy.deepcopy(env.plates[env.from_keys[0]])
    _, _, done, info = env.step((0, 0))
    assert env.plates[env.to_keys[0]][-1].id == before[-1].id
    env.step((0, 0))
    src, dst = env.get_masks()
    assert not src[0] and not dst[0]
    assert not src[len(env.from_keys) :].any() and not dst[len(env.to_keys) :].any()


def test_dynamic_idle_then_finish():
    df = generate(seed=8, dynamic=True)
    score, moves, _, status = run_rule(
        df, 12, h.source_random_selection, h.dest_random_stacking, 3
    )
    assert moves == len(df) and math.isfinite(score) and status == "from_cleared"


def test_invalid_action_and_step_limit():
    env = make_env(generate(), 12)
    env.reset()
    _, _, done, info = env.step((29, 29))
    assert done and "error" in info
    env = Locating(
        inbound_plates=to_plates(generate()),
        max_stack=12,
        min_obstacles=0,
        max_obstacles=0,
        max_steps=1,
    )
    env.reset()
    _, _, done, info = env.step((0, 0))
    assert done and info["episode_end_reason"] == "max_steps"


def test_gae_terminal_does_not_bootstrap():
    out = advantages([1.0, 2.0], [0.5, 0.5], [True, True], 100.0)
    assert torch.allclose(out, torch.tensor([0.5, 1.5]))


def test_network_and_eval_adapter():
    torch.set_num_threads(1)
    env = make_env(generate(), 12)
    state = env.reset()
    model = make_model(env)
    src, dst = env.get_masks()
    with torch.no_grad():
        a, log, val, _ = model.act_batch(state[None], src[None], dst[None])
    assert src[a[0, 0]] and dst[a[0, 1]] and torch.isfinite(log).all()
    result = evaluate_policy(model, [env], torch.device("cpu"), return_blocked=True)
    assert len(result) == 3 and math.isfinite(result[1])
