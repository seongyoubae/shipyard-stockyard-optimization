import numpy as np
import pandas as pd
from time import time
import copy
import random
import os
import itertools

try:
    from stockyard.config import get_cfg
    from stockyard.environment.yard import Locating, Plate
    from stockyard.models.network import MAX_SOURCE, MAX_DEST
except ImportError as e:
    print(f"[오류] 필요한 모듈 임포트 실패: {e}")
    print("env.py, cfg.py, data.py, network.py 파일이 필요합니다.")
    exit()


# ==============================================================================
# 섹션 1: 공통 계산 함수
# ==============================================================================
def _calculate_blocking_pairs(pile):
    if len(pile) <= 1:
        return 0
    total_blocking_pairs = 0
    outbounds = [p.outbound for p in pile]
    for i in range(len(outbounds)):
        for j in range(i + 1, len(outbounds)):
            if outbounds[i] < outbounds[j]:
                total_blocking_pairs += 1
    return total_blocking_pairs


def _calculate_total_blocking(plates_state, env):
    total_blocking = 0
    for key in env.to_keys:
        total_blocking += _calculate_blocking_pairs(plates_state.get(key, []))
    return total_blocking


def _get_index_from_key(env, key, is_dest):
    key_list = env.to_keys if is_dest else env.from_keys
    try:
        return key_list.index(key)
    except ValueError:
        return None


def _get_earliest_outbound_per_bay(plates_state, env):
    earliest_outbound_values = {}
    all_plates_in_state = list(itertools.chain.from_iterable(plates_state.values()))
    id_to_outbound = {
        p.id: p.outbound
        for p in all_plates_in_state
        if hasattr(p, "id") and hasattr(p, "outbound")
    }
    for key in env.to_keys:
        pile = plates_state.get(key, [])
        min_outbound = float("inf")
        for plate_obj in pile:
            outbound = id_to_outbound.get(plate_obj.id, float("inf"))
            if outbound < min_outbound:
                min_outbound = outbound
        earliest_outbound_values[key] = min_outbound
    return earliest_outbound_values


# ==============================================================================
# 섹션 2: Source Rule
# ==============================================================================

# --- [Naive & Strong Source Rules] ---
def source_random_selection(env, env_config):
    from_keys_ordered = list(env.from_keys)
    random.shuffle(from_keys_ordered)
    for key in from_keys_ordered:
        if env.plates.get(key):
            return key
    return None


def source_SOP_ShortestOutbound(env, env_config):
    """
    [핵심 로직] 야드 전체에서 가장 빨리 나가야 할(outbound 최소) 놈을 찾아서
    그놈이 들어있는 파일부터 파내기 시작함.
    """
    all_plates = [p for pile in env.plates.values() for p in pile]
    if not all_plates: return None

    # 1. 야드 전체에서 가장 급한 놈의 outbound 값 찾기
    target_outbound = min(p.outbound for p in all_plates)

    # 2. 그놈을 포함하는 파일을 찾아서 반환
    for key in env.from_keys:
        pile = env.plates.get(key)
        if pile and any(p.outbound == target_outbound for p in pile):
            return key

    return source_random_selection(env, env_config)

def source_MBT_MostBlockingTop(env, env_config):
    """
    [핵심 로직] 각 파일의 맨 위 강재(Top)가 가로막고 있는 개수를 세어,
    가장 많이 가로막고 있는 파일부터 공략하는 직관적 휴리스틱.
    """
    best_key = None
    max_blocking_top = -1

    for key in env.from_keys:
        pile = env.plates.get(key)
        if pile and len(pile) > 1:
            top_plate = pile[-1]
            # 맨 위 강재보다 먼저 나가야 할 강재(Blocking) 개수 세기
            blocking_count = sum(1 for p in pile[:-1] if p.outbound < top_plate.outbound)

            if blocking_count > max_blocking_top:
                max_blocking_top = blocking_count
                best_key = key

    # 가로막는 놈이 없으면 그냥 랜덤하게 하나 선택
    return best_key if best_key else source_random_selection(env, env_config)

def source_O3_most_blocking(env, env_config):
    best_key = None
    max_blocking = -1
    for key in env.from_keys:
        pile = env.plates.get(key)
        if pile:
            blocking_pairs_in_pile = _calculate_blocking_pairs(pile)
            if blocking_pairs_in_pile > max_blocking:
                max_blocking = blocking_pairs_in_pile
                best_key = key
    if best_key is not None:
        return best_key
    for key in env.from_keys:
        if env.plates.get(key):
            return key
    return None


# --- [Practical Source Rules (현업 직관형)] ---
def source_P1_EDD(env, env_config):
    best_key = None
    min_outbound = float("inf")
    for key in env.from_keys:
        pile = env.plates.get(key)
        if pile:
            top_plate = pile[-1]
            if hasattr(top_plate, "outbound") and top_plate.outbound < min_outbound:
                min_outbound = top_plate.outbound
                best_key = key
    return best_key


def source_P2_FIFO(env, env_config):
    best_key = None
    min_inbound = float("inf")
    for key in env.from_keys:
        pile = env.plates.get(key)
        if pile:
            top_plate = pile[-1]
            if hasattr(top_plate, "inbound") and top_plate.inbound < min_inbound:
                min_inbound = top_plate.inbound
                best_key = key
    return best_key


def source_P3_HPF(env, env_config):
    candidates = []
    max_height = -1

    for key in env.from_keys:
        pile = env.plates.get(key)
        if not pile:
            continue

        h = len(pile)
        if h > max_height:
            max_height = h
            candidates = [key]
        elif h == max_height:
            candidates.append(key)

    if not candidates:
        return None

    return min(
        candidates,
        key=lambda k: env.plates[k][-1].outbound
    )


# --- [💡 추가: Inbound Buffer (바지선 하역장) 전용 Source Rule] ---
def source_inbound_buffer(env, env_config):
    """바지선(from_00 등)에서 무조건 순차적으로 꺼냄"""
    for key in env.from_keys:
        if env.plates.get(key):
            return key
    return None


# ==============================================================================
# 섹션 3: Destination Rule
# ==============================================================================

# --- [Naive & Strong Destination Rules] ---
def dest_random_stacking(yard_state, blocking_plate_info, pickup_pos, env_config, env):
    pickup_key = env.from_keys[pickup_pos[0]]
    possible_dest_keys = [
        key for key in env.to_keys
        if key != pickup_key and len(yard_state.get(key, [])) < env_config["max_tier"]
    ]
    if not possible_dest_keys: return None
    return _get_index_from_key(env, random.choice(possible_dest_keys), is_dest=True)


def dest_OH1_minimize_final_blocking(yard_state, blocking_plate_info, pickup_pos, env_config, env):
    pickup_key = env.from_keys[pickup_pos[0]]
    current_plate = Plate(
        id=blocking_plate_info["id"],
        inbound=blocking_plate_info["inbound"],
        outbound=blocking_plate_info["outbound"],
        unitw=blocking_plate_info["unitw"],
    )

    best_dest_key = None
    best_score = (float("inf"), float("inf"), "")
    max_tier = env_config["max_tier"]

    for dest_key in env.to_keys:
        if dest_key == pickup_key:
            continue

        current_pile = yard_state.get(dest_key, [])
        if len(current_pile) >= max_tier:
            continue

        temp_pile = current_pile + [current_plate]
        resulting_blocking = _calculate_blocking_pairs(temp_pile)

        score = (resulting_blocking, len(current_pile), dest_key)
        if score < best_score:
            best_score = score
            best_dest_key = dest_key

    if best_dest_key is None:
        return None

    return _get_index_from_key(env, best_dest_key, is_dest=True)


# --- [Practical Destination Rules (현업 직관형)] ---
def dest_PH1_MOD(yard_state, blocking_plate_info, pickup_pos, env_config, env):
    pickup_key = env.from_keys[pickup_pos[0]]
    Di = blocking_plate_info["outbound"]
    max_tier = env_config["max_tier"]

    safe_candidates = []
    fallback_candidates = []

    for key in env.to_keys:
        if key == pickup_key:
            continue

        pile = yard_state.get(key, [])
        if len(pile) >= max_tier:
            continue

        if not pile:
            safe_candidates.append((float("inf"), 0, key))
            continue

        outbounds = [p.outbound for p in pile]
        min_outbound = min(outbounds)

        added_reversals = sum(1 for ob in outbounds if ob < Di)

        if added_reversals == 0:
            gap = min_outbound - Di
            safe_candidates.append((gap, len(pile), key))
        else:
            rev_gap = Di - min_outbound
            fallback_candidates.append((added_reversals, rev_gap, len(pile), key))

    if safe_candidates:
        safe_candidates.sort(key=lambda x: (x[0], x[1]))
        return _get_index_from_key(env, safe_candidates[0][2], is_dest=True)

    if fallback_candidates:
        fallback_candidates.sort(key=lambda x: (x[0], x[1], x[2]))
        return _get_index_from_key(env, fallback_candidates[0][3], is_dest=True)

    return None


def dest_PH2_SDF(yard_state, blocking_plate_info, pickup_pos, env_config, env):
    pickup_key = env.from_keys[pickup_pos[0]]
    max_tier = env_config["max_tier"]

    best_key = None
    min_height = float("inf")

    for key in env.to_keys:
        if key == pickup_key: continue
        pile = yard_state.get(key, [])
        if len(pile) < max_tier:
            if len(pile) < min_height:
                min_height = len(pile)
                best_key = key

    return _get_index_from_key(env, best_key, is_dest=True) if best_key else None


def dest_PH3_MCD(yard_state, blocking_plate_info, pickup_pos, env_config, env):
    pickup_key = env.from_keys[pickup_pos[0]]
    Di = blocking_plate_info["outbound"]
    max_tier = env_config["max_tier"]

    best_key = None
    min_abs_gap = float("inf")

    for key in env.to_keys:
        if key == pickup_key: continue
        pile = yard_state.get(key, [])
        if len(pile) < max_tier:
            if not pile:
                abs_gap = 9999
            else:
                top_outbound = pile[-1].outbound
                abs_gap = abs(top_outbound - Di)

            if abs_gap < min_abs_gap:
                min_abs_gap = abs_gap
                best_key = key

    if best_key is None:
        for k in env.to_keys:
            if k != pickup_key and len(yard_state.get(k, [])) < max_tier:
                return _get_index_from_key(env, k, is_dest=True)

    return _get_index_from_key(env, best_key, is_dest=True) if best_key else None


# --- [💡 추가: Inbound Buffer (바지선 하역장) 전용 Destination Rules] ---
def dest_inbound_random_stacking(yard_state, blocking_plate_info, pickup_pos, env_config, env):
    possible_dest_keys = [
        key for key in env.to_keys
        if len(yard_state.get(key, [])) < env_config["max_tier"]
    ]
    if not possible_dest_keys:
        return None
    return _get_index_from_key(env, random.choice(possible_dest_keys), is_dest=True)


def dest_inbound_minimize_conflicts(yard_state, blocking_plate_info, pickup_pos, env_config, env):
    current_dwell = blocking_plate_info["outbound"] - blocking_plate_info["inbound"]
    max_tier = env_config["max_tier"]

    candidates = []
    empty_candidates = []

    for key in env.to_keys:
        pile = yard_state.get(key, [])
        if len(pile) >= max_tier:
            continue
        if not pile:
            empty_candidates.append(key)
            continue
        Es = min([p.outbound - p.inbound for p in pile])
        if Es >= current_dwell:
            candidates.append(key)

    if candidates:
        return _get_index_from_key(env, random.choice(candidates + empty_candidates), is_dest=True)

    possible_dest_keys = [
        key for key in env.to_keys
        if len(yard_state.get(key, [])) < max_tier
    ]
    if not possible_dest_keys:
        return None
    return _get_index_from_key(env, random.choice(possible_dest_keys), is_dest=True)


def dest_inbound_delay_conflicts(yard_state, blocking_plate_info, pickup_pos, env_config, env):
    current_dwell = blocking_plate_info["outbound"] - blocking_plate_info["inbound"]
    max_tier = env_config["max_tier"]

    valid_candidates = []
    empty_candidates = []
    fallback_candidates = []

    for key in env.to_keys:
        pile = yard_state.get(key, [])
        if len(pile) >= max_tier:
            continue
        if not pile:
            empty_candidates.append(key)
            continue

        Es = min([p.outbound - p.inbound for p in pile])
        if Es >= current_dwell:
            valid_candidates.append(key)
        fallback_candidates.append((key, Es))

    if valid_candidates:
        return _get_index_from_key(env, random.choice(valid_candidates + empty_candidates), is_dest=True)
    if empty_candidates:
        return _get_index_from_key(env, random.choice(empty_candidates), is_dest=True)
    if not fallback_candidates:
        return None

    max_Es = max(es for _, es in fallback_candidates)
    best_keys = [key for key, es in fallback_candidates if es == max_Es]
    return _get_index_from_key(env, random.choice(best_keys), is_dest=True)


def dest_inbound_flexibility_optimization(yard_state, blocking_plate_info, pickup_pos, env_config, env):
    current_dwell = blocking_plate_info["outbound"] - blocking_plate_info["inbound"]
    max_tier = env_config["max_tier"]

    valid_candidates = []
    empty_candidates = []
    fallback_candidates = []

    for key in env.to_keys:
        pile = yard_state.get(key, [])
        if len(pile) >= max_tier:
            continue
        if not pile:
            empty_candidates.append(key)
            continue

        Es = min([p.outbound - p.inbound for p in pile])
        dF = current_dwell - Es

        if dF <= 0:
            valid_candidates.append((key, dF))
        fallback_candidates.append((key, dF))

    if valid_candidates:
        max_dF = max(df for _, df in valid_candidates)
        best_keys = [key for key, df in valid_candidates if df == max_dF]
        return _get_index_from_key(env, random.choice(best_keys + empty_candidates), is_dest=True)

    if empty_candidates:
        return _get_index_from_key(env, random.choice(empty_candidates), is_dest=True)
    if not fallback_candidates:
        return None

    min_dF = min(df for _, df in fallback_candidates)
    best_keys = [key for key, df in fallback_candidates if df == min_dF]
    return _get_index_from_key(env, random.choice(best_keys), is_dest=True)


# ==============================================================================
# 섹션 4: Agent
# ==============================================================================
class ConfigurableHeuristicAgent:
    def __init__(self, source_rule, destination_rule, env_config):
        self.source_rule = source_rule
        self.destination_rule = destination_rule
        self.env_config = env_config

    def get_action(self, env):
        source_key = self.source_rule(env, self.env_config)
        if source_key is None: return None
        pickup_pile = env.plates.get(source_key, [])
        if not pickup_pile: return None
        source_idx = _get_index_from_key(env, source_key, is_dest=False)
        if source_idx is None: return None

        pickup_pos = (source_idx, len(pickup_pile) - 1)
        blocking_plate_info = pickup_pile[-1].__dict__

        dest_idx = self.destination_rule(env.plates, blocking_plate_info, pickup_pos, self.env_config, env)
        if dest_idx is None: return None
        return pickup_pos[0], dest_idx


# ==============================================================================
# 섹션 5: Evaluation
# ==============================================================================
def build_schedule_from_df(current_problem_df):
    schedule = []
    for _, r in current_problem_df.iterrows():
        p = Plate(
            id=r["markno"],
            inbound=r["inbound"],
            outbound=r["outbound"],
            unitw=r["unitw"],
        )
        p.from_pile = str(r["pileno"])
        p.topile = str(r["topile"])
        schedule.append(p)
    return schedule


def should_evaluate_strategy_on_scenario(strategy_name, scenario_type):
    # 💡 [핵심 패치] 바지선(barge_inbound) 및 기존 IB 모드일 때만 IB_ 룰 실행!
    if scenario_type in ["inbound_buffer", "barge_inbound"]:
        return strategy_name.startswith("IB_")

    # 일반 재취급(Reshuffling) 상황에서는 IB_ 룰 실행 안 함!
    return not strategy_name.startswith("IB_")


def run_evaluation_for_strategy(strategy_name, agent, env_config, scenarios, df_all_data, cfg, eval_file_name):
    print(f"\n--- Evaluating Strategy {strategy_name} on File: {eval_file_name} ---")
    results_per_scenario = []

    for i, scenario_id in enumerate(scenarios):
        current_problem_df = df_all_data[df_all_data["scenario_id"] == scenario_id].copy()

        distribution = current_problem_df.get("distribution", pd.Series(["UNKNOWN"])).iloc[0]
        scenario_type = current_problem_df.get("scenario_type", pd.Series(["reshuffling"])).iloc[0]
        test_group = current_problem_df.get("test_group", pd.Series(["UNKNOWN"])).iloc[0]

        if not should_evaluate_strategy_on_scenario(strategy_name, scenario_type):
            continue

        print(f"  > {strategy_name} | {scenario_id} ({i + 1}/{len(scenarios)})", end="\r", flush=True)

        scenario_num_plates = len(current_problem_df)
        schedule = build_schedule_from_df(current_problem_df)

        scenario_obs = int(current_problem_df.get("obs_max", pd.Series([0])).iloc[0])
        scenario_max_stack = int(current_problem_df.get("max_stack", pd.Series([cfg.max_stack])).iloc[0])

        scenario_env_config = dict(env_config)
        scenario_env_config["max_tier"] = scenario_max_stack
        agent.env_config = scenario_env_config

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
        env.reset(shuffle_schedule=False)
        initial_blocking = _calculate_total_blocking(env.plates, env)

        start_time = time()
        step = 0
        max_eval_steps = max(cfg.max_steps * 5, scenario_num_plates * 5)

        while True:
            step += 1
            if len(env.pending_inbound_events) == 0 and all(not env.plates.get(k) for k in env.from_keys): break
            if step > max_eval_steps: break

            action = agent.get_action(env)
            if action is None:
                if len(env.pending_inbound_events) > 0:
                    _, _, done, _ = env.step((0, 0))
                    if done: break
                    continue
                break

            _, _, done, _ = env.step(action)
            if done: break

        elapsed_time = time() - start_time
        final_reversals = _calculate_total_blocking(env.plates, env)

        results_per_scenario.append({
            "scenario_id": scenario_id,
            "distribution": distribution,
            "scenario_type": scenario_type,
            "test_group": test_group,
            "strategy": strategy_name,
            "num_plates_in_scenario": scenario_num_plates,
            "final_reversals": final_reversals,
            "moves": env.crane_move,
            "time_s": elapsed_time,
            "eval_steps": step,
            "initial_reversals": initial_blocking,
            "strategy_display_name": STRATEGY_DISPLAY_NAME.get(strategy_name, strategy_name),
        })

    final_results_df = pd.DataFrame(results_per_scenario)
    if final_results_df.empty: return final_results_df, None

    avg_results = {
        "File_Name": eval_file_name,
        "Strategy": strategy_name,
        "Avg_Plates": final_results_df["num_plates_in_scenario"].mean(),
        "Avg_Reversals": final_results_df["final_reversals"].mean(),
        "Avg_Total_Moves": final_results_df["moves"].mean(),
        "Avg_Time_s": final_results_df["time_s"].mean(),
    }
    print(f"\n  > Strategy {strategy_name}: Reversals={avg_results['Avg_Reversals']:.2f}")
    return final_results_df, avg_results


def summarize_by_group(all_detail_df):
    if all_detail_df.empty: return pd.DataFrame()
    summary = (
        all_detail_df
        .groupby(["distribution", "scenario_type", "test_group", "strategy"])
        .agg(
            scenarios=("scenario_id", "nunique"),
            avg_plates=("num_plates_in_scenario", "mean"),
            avg_reversals=("final_reversals", "mean"),
            avg_moves=("moves", "mean"),
            avg_time_s=("time_s", "mean"),
        ).reset_index()
    )
    return summary.sort_values(by=["distribution", "scenario_type", "test_group", "avg_reversals"]).reset_index(
        drop=True)


# ==============================================================================
# 섹션 6: Main
# ==============================================================================
