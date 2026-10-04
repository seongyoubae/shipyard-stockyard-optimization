"""
ACO.py

Wu-style ACO with heuristic baseline-centered search
for static 0% inbound steel plate reshuffling evaluation.

기본 실행 목적
-------------
- EXP4_DYNAMIC_INBOUND_RATIO 중 static 0% inbound 조건
- static 0% 조건의 scenario 10개
- repeat 1회
- heuristic greedy baseline 초기해 생성
- baseline sequence를 중심으로 ACO 탐색
- 각 scenario당 최대 30분 탐색
- PyCharm 초록색 Run 버튼으로 바로 실행 가능

중요
----
이 코드는 Wu et al.의 mixed storage + pre-marshalling 문제를 그대로 복제한 코드는 아님.
사용자의 정적 0% 입고 조건에 맞게, Wu-style ACO의 핵심 구조를 변형 적용한 비교 알고리즘임.
"""

from __future__ import annotations

import copy
import math
import os
import random
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from stockyard.data.generator import Plate
from stockyard.environment.yard import Locating


# ============================================================
# 0. 실행 설정
# ============================================================

INPUT_PATH = "data/sample/scenario.csv"
SHEET_NAME = "reshuffle"

OUTPUT_DIR = "aco_results_baseline_centered_30min"

EXPERIMENT_ID = "EXP4_DYNAMIC_INBOUND_RATIO"

BASE_SEED = 42

MAX_STACK = 50
MAX_STEP_BUFFER = 10

# ACO settings
N_ANTS = 30
N_ITERATIONS = 600
RHO = 0.9
Q = 1.0

USE_DYNAMIC_ALPHA_BETA = True

# ============================================================
# 실행 모드
# ============================================================
# "SINGLE_30MIN" : static 0% scenario 10개를 각 30분 제한으로 탐색
# "FULL_600"     : static 0% 전체 scenario × repeat 5회 × 600 iteration
RUN_MODE = "SINGLE_30MIN"

# SINGLE_30MIN 설정
SINGLE_SCENARIOS = 10
SINGLE_REPEATS = 1
SINGLE_ITERATIONS = 120
SINGLE_TIME_LIMIT_SEC = 30 * 60

# FULL_600 설정
FULL_REPEATS = 5
FULL_TIME_LIMIT_SEC = None

# ============================================================
# baseline-centered ACO 설정
# ============================================================

USE_HEURISTIC_BASELINE = True

# 기본 pheromone을 낮게 둬야 baseline pheromone이 실제로 의미를 가짐
TAU0 = 0.01

# baseline sequence에 강한 pheromone 부여
BASELINE_PHEROMONE_WEIGHT = 50.0

# 매 iteration마다 baseline pheromone을 유지
KEEP_BASELINE_PHEROMONE = True

# baseline action을 직접 따라갈 확률
# 초반에는 baseline을 강하게 따르고, 후반에는 조금 더 탐색하도록 낮춤
BASELINE_FOLLOW_PROB_START = 0.98
BASELINE_FOLLOW_PROB_END = 0.9

# 확률 선택 시 baseline action에 추가 bonus
BASELINE_ACTION_BONUS = 100.0

# baseline보다 개선된 sequence에 대한 pheromone 보상
IMPROVEMENT_REWARD_SCALE = 1.0
DEPOSIT_CAP = 100.0

# 출력 설정
PRINT_PROGRESS = True
PRINT_EVERY_ITER = 5
PRINT_EVERY_IMPROVEMENT = True
PRINT_FIRST_ITER_ANT_PROGRESS = True
PRINT_FIRST_ITER_ANT_EVERY = 5

Action = Tuple[int, int]


# ============================================================
# 1. 기본 유틸
# ============================================================

def set_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2 ** 32 - 1))


def load_table(path: str, sheet_name: str = "reshuffle") -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Input file not found: {path}\n"
            f"코드 상단 INPUT_PATH를 확인하세요."
        )

    ext = os.path.splitext(path)[1].lower()

    if ext == ".csv":
        return pd.read_csv(path, low_memory=False)

    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path, sheet_name=sheet_name)

    raise ValueError(f"Unsupported input file extension: {ext}")


def save_outputs(output_dir: str, results: pd.DataFrame, summary: pd.DataFrame) -> None:
    os.makedirs(output_dir, exist_ok=True)

    results_csv = os.path.join(output_dir, "aco_results.csv")
    summary_csv = os.path.join(output_dir, "aco_summary.csv")
    results_xlsx = os.path.join(output_dir, "aco_results.xlsx")

    results.to_csv(results_csv, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(results_xlsx, engine="openpyxl") as writer:
        results.to_excel(writer, sheet_name="aco_results", index=False)
        summary.to_excel(writer, sheet_name="summary", index=False)

    print("\n[Saved]")
    print(f"- {results_csv}")
    print(f"- {summary_csv}")
    print(f"- {results_xlsx}")


# ============================================================
# 2. DataFrame -> Plate 변환
# ============================================================

def row_to_plate(row: pd.Series) -> Any:
    markno = str(row.get("markno", row.name))
    inbound = int(row.get("inbound", 1))
    outbound = int(row.get("outbound", row.get("planned_outbound", inbound + 1)))
    unitw = float(row.get("unitw", 1.0))

    try:
        plate = Plate(id=markno, inbound=inbound, outbound=outbound, unitw=unitw)
    except Exception:
        try:
            plate = Plate(markno=markno, inbound=inbound, outbound=outbound, unitw=unitw)
        except Exception:
            try:
                plate = Plate(markno, inbound, outbound, unitw)
            except Exception:
                plate = SimpleNamespace()

    setattr(plate, "id", getattr(plate, "id", markno))
    setattr(plate, "markno", markno)
    setattr(plate, "inbound", inbound)
    setattr(plate, "outbound", outbound)
    setattr(plate, "unitw", unitw)

    pileno = str(row.get("pileno", row.get("from_pile", ""))).strip()
    from_pile = str(row.get("from_pile", pileno)).strip()
    topile = str(row.get("topile", "")).strip()
    pileseq = int(row.get("pileseq", 0))

    setattr(plate, "pileno", pileno)
    setattr(plate, "from_pile", from_pile)
    setattr(plate, "topile", topile)
    setattr(plate, "pileseq", pileseq)

    return plate


def scenario_to_plates(scenario_df: pd.DataFrame, force_static: bool = True) -> List[Any]:
    df = scenario_df.copy()

    sort_cols = [c for c in ["pileno", "pileseq", "markno"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    plates = [row_to_plate(row) for _, row in df.iterrows()]

    if force_static:
        for p in plates:
            p.inbound = 1

    return plates


def filter_static_0_percent(df: pd.DataFrame, experiment_id: str) -> pd.DataFrame:
    out = df.copy()

    if experiment_id:
        out = out[out["experiment_id"].astype(str).eq(str(experiment_id))].copy()

    static_mask = pd.Series(False, index=out.index)

    if "inbound_pattern" in out.columns:
        static_mask |= out["inbound_pattern"].astype(str).eq("all_initial")

    if "condition_value" in out.columns:
        cond = pd.to_numeric(out["condition_value"], errors="coerce")
        static_mask |= cond.fillna(-9999).eq(0)

    if "inbound" in out.columns and "scenario_id" in out.columns:
        static_by_scenario = out.groupby("scenario_id")["inbound"].transform(
            lambda s: bool(
                (pd.to_numeric(s, errors="coerce").fillna(9999).astype(int) <= 1).all()
            )
        )
        static_mask |= static_by_scenario

    out = out[static_mask].copy()

    if out.empty:
        raise RuntimeError(
            "No static 0% inbound scenarios found. "
            "EXPERIMENT_ID, inbound_pattern, condition_value, inbound column을 확인하세요."
        )

    return out


# ============================================================
# 3. ACO
# ============================================================

@dataclass
class ACOResult:
    final_blocking: int
    runtime_sec: float
    move_count: int
    best_sequence: List[Action]
    best_info: Dict[str, Any]
    best_reason: str
    completed_iterations: int

    baseline_blocking: int
    baseline_move_count: int
    baseline_reason: str
    improvement_over_baseline: int

    n_ants: int
    n_iterations: int
    rho: float
    q: float
    dynamic_alpha_beta: bool
    final_alpha: float
    final_beta: float
    seed: Optional[int]


class WuStyleStaticACO:
    """
    Wu-style ACO transformed for static 0% steel plate reshuffling.

    선택 확률:
        P(i,j) ∝ tau(i,j)^alpha * eta(i,j)^beta

    baseline-centered search:
        - baseline action을 일정 확률로 직접 선택
        - 확률 선택 시 baseline action에 bonus 부여
        - baseline sequence에 강한 pheromone 부여
    """

    def __init__(
        self,
        n_ants: int = 30,
        n_iterations: int = 600,
        rho: float = 0.9,
        q: float = 1.0,
        tau0: float = 0.01,
        tau_min: float = 1e-12,
        elite_weight: float = 2.0,
        max_steps: Optional[int] = None,
        time_limit_sec: Optional[float] = None,
        dynamic_alpha_beta: bool = True,
        seed: Optional[int] = None,
        progress_label: str = "",
        blocking_weight: float = 2.0,
        due_gap_weight: float = 0.1,
        height_weight: float = 0.01,
        baseline_pheromone_weight: float = 50.0,
        improvement_reward_scale: float = 1.0,
        deposit_cap: float = 100.0,
        baseline_action_bonus: float = 100.0,
        baseline_follow_prob_start: float = 0.98,
        baseline_follow_prob_end: float = 0.75,
    ) -> None:
        self.n_ants = int(n_ants)
        self.n_iterations = int(n_iterations)
        self.rho = float(rho)
        self.q = float(q)
        self.tau0 = float(tau0)
        self.tau_min = float(tau_min)
        self.elite_weight = float(elite_weight)
        self.max_steps = max_steps
        self.time_limit_sec = time_limit_sec
        self.dynamic_alpha_beta = bool(dynamic_alpha_beta)
        self.seed = seed
        self.progress_label = str(progress_label)

        self.blocking_weight = float(blocking_weight)
        self.due_gap_weight = float(due_gap_weight)
        self.height_weight = float(height_weight)

        self.baseline_pheromone_weight = float(baseline_pheromone_weight)
        self.improvement_reward_scale = float(improvement_reward_scale)
        self.deposit_cap = float(deposit_cap)

        self.baseline_action_bonus = float(baseline_action_bonus)
        self.baseline_follow_prob_start = float(baseline_follow_prob_start)
        self.baseline_follow_prob_end = float(baseline_follow_prob_end)

        self.rng = random.Random(seed)

    def solve(self, reset_env: Any) -> ACOResult:
        start_time = time.perf_counter()

        n_src = len(reset_env.from_keys)
        n_dst = len(reset_env.to_keys)
        tau = np.full((n_src, n_dst), self.tau0, dtype=np.float64)

        initial_remaining = self._remaining_source_count(reset_env)
        safety_max_steps = int(self.max_steps or (initial_remaining + MAX_STEP_BUFFER))

        # ------------------------------------------------------------
        # 1) heuristic baseline 초기해 생성
        # ------------------------------------------------------------
        if USE_HEURISTIC_BASELINE:
            baseline_env = copy.deepcopy(reset_env)
            baseline_seq, baseline_obj, baseline_info, baseline_reason = self._construct_greedy_baseline(
                env=baseline_env,
                safety_max_steps=safety_max_steps,
            )
        else:
            baseline_seq = []
            baseline_obj = math.inf
            baseline_info = {}
            baseline_reason = "no_baseline"

        baseline_blocking = int(baseline_obj) if math.isfinite(baseline_obj) else 10**12

        global_best_seq: List[Action] = list(baseline_seq)
        global_best_obj = float(baseline_blocking)
        global_best_info: Dict[str, Any] = dict(baseline_info)
        global_best_reason = str(baseline_reason)

        completed_iterations = 0
        final_alpha = 0.0
        final_beta = 0.0

        if PRINT_PROGRESS:
            print(
                f"[ACO Start] {self.progress_label} | "
                f"initial_source_plates={initial_remaining}, "
                f"safety_max_steps={safety_max_steps}, "
                f"ants={self.n_ants}, iterations={self.n_iterations}, "
                f"time_limit={self.time_limit_sec}, tau0={self.tau0}",
                flush=True,
            )
            print(
                f"[Baseline] {self.progress_label} | "
                f"BP={baseline_blocking} "
                f"moves={len(baseline_seq)} "
                f"reason={baseline_reason}",
                flush=True,
            )

        if USE_HEURISTIC_BASELINE and baseline_seq:
            self._deposit(
                tau=tau,
                sequence=baseline_seq,
                obj=baseline_blocking,
                weight=self.baseline_pheromone_weight,
                baseline_obj=baseline_blocking,
                allow_non_improved=True,
            )

        # ------------------------------------------------------------
        # 2) ACO 반복 탐색
        # ------------------------------------------------------------
        for iteration in range(self.n_iterations):
            if self._time_exceeded(start_time):
                break

            iter_start = time.perf_counter()

            alpha_t, beta_t = self._alpha_beta(iteration)
            final_alpha, final_beta = alpha_t, beta_t

            baseline_follow_prob = self._baseline_follow_prob(iteration)

            iter_records = []
            improved_this_iter = False

            for ant_idx in range(self.n_ants):
                if self._time_exceeded(start_time):
                    break

                ant_env = copy.deepcopy(reset_env)

                seq, obj, info, reason = self._construct_solution(
                    env=ant_env,
                    tau=tau,
                    safety_max_steps=safety_max_steps,
                    alpha_t=alpha_t,
                    beta_t=beta_t,
                    baseline_seq=baseline_seq,
                    baseline_follow_prob=baseline_follow_prob,
                )

                iter_records.append((obj, seq, info, reason))

                if obj < global_best_obj:
                    global_best_obj = float(obj)
                    global_best_seq = list(seq)
                    global_best_info = dict(info)
                    global_best_reason = str(reason)
                    improved_this_iter = True

                    if PRINT_PROGRESS and PRINT_EVERY_IMPROVEMENT:
                        elapsed = time.perf_counter() - start_time
                        improvement = baseline_blocking - int(global_best_obj)
                        print(
                            f"[Improve] {self.progress_label} | "
                            f"iter={iteration + 1:04d}/{self.n_iterations:04d} "
                            f"ant={ant_idx + 1:02d}/{self.n_ants:02d} "
                            f"global_best={int(global_best_obj)} "
                            f"baseline={baseline_blocking} "
                            f"improve={improvement} "
                            f"moves={len(global_best_seq)} "
                            f"reason={global_best_reason} "
                            f"elapsed={elapsed:.1f}s",
                            flush=True,
                        )

                if (
                    PRINT_PROGRESS
                    and PRINT_FIRST_ITER_ANT_PROGRESS
                    and iteration == 0
                    and (
                        ant_idx == 0
                        or (ant_idx + 1) % PRINT_FIRST_ITER_ANT_EVERY == 0
                        or (ant_idx + 1) == self.n_ants
                    )
                ):
                    current_best = int(global_best_obj) if math.isfinite(global_best_obj) else -1
                    print(
                        f"[Ant] {self.progress_label} | "
                        f"iter=0001/{self.n_iterations:04d} "
                        f"ant={ant_idx + 1:02d}/{self.n_ants:02d} "
                        f"obj={int(obj)} "
                        f"baseline={baseline_blocking} "
                        f"current_global_best={current_best} "
                        f"follow_prob={baseline_follow_prob:.3f} "
                        f"reason={reason}",
                        flush=True,
                    )

            if not iter_records:
                break

            # pheromone retention
            tau *= max(0.0, min(1.0, self.rho))
            tau = np.maximum(tau, self.tau_min)

            # baseline sequence를 탐색 중심으로 계속 유지
            if USE_HEURISTIC_BASELINE and KEEP_BASELINE_PHEROMONE and baseline_seq:
                self._deposit(
                    tau=tau,
                    sequence=baseline_seq,
                    obj=baseline_blocking,
                    weight=self.baseline_pheromone_weight,
                    baseline_obj=baseline_blocking,
                    allow_non_improved=True,
                )

            # iteration-best
            iter_records.sort(key=lambda x: x[0])
            iter_best_obj, iter_best_seq, _, iter_best_reason = iter_records[0]

            # baseline보다 개선된 경우에만 iteration-best에 pheromone reward
            self._deposit(
                tau=tau,
                sequence=iter_best_seq,
                obj=iter_best_obj,
                weight=1.0,
                baseline_obj=baseline_blocking,
                allow_non_improved=False,
            )

            # baseline보다 개선된 global-best에 elite reward
            if global_best_obj < baseline_blocking:
                self._deposit(
                    tau=tau,
                    sequence=global_best_seq,
                    obj=global_best_obj,
                    weight=self.elite_weight,
                    baseline_obj=baseline_blocking,
                    allow_non_improved=False,
                )

            completed_iterations = iteration + 1

            iter_runtime = time.perf_counter() - iter_start
            elapsed = time.perf_counter() - start_time

            iter_objs = [r[0] for r in iter_records if math.isfinite(r[0])]
            iter_mean = float(np.mean(iter_objs)) if iter_objs else float("nan")

            should_print_iter = (
                iteration == 0
                or improved_this_iter
                or ((iteration + 1) % PRINT_EVERY_ITER == 0)
                or ((iteration + 1) == self.n_iterations)
            )

            if PRINT_PROGRESS and should_print_iter:
                improvement = baseline_blocking - int(global_best_obj)
                print(
                    f"[Iter] {self.progress_label} | "
                    f"{iteration + 1:04d}/{self.n_iterations:04d} | "
                    f"iter_best={int(iter_best_obj)} | "
                    f"iter_mean={iter_mean:.2f} | "
                    f"global_best={int(global_best_obj)} | "
                    f"baseline={baseline_blocking} | "
                    f"improve={improvement} | "
                    f"alpha={alpha_t:.4f} beta={beta_t:.4f} | "
                    f"follow_prob={baseline_follow_prob:.3f} | "
                    f"iter_time={iter_runtime:.2f}s | "
                    f"elapsed={elapsed:.1f}s | "
                    f"reason={iter_best_reason}",
                    flush=True,
                )

        runtime = time.perf_counter() - start_time
        final_blocking = int(global_best_obj) if math.isfinite(global_best_obj) else 10**12
        improvement_over_baseline = int(baseline_blocking - final_blocking)

        return ACOResult(
            final_blocking=final_blocking,
            runtime_sec=float(runtime),
            move_count=len(global_best_seq),
            best_sequence=global_best_seq,
            best_info=global_best_info,
            best_reason=global_best_reason,
            completed_iterations=completed_iterations,

            baseline_blocking=int(baseline_blocking),
            baseline_move_count=len(baseline_seq),
            baseline_reason=str(baseline_reason),
            improvement_over_baseline=int(improvement_over_baseline),

            n_ants=self.n_ants,
            n_iterations=self.n_iterations,
            rho=self.rho,
            q=self.q,
            dynamic_alpha_beta=self.dynamic_alpha_beta,
            final_alpha=float(final_alpha),
            final_beta=float(final_beta),
            seed=self.seed,
        )

    def _construct_greedy_baseline(
        self,
        env: Any,
        safety_max_steps: int,
    ) -> Tuple[List[Action], int, Dict[str, Any], str]:
        sequence: List[Action] = []
        info: Dict[str, Any] = {}
        reason = "not_finished"

        for _step in range(int(safety_max_steps)):
            if self._remaining_source_count(env) <= 0:
                reason = "all_source_moved"
                break

            actions = self._valid_actions(env)

            if not actions:
                reason = "no_valid_action"
                break

            best_action = max(actions, key=lambda a: self._heuristic(env, a))

            _state, _reward, done, info = env.step(best_action)
            sequence.append(best_action)

            if done:
                reason = str(info.get("episode_end_reason", "done"))
                break

        else:
            reason = "safety_max_steps"

        obj = self._objective(env, info)
        return sequence, obj, info, reason

    def _construct_solution(
        self,
        env: Any,
        tau: np.ndarray,
        safety_max_steps: int,
        alpha_t: float,
        beta_t: float,
        baseline_seq: Sequence[Action],
        baseline_follow_prob: float,
    ) -> Tuple[List[Action], int, Dict[str, Any], str]:
        sequence: List[Action] = []
        info: Dict[str, Any] = {}
        reason = "not_finished"

        for step_idx in range(int(safety_max_steps)):
            if self._remaining_source_count(env) <= 0:
                reason = "all_source_moved"
                break

            actions = self._valid_actions(env)

            if not actions:
                reason = "no_valid_action"
                break

            baseline_action = None
            if baseline_seq and step_idx < len(baseline_seq):
                baseline_action = baseline_seq[step_idx]

            action = self._select_action(
                env=env,
                actions=actions,
                tau=tau,
                alpha_t=alpha_t,
                beta_t=beta_t,
                baseline_action=baseline_action,
                baseline_follow_prob=baseline_follow_prob,
            )

            _state, _reward, done, info = env.step(action)
            sequence.append(action)

            if done:
                reason = str(info.get("episode_end_reason", "done"))
                break

        else:
            reason = "safety_max_steps"

        obj = self._objective(env, info)

        return sequence, obj, info, reason

    def _valid_actions(self, env: Any) -> List[Action]:
        source_mask, dest_mask = env.get_masks()

        source_flags = self._to_bool_list(source_mask)
        dest_flags = self._to_bool_list(dest_mask)

        actions: List[Action] = []

        for i in range(len(env.from_keys)):
            if not source_flags[i]:
                continue

            from_key = env.from_keys[i]

            if len(env.plates.get(from_key, [])) == 0:
                continue

            for j in range(len(env.to_keys)):
                if not dest_flags[j]:
                    continue

                to_key = env.to_keys[j]

                if str(from_key) == str(to_key):
                    continue

                actions.append((i, j))

        return actions

    def _select_action(
        self,
        env: Any,
        actions: Sequence[Action],
        tau: np.ndarray,
        alpha_t: float,
        beta_t: float,
        baseline_action: Optional[Action],
        baseline_follow_prob: float,
    ) -> Action:
        action_list = list(actions)

        # baseline action이 현재 상태에서도 가능하면 일정 확률로 직접 선택
        if baseline_action is not None and baseline_action in action_list:
            if self.rng.random() < float(baseline_follow_prob):
                return baseline_action

        weights: List[float] = []

        for action in action_list:
            i, j = action

            eta = self._heuristic(env, action)
            pheromone = max(float(tau[i, j]), self.tau_min)

            weight = (pheromone ** alpha_t) * (eta ** beta_t)

            # baseline action은 확률 선택에서도 추가 가중치 부여
            if baseline_action is not None and action == baseline_action:
                weight *= self.baseline_action_bonus

            if not math.isfinite(weight) or weight <= 0.0:
                weight = self.tau_min

            weights.append(weight)

        total = float(sum(weights))

        if total <= 0.0 or not math.isfinite(total):
            return self.rng.choice(action_list)

        threshold = self.rng.random() * total
        cumulative = 0.0

        for action, weight in zip(action_list, weights):
            cumulative += weight

            if cumulative >= threshold:
                return action

        return action_list[-1]

    def _heuristic(self, env: Any, action: Action) -> float:
        i, j = action

        from_key = env.from_keys[i]
        to_key = env.to_keys[j]

        source_stack = env.plates.get(from_key, [])
        dest_stack = env.plates.get(to_key, [])

        if not source_stack:
            return self.tau_min

        moving_plate = source_stack[-1]
        moving_out = int(getattr(moving_plate, "outbound", 10**9))

        estimated_relocations = 0
        due_gap_penalty = 0.0

        for lower_plate in dest_stack:
            lower_out = int(getattr(lower_plate, "outbound", 10**9))

            if lower_out < moving_out:
                estimated_relocations += 1
                due_gap_penalty += 1.0 / max(1, moving_out - lower_out)

        height_penalty = float(len(dest_stack))

        estimate = 1.0
        estimate += self.blocking_weight * float(estimated_relocations)
        estimate += self.due_gap_weight * float(due_gap_penalty)
        estimate += self.height_weight * float(height_penalty)

        return 1.0 / max(estimate, 1e-12)

    def _objective(self, env: Any, info: Optional[Dict[str, Any]]) -> int:
        if info and "final_blocking_metric" in info:
            base_obj = int(info["final_blocking_metric"])
        else:
            base_obj = int(self._dest_blocking(env))

        remaining = self._remaining_source_count(env)

        if remaining > 0:
            return int(base_obj + 100000 * remaining)

        return int(base_obj)

    def _dest_blocking(self, env: Any) -> int:
        total = 0

        for key in env.to_keys:
            total += int(env._get_total_blocking_pairs(env.plates.get(key, [])))

        return total

    def _remaining_source_count(self, env: Any) -> int:
        total = 0

        for key in env.from_keys:
            total += len(env.plates.get(key, []))

        return int(total)

    def _deposit(
        self,
        tau: np.ndarray,
        sequence: Sequence[Action],
        obj: float,
        weight: float = 1.0,
        baseline_obj: Optional[float] = None,
        allow_non_improved: bool = False,
    ) -> None:
        if not sequence or not math.isfinite(obj):
            return

        if baseline_obj is not None and math.isfinite(float(baseline_obj)):
            improvement = float(baseline_obj) - float(obj)

            if improvement > 0:
                amount = (
                    float(weight)
                    * self.q
                    * self.improvement_reward_scale
                    * (1.0 + improvement)
                    / (1.0 + float(obj))
                )
            elif allow_non_improved:
                amount = float(weight)
            else:
                return
        else:
            amount = float(weight) * self.q / (1.0 + float(obj))

        amount = min(float(amount), self.deposit_cap)

        for i, j in sequence:
            if 0 <= i < tau.shape[0] and 0 <= j < tau.shape[1]:
                tau[i, j] += amount

    def _alpha_beta(self, iteration: int) -> Tuple[float, float]:
        if not self.dynamic_alpha_beta:
            return 1.0, 2.0

        I = max(1, int(self.n_iterations))
        i = int(iteration) + 1

        beta_t = 0.9 - 0.8 * (i / I)
        beta_t = max(0.1, min(0.9, beta_t))
        alpha_t = 1.0 - beta_t

        return float(alpha_t), float(beta_t)

    def _baseline_follow_prob(self, iteration: int) -> float:
        I = max(1, int(self.n_iterations) - 1)
        progress = min(1.0, max(0.0, float(iteration) / float(I)))

        prob = (
            self.baseline_follow_prob_start
            + (self.baseline_follow_prob_end - self.baseline_follow_prob_start) * progress
        )

        return float(min(1.0, max(0.0, prob)))

    def _time_exceeded(self, start_time: float) -> bool:
        if self.time_limit_sec is None:
            return False

        return (time.perf_counter() - start_time) >= float(self.time_limit_sec)

    @staticmethod
    def _to_bool_list(mask: Any) -> List[bool]:
        if hasattr(mask, "detach"):
            return mask.detach().cpu().numpy().astype(bool).tolist()

        return np.asarray(mask, dtype=bool).tolist()


# ============================================================
# 4. 평가 실행
# ============================================================

def make_env_for_scenario(
    scenario_df: pd.DataFrame,
    max_stack: int,
    max_steps: int,
) -> Any:
    first = scenario_df.iloc[0]

    plates = scenario_to_plates(scenario_df, force_static=True)

    obs = int(first.get("obs", first.get("prestacked_plates", 0)))

    env = Locating(
        max_stack=max_stack,
        inbound_plates=plates,
        max_steps=max_steps,
        min_obstacles=obs,
        max_obstacles=obs,
        reward_scale=1.0,
    )

    env.reset(shuffle_schedule=False)

    if getattr(env, "pending_inbound_events", None):
        raise RuntimeError(
            "Static ACO expected no pending inbound events after reset. "
            "static filtering 또는 Plate.inbound 값을 확인하세요."
        )

    return env


def count_initial_source_plates(env: Any) -> int:
    total = 0

    for key in env.from_keys:
        total += len(env.plates.get(key, []))

    return int(total)


def evaluate_scenario_repeat(
    scenario_df: pd.DataFrame,
    repeat_id: int,
    eval_seed: int,
    max_stack: int,
    n_ants: int,
    n_iterations: int,
    time_limit_sec: Optional[float],
) -> Dict[str, Any]:
    set_seed(eval_seed)

    first = scenario_df.iloc[0]
    scenario_id = str(first["scenario_id"])

    provisional_max_steps = int(len(scenario_df) + MAX_STEP_BUFFER)

    env = make_env_for_scenario(
        scenario_df=scenario_df,
        max_stack=max_stack,
        max_steps=provisional_max_steps,
    )

    initial_source_plates = count_initial_source_plates(env)
    effective_max_steps = int(initial_source_plates + MAX_STEP_BUFFER)

    progress_label = f"sid={scenario_id}, repeat={repeat_id}"

    solver = WuStyleStaticACO(
        n_ants=n_ants,
        n_iterations=n_iterations,
        rho=RHO,
        q=Q,
        tau0=TAU0,
        max_steps=effective_max_steps,
        time_limit_sec=time_limit_sec,
        dynamic_alpha_beta=USE_DYNAMIC_ALPHA_BETA,
        seed=eval_seed,
        progress_label=progress_label,
        baseline_pheromone_weight=BASELINE_PHEROMONE_WEIGHT,
        improvement_reward_scale=IMPROVEMENT_REWARD_SCALE,
        deposit_cap=DEPOSIT_CAP,
        baseline_action_bonus=BASELINE_ACTION_BONUS,
        baseline_follow_prob_start=BASELINE_FOLLOW_PROB_START,
        baseline_follow_prob_end=BASELINE_FOLLOW_PROB_END,
    )

    result = solver.solve(env)

    return {
        "algorithm": "ACO",
        "aco_variant": "Wu-style static transformed + baseline-centered search",
        "run_mode": RUN_MODE,
        "scenario_id": scenario_id,
        "scenario_index": int(first.get("scenario_index", -1)),
        "repeat_id": int(repeat_id),
        "eval_seed": int(eval_seed),

        "experiment_id": first.get("experiment_id", ""),
        "experiment_name": first.get("experiment_name", ""),
        "test_group": first.get("test_group", first.get("group", "")),
        "condition_order": first.get("condition_order", np.nan),
        "condition_label": first.get("condition_label", ""),
        "condition_value": first.get("condition_value", ""),
        "condition_display": first.get("condition_display", ""),
        "distribution": first.get("distribution", ""),
        "scenario_type": first.get("scenario_type", ""),
        "inbound_pattern": first.get("inbound_pattern", ""),
        "outbound_pattern": first.get("outbound_pattern", ""),

        "nf": int(first.get("nf", scenario_df["pileno"].nunique())),
        "nt": int(first.get("nt", scenario_df["topile"].nunique())),
        "ppp": int(first.get("ppp", 0)),
        "obs": int(first.get("obs", first.get("prestacked_plates", 0))),
        "total_plates": int(len(scenario_df)),
        "initial_source_plates": int(initial_source_plates),
        "max_stack": int(max_stack),
        "safety_max_steps": int(effective_max_steps),
        "max_step_rule": f"initial_source_plates + {MAX_STEP_BUFFER}",

        "baseline_blocking": int(result.baseline_blocking),
        "baseline_move_count": int(result.baseline_move_count),
        "baseline_reason": str(result.baseline_reason),

        "final_blocking": int(result.final_blocking),
        "improvement_over_baseline": int(result.improvement_over_baseline),
        "runtime_sec": float(result.runtime_sec),
        "time_limit_sec": time_limit_sec if time_limit_sec is not None else "",
        "move_count": int(result.move_count),
        "best_reason": result.best_reason,
        "completed_iterations": int(result.completed_iterations),

        "n_ants": int(result.n_ants),
        "n_iterations": int(result.n_iterations),
        "rho": float(result.rho),
        "q": float(result.q),
        "dynamic_alpha_beta": bool(result.dynamic_alpha_beta),
        "final_alpha": float(result.final_alpha),
        "final_beta": float(result.final_beta),

        "tau0": float(TAU0),
        "baseline_pheromone_weight": float(BASELINE_PHEROMONE_WEIGHT),
        "baseline_follow_prob_start": float(BASELINE_FOLLOW_PROB_START),
        "baseline_follow_prob_end": float(BASELINE_FOLLOW_PROB_END),
        "baseline_action_bonus": float(BASELINE_ACTION_BONUS),
    }


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "algorithm",
        "aco_variant",
        "run_mode",
        "experiment_id",
        "condition_order",
        "condition_display",
        "test_group",
        "nf",
        "nt",
        "ppp",
        "obs",
        "max_stack",
        "n_ants",
        "n_iterations",
        "rho",
        "dynamic_alpha_beta",
    ]

    summary = (
        results
        .groupby(group_cols, dropna=False)
        .agg(
            runs=("final_blocking", "count"),
            scenarios=("scenario_id", "nunique"),

            baseline_blocking_mean=("baseline_blocking", "mean"),
            baseline_blocking_min=("baseline_blocking", "min"),
            baseline_blocking_max=("baseline_blocking", "max"),

            final_blocking_mean=("final_blocking", "mean"),
            final_blocking_std=("final_blocking", "std"),
            final_blocking_min=("final_blocking", "min"),
            final_blocking_max=("final_blocking", "max"),

            improvement_over_baseline_mean=("improvement_over_baseline", "mean"),
            improvement_over_baseline_max=("improvement_over_baseline", "max"),

            runtime_mean_sec=("runtime_sec", "mean"),
            runtime_std_sec=("runtime_sec", "std"),
            move_count_mean=("move_count", "mean"),
            completed_iterations_mean=("completed_iterations", "mean"),
        )
        .reset_index()
        .sort_values(["experiment_id", "condition_order", "test_group"])
    )

    return summary


def get_run_settings() -> Tuple[int, int, Optional[int], Optional[float]]:
    mode = str(RUN_MODE).upper().strip()

    if mode == "SINGLE_30MIN":
        repeats = int(SINGLE_REPEATS)
        n_iterations = int(SINGLE_ITERATIONS)
        max_scenarios = int(SINGLE_SCENARIOS)
        time_limit_sec = float(SINGLE_TIME_LIMIT_SEC)
        return repeats, n_iterations, max_scenarios, time_limit_sec

    if mode == "FULL_600":
        repeats = int(FULL_REPEATS)
        n_iterations = int(N_ITERATIONS)
        max_scenarios = None
        time_limit_sec = FULL_TIME_LIMIT_SEC
        return repeats, n_iterations, max_scenarios, time_limit_sec

    raise ValueError(
        f"Unknown RUN_MODE: {RUN_MODE}. "
        f'Use "SINGLE_30MIN" or "FULL_600".'
    )
