import random
import pandas as pd
import numpy as np
import os
import math

try:
    from stockyard.config import get_cfg
except ImportError:
    print("Error: 'cfg.py' not found.")
    raise


class Plate:
    def __init__(
        self,
        id,
        inbound,
        outbound,
        unitw=0.0,
        planned_outbound=None,
        confirmed_outbound=None,
        confirm_time=None,
    ):
        self.id = id
        self.inbound = inbound
        self.outbound = outbound
        self.unitw = unitw

        self.planned_outbound = planned_outbound if planned_outbound is not None else outbound
        self.confirmed_outbound = confirmed_outbound if confirmed_outbound is not None else outbound
        self.confirm_time = confirm_time

        self.from_pile = None
        self.topile = None

    def __repr__(self):
        return (
            f"Plate({self.id}, inbound={self.inbound}, outbound={self.outbound}, "
            f"unitw={self.unitw:.2f})"
        )


def _build_dynamic_inbound_series(total_jobs, start_day, end_day, divisor=80, cap=5):
    """
    동적 입고 강재의 inbound day를 생성한다.

    total_jobs:
        동적 입고로 생성할 강재 수

    start_day, end_day:
        동적 입고가 발생하는 기간

    divisor, cap:
        하루 입고량을 조절하는 파라미터
    """
    total_jobs = int(total_jobs)

    if total_jobs <= 0:
        return np.array([], dtype=np.int32)

    start_day = int(start_day)
    end_day = int(max(start_day, end_day))

    days = []
    remaining = total_jobs
    current_day = start_day

    while remaining > 0:
        daily_cap = max(1, min(int(cap), max(1, total_jobs // max(1, divisor))))
        daily_count = random.randint(1, daily_cap)
        daily_count = min(daily_count, remaining)

        days.extend([current_day] * daily_count)

        remaining -= daily_count
        current_day += 1

        if current_day > end_day:
            current_day = start_day

    days = np.array(days, dtype=np.int32)
    np.random.shuffle(days)

    return days


def _build_inbound_series(
    total_jobs,
    inbound_min,
    inbound_max,
    initial_stock_ratio=1.0,
    initial_stock_window=1,
    dynamic_inbound_end=None,
    divisor=80,
    cap=5,
):
    """
    initial_stock_ratio에 따라 정적/동적 입고를 모두 지원한다.

    initial_stock_ratio = 1.0:
        모든 강재가 초기 시점에 존재한다.
        정적 reshuffling 문제에 해당한다.

    initial_stock_ratio = 0.3:
        전체 강재 중 약 30%만 초기 시점에 존재하고,
        나머지 70%는 시간 흐름에 따라 동적으로 입고된다.
    """
    total_jobs = int(total_jobs)

    if total_jobs <= 0:
        return np.array([], dtype=np.int32)

    inbound_min = int(max(1, inbound_min))
    inbound_max = int(max(inbound_min, inbound_max))
    initial_stock_window = int(max(1, initial_stock_window))

    if dynamic_inbound_end is None:
        dynamic_inbound_end = max(inbound_max, inbound_min + 30)

    dynamic_inbound_end = int(max(inbound_min, dynamic_inbound_end))

    ratio = float(np.clip(initial_stock_ratio, 0.0, 1.0))

    n_initial = int(round(total_jobs * ratio))
    n_initial = int(np.clip(n_initial, 0, total_jobs))
    n_dynamic = total_jobs - n_initial

    # 초기 존재 강재는 initial_stock_window 안에 배치
    # initial_stock_window=1이면 모두 inbound_min에 존재
    initial_upper = inbound_min + initial_stock_window - 1
    initial_upper = min(initial_upper, inbound_max, dynamic_inbound_end)

    if n_initial > 0:
        initial_inbound = np.random.randint(
            inbound_min,
            max(inbound_min, initial_upper) + 1,
            size=n_initial,
            dtype=np.int32,
        )
    else:
        initial_inbound = np.array([], dtype=np.int32)

    # 동적 입고 강재는 initial window 이후부터 발생
    if n_dynamic > 0:
        dynamic_start = max(inbound_min + initial_stock_window, inbound_min + 1)
        dynamic_start = min(dynamic_start, dynamic_inbound_end)

        dynamic_inbound = _build_dynamic_inbound_series(
            total_jobs=n_dynamic,
            start_day=dynamic_start,
            end_day=dynamic_inbound_end,
            divisor=divisor,
            cap=cap,
        )
    else:
        dynamic_inbound = np.array([], dtype=np.int32)

    inbound = np.concatenate([initial_inbound, dynamic_inbound]).astype(np.int32)
    np.random.shuffle(inbound)

    return inbound


def _sample_synthetic_lead_time(cfg, inbound_val):
    """Generic uniform dwell time; no industrial distribution is disclosed."""
    return random.randint(max(1, cfg.lead_time_min), max(1, cfg.lead_time_min, cfg.lead_time_max))


def generate_reshuffle_plan(
    rows,
    n_from_piles_reshuffle,
    n_to_piles_reshuffle,
    n_plates_reshuffle,
    safety_margin,
    max_stack_override=None,
    fixed_obstacles_count=0,
    initial_stock_ratio=None,
):
    """
    Synthetic reshuffling plan generator.

    특징:
        1. 기존 synthetic 공간 생성 구조 유지
        2. No industrial records are read
        3. lead time은 공개용 균등 분포 사용
        4. initial_stock_ratio로 정적/동적 입고 모두 지원

    Output columns:
        pileno
        pileseq
        markno
        unitw
        inbound
        outbound
        planned_outbound
        confirmed_outbound
        confirm_time
        topile
    """
    cfg = get_cfg()

    # ------------------------------------------------------------
    # 1. 최대 적재 높이 설정
    # ------------------------------------------------------------
    max_stack_limit = max_stack_override if max_stack_override is not None else cfg.max_stack
    max_stack_limit = int(max_stack_limit)

    # ------------------------------------------------------------
    # 2. 전체 pile 목록 생성
    # ------------------------------------------------------------
    piles_all = []

    for row_id in rows:
        for col_id in range(1, 31):
            pile = row_id + str(col_id).rjust(2, "0")
            piles_all.append(pile)

    if len(piles_all) == 0:
        return pd.DataFrame([])

    # ------------------------------------------------------------
    # 3. 출발지 pile / 도착지 pile 선정
    # ------------------------------------------------------------
    n_from = min(int(n_from_piles_reshuffle), len(piles_all))
    from_piles = random.sample(piles_all, n_from)

    candidates = [p for p in piles_all if p not in from_piles]
    n_to = min(int(n_to_piles_reshuffle), len(candidates))
    to_piles = random.sample(candidates, n_to)

    if n_from <= 0 or n_to <= 0:
        return pd.DataFrame([])

    # ------------------------------------------------------------
    # 4. 도착지 capacity 및 obstacle 반영
    # ------------------------------------------------------------
    capacity_map = {p: max_stack_limit for p in to_piles}
    current_obstacles = 0

    for _ in range(int(fixed_obstacles_count) * 2):
        if current_obstacles >= int(fixed_obstacles_count):
            break

        target_pile = random.choice(to_piles)

        if capacity_map[target_pile] > 0:
            capacity_map[target_pile] -= 1
            current_obstacles += 1

    # ------------------------------------------------------------
    # 5. 요청 물량이 도착지 capacity를 넘지 않도록 조정
    # ------------------------------------------------------------
    total_capacity = sum(capacity_map.values())
    requested_demand = n_from * int(n_plates_reshuffle)

    if total_capacity <= 0:
        raise ValueError("Generation Failed: destination capacity is zero after obstacle placement.")

    hard_mode_ratio = float(getattr(cfg, "hard_mode_ratio", 0.90))
    actual_plates_per_pile = int(n_plates_reshuffle)

    if requested_demand > total_capacity * hard_mode_ratio:
        adjusted_total = int(total_capacity * hard_mode_ratio)

        if n_from > 0:
            adjusted_ppp = max(1, adjusted_total // n_from)
            actual_plates_per_pile = adjusted_ppp
        else:
            actual_plates_per_pile = 0

    # source pile 높이를 max_stack으로 제한하고 싶을 때만 사용
    # 기본값은 False로 둬야 PPP 22, 29 같은 케이스가 유지됨
    source_stack_limit_enabled = bool(getattr(cfg, "source_stack_limit_enabled", False))

    if source_stack_limit_enabled:
        actual_plates_per_pile = min(actual_plates_per_pile, max_stack_limit)

    # ------------------------------------------------------------
    # 6. 기본 작업 row 생성
    # ------------------------------------------------------------
    base_rows = []

    for fp in from_piles:
        for i in range(1, actual_plates_per_pile + 1):
            feasible_targets = [p for p in to_piles if capacity_map[p] > 0]

            if not feasible_targets:
                raise ValueError(
                    f"Generation Failed: destination capacity is full. "
                    f"Total capacity={total_capacity}, generated={len(base_rows)}"
                )

            selected_to = random.choice(feasible_targets)
            capacity_map[selected_to] -= 1

            base_rows.append({
                "pileno": fp,
                "pileseq": str(i).zfill(3),
                "markno": f"SP-{fp}-{i}",
                "unitw": round(float(np.random.uniform(cfg.unitw_min, cfg.unitw_max)), 2),
                "topile": selected_to,
            })

    if len(base_rows) == 0:
        return pd.DataFrame([])

    random.shuffle(base_rows)
    total_jobs = len(base_rows)

    # ------------------------------------------------------------
    # 7. inbound 생성
    # ------------------------------------------------------------
    if initial_stock_ratio is None:
        ratio = float(getattr(cfg, "initial_stock_ratio", 1.0))
    else:
        ratio = float(initial_stock_ratio)

    inbound_arr = _build_inbound_series(
        total_jobs=total_jobs,
        inbound_min=getattr(cfg, "inbound_min", 1),
        inbound_max=getattr(cfg, "inbound_max", 30),
        initial_stock_ratio=ratio,
        initial_stock_window=getattr(cfg, "initial_stock_window", 1),
        dynamic_inbound_end=getattr(
            cfg,
            "dynamic_inbound_end",
            max(getattr(cfg, "inbound_max", 30), 300),
        ),
        divisor=getattr(cfg, "dynamic_inbound_divisor", 80),
        cap=getattr(cfg, "dynamic_inbound_cap", 5),
    )

    # ------------------------------------------------------------
    # 8. 최종 DataFrame 생성
    # ------------------------------------------------------------
    df_rows = []

    for idx, row in enumerate(base_rows):
        inbound_val = int(inbound_arr[idx])
        stay_duration = _sample_synthetic_lead_time(cfg, inbound_val)

        outbound_val = int(inbound_val + stay_duration)

        # 현재는 계획/확정 출고를 동일하게 둔다.
        # 추후 planned/confirmed 차이를 만들고 싶으면 여기에서 분리하면 됨
        planned_outbound_val = outbound_val
        confirmed_outbound_val = outbound_val
        confirm_time_val = None

        df_rows.append({
            "pileno": row["pileno"],
            "pileseq": row["pileseq"],
            "markno": row["markno"],
            "unitw": row["unitw"],
            "inbound": inbound_val,
            "outbound": confirmed_outbound_val,
            "planned_outbound": planned_outbound_val,
            "confirmed_outbound": confirmed_outbound_val,
            "confirm_time": confirm_time_val,
            "topile": row["topile"],
        })

    df = pd.DataFrame(df_rows)

    # ------------------------------------------------------------
    # 9. 안전성 검사
    # ------------------------------------------------------------
    if len(df) > 0:
        df["inbound"] = pd.to_numeric(df["inbound"], errors="coerce").fillna(1).astype(int)

        df["outbound"] = (
            pd.to_numeric(df["outbound"], errors="coerce")
            .fillna(df["inbound"])
            .astype(int)
        )

        df["planned_outbound"] = (
            pd.to_numeric(df["planned_outbound"], errors="coerce")
            .fillna(df["outbound"])
            .astype(int)
        )

        df["confirmed_outbound"] = (
            pd.to_numeric(df["confirmed_outbound"], errors="coerce")
            .fillna(df["outbound"])
            .astype(int)
        )

        df.loc[df["outbound"] < df["inbound"], "outbound"] = df.loc[
            df["outbound"] < df["inbound"], "inbound"
        ]

        df.loc[df["planned_outbound"] < df["inbound"], "planned_outbound"] = df.loc[
            df["planned_outbound"] < df["inbound"], "inbound"
        ]

        df.loc[df["confirmed_outbound"] < df["inbound"], "confirmed_outbound"] = df.loc[
            df["confirmed_outbound"] < df["inbound"], "inbound"
        ]

    # ------------------------------------------------------------
    # 10. 선택적 분포 확인 로그
    # ------------------------------------------------------------
    if len(df) > 0 and bool(getattr(cfg, "print_data_distribution", False)):
        lead = df["outbound"] - df["inbound"]

        inbound_min = int(getattr(cfg, "inbound_min", 1))
        initial_stock_window = int(getattr(cfg, "initial_stock_window", 1))
        initial_upper = inbound_min + initial_stock_window - 1
        real_initial_ratio = float((df["inbound"] <= initial_upper).mean())

        print(
            "[SYNTHETIC DATA DIST]",
            f"N={len(df)}",
            f"inbound=({df['inbound'].min()}, {df['inbound'].max()})",
            f"outbound=({df['outbound'].min()}, {df['outbound'].max()})",
            f"lead_mean={lead.mean():.2f}",
            f"lead_q50={lead.quantile(0.50):.1f}",
            f"lead_q75={lead.quantile(0.75):.1f}",
            f"lead_q90={lead.quantile(0.90):.1f}",
            f"lead_q95={lead.quantile(0.95):.1f}",
            f"lead_max={lead.max()}",
            f"initial_ratio={real_initial_ratio:.2f}",
        )

    return df


def save_reshuffle_plan_to_excel(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="reshuffle", index=False)

    print(f"  -> Saved: {os.path.basename(path)} (Total Plates: {len(df)})")
