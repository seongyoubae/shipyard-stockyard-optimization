import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import os
import time
from stockyard.config import get_cfg
import random
def solve_scenario_sequential_gurobi(
    scenario_df,
    scenario_name,
    file_name,
    max_stack,
    time_limit,
    mip_gap=0.05
):
    """
    Sequential Gurobi Baseline

    - source pile의 top-only 이동 제약 반영
    - 각 step에서 정확히 하나의 강재 이동
    - destination pile에는 이동 순서대로 적치
    - 최종 blocking pair 최소화
    """

    scenario_df = scenario_df.sort_values(
        by=["pileno", "pileseq"]
    ).reset_index(drop=True)

    plates = []
    source_piles = {}

    for idx, row in scenario_df.iterrows():
        plates.append({
            "idx": idx,
            "id": row["markno"],
            "outbound": int(row["outbound"]),
            "pileno": row["pileno"],
            "pileseq": int(row["pileseq"]),
        })

        if row["pileno"] not in source_piles:
            source_piles[row["pileno"]] = []

        source_piles[row["pileno"]].append((idx, int(row["pileseq"])))

    N = len(plates)
    dest_names = sorted(scenario_df["topile"].unique())
    M = len(dest_names)

    if N > M * max_stack:
        return -1, 0.0, "Infeasible_Capacity", -1.0

    I = range(N)
    S = range(M)
    T = range(N)

    pairs = [(i, j) for i in I for j in I if i < j]

    # ------------------------------------------------------------
    # Source top-only 제약
    # pileseq가 작을수록 아래, 클수록 위라고 가정
    # 위에 있는 강재가 아래 강재보다 먼저 이동해야 함
    # ------------------------------------------------------------
    source_precedence = []

    for pileno, items in source_piles.items():
        sorted_items = sorted(items, key=lambda x: x[1])

        for lower_pos in range(len(sorted_items)):
            for upper_pos in range(lower_pos + 1, len(sorted_items)):
                lower_idx = sorted_items[lower_pos][0]
                upper_idx = sorted_items[upper_pos][0]

                # upper가 lower보다 먼저 이동
                source_precedence.append((upper_idx, lower_idx))

    # ------------------------------------------------------------
    # Gurobi model
    # ------------------------------------------------------------
    model = gp.Model(f"Sequential_SPSP_{scenario_name}")

    model.setParam("OutputFlag", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("MIPGap", mip_gap)
    model.setParam("Threads", 0)
    model.setParam("MIPFocus", 1)
    model.setParam("Heuristics", 0.8)
    model.setParam("NoRelHeurTime", min(300, max(0, time_limit * 0.25)))
    print(
        "[CHECK] Params:",
        "MIPFocus =", model.Params.MIPFocus,
        "Heuristics =", model.Params.Heuristics,
        "NoRelHeurTime =", model.Params.NoRelHeurTime,
    )
    # ------------------------------------------------------------
    # Variables
    # ------------------------------------------------------------

    # z[i,t] = plate i가 t번째 이동이면 1
    z = model.addVars(I, T, vtype=GRB.BINARY, name="z")

    # x[i,s] = plate i가 destination pile s에 배정되면 1
    x = model.addVars(I, S, vtype=GRB.BINARY, name="x")

    # move_time[i]
    move_time = model.addVars(I, vtype=GRB.INTEGER, lb=0, ub=N - 1, name="move_time")

    # after[i,j] = 1이면 i가 j보다 나중에 이동
    after = model.addVars(pairs, vtype=GRB.BINARY, name="after")

    # same[i,j] = 1이면 i와 j가 같은 destination pile
    same = model.addVars(pairs, vtype=GRB.BINARY, name="same")

    # r[i,j] = same[i,j] AND after[i,j]
    # 즉, i와 j가 같은 pile이고 i가 j보다 나중에 이동
    r = model.addVars(pairs, vtype=GRB.BINARY, name="same_and_after")

    # p[i,j,s] = i와 j가 둘 다 destination s에 배정되면 1
    p = model.addVars(
        [(i, j, s) for (i, j) in pairs for s in S],
        vtype=GRB.CONTINUOUS,
        lb=0.0,
        ub=1.0,
        name="same_dest_s"
    )

    # ------------------------------------------------------------
    # 1. 각 강재는 정확히 한 step에서 이동
    # ------------------------------------------------------------
    model.addConstrs(
        (gp.quicksum(z[i, t] for t in T) == 1 for i in I),
        name="EachPlateMovedOnce"
    )

    # ------------------------------------------------------------
    # 2. 각 step에서는 정확히 하나의 강재만 이동
    # ------------------------------------------------------------
    model.addConstrs(
        (gp.quicksum(z[i, t] for i in I) == 1 for t in T),
        name="OnePlatePerStep"
    )

    # ------------------------------------------------------------
    # 3. move_time 정의
    # ------------------------------------------------------------
    model.addConstrs(
        (
            move_time[i] == gp.quicksum(t * z[i, t] for t in T)
            for i in I
        ),
        name="MoveTimeDef"
    )

    # ------------------------------------------------------------
    # 4. 각 강재는 하나의 destination pile에 배정
    # ------------------------------------------------------------
    model.addConstrs(
        (gp.quicksum(x[i, s] for s in S) == 1 for i in I),
        name="AssignDestination"
    )

    # ------------------------------------------------------------
    # 5. destination pile capacity
    # ------------------------------------------------------------
    model.addConstrs(
        (gp.quicksum(x[i, s] for i in I) <= max_stack for s in S),
        name="DestinationCapacity"
    )

    # ------------------------------------------------------------
    # 6. Source top-only 제약
    # upper plate가 lower plate보다 먼저 이동해야 함
    # ------------------------------------------------------------
    model.addConstrs(
        (
            move_time[upper] + 1 <= move_time[lower]
            for upper, lower in source_precedence
        ),
        name="SourceTopOnly"
    )

    # ------------------------------------------------------------
    # 7. destination symmetry breaking
    # 같은 크기의 destination pile들이므로 앞 pile부터 채우도록 유도
    # ------------------------------------------------------------
    for s in range(M - 1):
        model.addConstr(
            gp.quicksum(x[i, s] for i in I) >= gp.quicksum(x[i, s + 1] for i in I),
            name=f"SymBreak_{s}"
        )

    # ------------------------------------------------------------
    # 8. after[i,j] 정의
    # after[i,j] = 1이면 i가 j보다 나중에 이동
    # ------------------------------------------------------------
    big_m = N

    for i, j in pairs:
        model.addConstr(
            move_time[i] - move_time[j] >= 1 - big_m * (1 - after[i, j]),
            name=f"AfterDef1_{i}_{j}"
        )

        model.addConstr(
            move_time[j] - move_time[i] >= 1 - big_m * after[i, j],
            name=f"AfterDef2_{i}_{j}"
        )

    # ------------------------------------------------------------
    # 9. same[i,j] 정의
    # ------------------------------------------------------------
    for i, j in pairs:
        for s in S:
            model.addConstr(p[i, j, s] <= x[i, s])
            model.addConstr(p[i, j, s] <= x[j, s])
            model.addConstr(p[i, j, s] >= x[i, s] + x[j, s] - 1)

        model.addConstr(
            same[i, j] == gp.quicksum(p[i, j, s] for s in S),
            name=f"SameDest_{i}_{j}"
        )

    # ------------------------------------------------------------
    # 10. r[i,j] = same[i,j] AND after[i,j]
    # ------------------------------------------------------------
    for i, j in pairs:
        model.addConstr(r[i, j] <= same[i, j])
        model.addConstr(r[i, j] <= after[i, j])
        model.addConstr(r[i, j] >= same[i, j] + after[i, j] - 1)

    # ------------------------------------------------------------
    # 11. Objective: final blocking pair minimize
    # ------------------------------------------------------------
    obj = gp.LinExpr()

    for i, j in pairs:
        out_i = plates[i]["outbound"]
        out_j = plates[j]["outbound"]

        # case 1: i가 j보다 나중에 이동
        # destination에서는 i가 j 위에 있음
        # j가 더 빨리 나가야 하면 blocking
        if out_j < out_i:
            obj += r[i, j]

        # case 2: j가 i보다 나중에 이동
        # destination에서는 j가 i 위에 있음
        # i가 더 빨리 나가야 하면 blocking
        if out_i < out_j:
            obj += same[i, j] - r[i, j]

    model.setObjective(obj, GRB.MINIMIZE)
    # ------------------------------------------------------------
    # 12. MIP Start: random feasible 초기해
    # - 반출 예정일 / blocking 정보 사용하지 않음
    # - source top-only 제약만 만족
    # - destination은 symmetry breaking을 만족하도록 앞 파일부터 순차 배정
    # ------------------------------------------------------------

    random.seed(42)

    # source pile별 남은 강판 목록 구성
    # pileseq가 클수록 위에 있다고 가정
    source_remaining = {}

    for pileno, items in source_piles.items():
        source_remaining[pileno] = sorted(items, key=lambda x: x[1])

    start_order = []

    while len(start_order) < N:
        candidate_piles = [
            pileno for pileno, items in source_remaining.items()
            if len(items) > 0
        ]

        selected_pileno = random.choice(candidate_piles)

        # 가장 위 강판 선택
        plate_idx, _ = source_remaining[selected_pileno].pop()

        start_order.append(plate_idx)

    assert len(start_order) == N
    assert len(set(start_order)) == N

    # 기존 start 값을 명시적으로 0으로 초기화
    for i in I:
        for t in T:
            z[i, t].Start = 0

    for i in I:
        for s in S:
            x[i, s].Start = 0

    # 이동 순서 start 입력
    for t, plate_idx in enumerate(start_order):
        z[plate_idx, t].Start = 1
        move_time[plate_idx].Start = t

    # destination 배정
    # SymBreak 제약 때문에 완전 랜덤 destination은 피하고,
    # 앞 destination부터 max_stack까지 채우는 방식으로 feasible start 구성
    dest_load = {s: 0 for s in S}
    current_s = 0

    for plate_idx in start_order:
        while dest_load[current_s] >= max_stack:
            current_s += 1

        x[plate_idx, current_s].Start = 1
        dest_load[current_s] += 1

    print("[MIP START] random top-only source order + sequential destination assignment")
    print("[MIP START] destination loads:", dest_load)

    # ------------------------------------------------------------
    # Optimize
    # ------------------------------------------------------------
    st = time.time()
    model.optimize()
    dur = time.time() - st

    try:
        gap = model.MIPGap
    except Exception:
        gap = -1.0

    if model.Status == GRB.OPTIMAL:
        return model.ObjVal, dur, "Optimal", gap

    elif model.Status == GRB.TIME_LIMIT:
        if model.SolCount > 0:
            return model.ObjVal, dur, "Time_Limit (Feasible)", gap
        else:
            return -1, dur, "Time_Limit (No Sol)", gap

    elif model.Status == GRB.INFEASIBLE:
        return -1, dur, "Infeasible", -1.0

    else:
        if model.SolCount > 0:
            return model.ObjVal, dur, f"Status_{model.Status}_Feasible", gap
        return -1, dur, f"Status_{model.Status}", gap
