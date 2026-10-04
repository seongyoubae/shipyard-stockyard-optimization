import os
from collections import Counter

# 환경 변수 설정 (CPU 병렬 처리 제어)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["IN_MP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import torch
import numpy as np
import random
import csv
import datetime
import copy
import pandas as pd
import gc
from collections import defaultdict

# 필수 모듈 임포트
try:
    from stockyard.config import get_cfg
    from stockyard.environment.yard import Locating
    from stockyard.models.network import SteelPlateConditionalMLPModel
    from stockyard.evaluation.policy import evaluate_policy
    from stockyard.data.generator import Plate, generate_reshuffle_plan
except ImportError as e:
    print(f"[오류] 필수 모듈을 찾을 수 없습니다: {e}")
    exit()

import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR

try:
    import vessl

    USE_VESSL = True
except ImportError:
    USE_VESSL = False
    print("Vessl not installed. Skipping Vessl logging.")

DEFAULT_BASE_SEED = 42

def calculate_blocking_cnt(pile):
    if len(pile) <= 1: return 0
    cnt = 0
    outbounds = [p.outbound for p in pile]
    for i in range(len(outbounds)):
        for j in range(i + 1, len(outbounds)):
            if outbounds[i] < outbounds[j]: cnt += 1
    return cnt


def apply_s1_heuristic(original_plates, n_piles, max_stack, strategy):
    piles = {i: [] for i in range(n_piles)}

    for plate in original_plates:
        best_idx = 0

        if strategy == "Random":
            valid = [k for k, v in piles.items() if len(v) < max_stack]
            best_idx = random.choice(valid) if valid else 0

        elif strategy == "H1":
            min_gap = float('inf')
            candidates = []
            for k, v in piles.items():
                if len(v) >= max_stack: continue
                if not v:
                    gap = 9999
                else:
                    top = v[-1]
                    if top.outbound >= plate.outbound:
                        gap = top.outbound - plate.outbound
                    else:
                        continue
                if gap < min_gap:
                    min_gap = gap
                    candidates = [k]
                elif gap == min_gap:
                    candidates.append(k)

            if not candidates:
                empties = [k for k in piles if not piles[k]]
                if empties:
                    best_idx = empties[0]
                else:
                    min_rev = float('inf')
                    rev_cands = []
                    for k, v in piles.items():
                        if len(v) < max_stack:
                            gap = plate.outbound - v[-1].outbound
                            if gap < min_rev:
                                min_rev = gap
                                rev_cands = [k]
                            elif gap == min_rev:
                                rev_cands.append(k)

                    best_idx = random.choice(rev_cands) if rev_cands else 0
            else:
                best_idx = random.choice(candidates)

        elif strategy == "H2":
            min_blk = float('inf')
            candidates = []
            for k, v in piles.items():
                if len(v) >= max_stack: continue
                blk = calculate_blocking_cnt(v + [plate])
                if blk < min_blk:
                    min_blk = blk
                    candidates = [k]
                elif blk == min_blk:
                    candidates.append(k)
            best_idx = random.choice(candidates) if candidates else 0

        elif strategy == "H3":
            min_top = float('inf')
            candidates = []
            for k, v in piles.items():
                if len(v) >= max_stack: continue
                top_val = v[-1].outbound if v else 99999
                if top_val < min_top:
                    min_top = top_val
                    candidates = [k]
                elif top_val == min_top:
                    candidates.append(k)
            best_idx = random.choice(candidates) if candidates else 0

        elif strategy == "Reverse_H1":
            min_gap = float('inf')
            candidates = []
            for k, v in piles.items():
                if len(v) >= max_stack: continue
                if not v:
                    gap = 9999
                else:
                    top = v[-1]
                    if top.outbound <= plate.outbound:
                        gap = plate.outbound - top.outbound
                    else:
                        continue
                if gap < min_gap:
                    min_gap = gap
                    candidates = [k]
                elif gap == min_gap:
                    candidates.append(k)

            if not candidates:
                valid = [k for k in piles if len(piles[k]) < max_stack]
                best_idx = random.choice(valid) if valid else 0
            else:
                best_idx = random.choice(candidates)

        else:
            valid = [k for k, v in piles.items() if len(v) < max_stack]
            best_idx = random.choice(valid) if valid else 0

        piles[best_idx].append(plate)
        plate.from_pile = str(best_idx)

    return original_plates

class RewardNormalizer:
    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.running_mean = np.zeros((), dtype=np.float64)
        self.running_var = np.ones((), dtype=np.float64)
        self.count = 1e-4

    def update(self, rewards):
        batch_mean = np.mean(rewards)
        batch_var = np.var(rewards)
        batch_count = rewards.size
        delta = batch_mean - self.running_mean
        tot_count = self.count + batch_count
        new_mean = self.running_mean + delta * batch_count / tot_count
        m_a = self.running_var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = m2 / tot_count
        self.running_mean = new_mean
        self.running_var = new_var
        self.count = tot_count

    def normalize(self, rewards):
        return (rewards - self.running_mean) / np.sqrt(self.running_var + 1e-8)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def snapshot_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().clone(),
    }

    if torch.cuda.is_available():
        state["cuda"] = [s.clone() for s in torch.cuda.get_rng_state_all()]

    return state


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])

    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])

def build_fixed_anchor_eval_scenarios():
    return [
    {"name": "Basic_15x15", "nf": 15, "nt": 15, "ppp": 15, "obs": 10},
    {"name": "Mini_10x5",   "nf": 10, "nt": 5,  "ppp": 15, "obs": 10},
    {"name": "Lite_20x9",   "nf": 20, "nt": 9,  "ppp": 15, "obs": 20},
    {"name": "HELL_20x8",   "nf": 20, "nt": 8,  "ppp": 15, "obs": 20},
]

def stable_name_seed(name):
    return sum((i + 1) * ord(ch) for i, ch in enumerate(str(name)))


def generate_schedule_from_scenario(cfg, max_stack_limit, nf, nt, target_ppp, obs, s1_strategy="Random"):
    total_capacity = nt * max_stack_limit
    safe_capacity = (total_capacity - obs) * 0.90
    calculated_safe_ppp = int(safe_capacity / max(nf, 1))
    real_ppp = min(target_ppp, max(calculated_safe_ppp, 0))

    if real_ppp < 1:
        return [], real_ppp

    while True:
        try:
            # print(
            #     "[EVAL CFG]",
            #     f"cfg.initial_stock_ratio={cfg.initial_stock_ratio}",
            #     f"nf={nf}",
            #     f"nt={nt}",
            #     f"target_ppp={target_ppp}",
            #     f"real_ppp={real_ppp}",
            #     f"obs={obs}",
            #     f"strategy={s1_strategy}"
            # )

            df_plan = generate_reshuffle_plan(
                rows=['A', 'B'],
                n_from_piles_reshuffle=nf,
                n_to_piles_reshuffle=nt,
                n_plates_reshuffle=real_ppp,
                safety_margin=cfg.safety_margin,
                max_stack_override=max_stack_limit,
                fixed_obstacles_count=obs,
                initial_stock_ratio=cfg.initial_stock_ratio
            )

            schedule = []
            for _, row in df_plan.iterrows():
                p = Plate(
                    id=row['markno'] if 'markno' in df_plan.columns else row['pileno'],
                    inbound=row['inbound'],
                    outbound=row['outbound'],
                    unitw=row['unitw'],
                    planned_outbound=row['planned_outbound'] if 'planned_outbound' in df_plan.columns else row['outbound'],
                    confirmed_outbound=row['confirmed_outbound'] if 'confirmed_outbound' in df_plan.columns else row['outbound'],
                    confirm_time=row['confirm_time'] if (
                        'confirm_time' in df_plan.columns and pd.notna(row['confirm_time'])
                    ) else None,
                )
                p.from_pile = str(row['pileno']).strip()
                p.topile = str(row['topile']).strip()
                schedule.append(p)

            schedule = apply_s1_heuristic(schedule, nf, max_stack_limit, s1_strategy)
            return schedule, real_ppp

        except Exception as e:
            print("[SCENARIO GEN ERROR]", e)
            continue


def run_scenario_benchmark(model, device, cfg, stack_limit, scenarios, episodes, seed_prefix, strategy="Random"):
    scores = []
    details = []

    for scenario in scenarios:
        name = scenario["name"]
        nf, nt, target_ppp, obs = scenario["nf"], scenario["nt"], scenario["ppp"], scenario["obs"]
        scenario_seed = int(seed_prefix) + stable_name_seed(name) % 10000 + stable_name_seed(strategy) % 1000
        scenario_metrics = []
        real_ppp_logged = None

        for ep in range(max(1, int(episodes))):
            set_seed(scenario_seed + ep)
            schedule, real_ppp = generate_schedule_from_scenario(
                cfg=cfg,
                max_stack_limit=stack_limit,
                nf=nf,
                nt=nt,
                target_ppp=target_ppp,
                obs=obs,
                s1_strategy=strategy
            )
            real_ppp_logged = real_ppp

            if not schedule:
                scenario_metrics.append(1000.0)
                continue

            eval_env = Locating(
                max_stack=stack_limit,
                inbound_plates=schedule,
                min_obstacles=obs,
                max_obstacles=obs,
                observed_top_n_plates=cfg.OBSERVED_TOP_N_PLATES,
                num_summary_stats_deeper=cfg.NUM_SUMMARY_STATS_DEEPER,
                max_steps=cfg.max_steps
            )

            _, metric, _ = evaluate_policy(model, [eval_env], device, return_blocked=True)
            scenario_metrics.append(metric if metric != float("inf") else 1000.0)

        avg_score = float(np.mean(scenario_metrics)) if scenario_metrics else 1000.0
        scores.append(avg_score)
        details.append((name, nf, nt, obs, real_ppp_logged, avg_score))

    overall = float(np.mean(scores)) if scores else float("inf")
    return overall, details

def main():
    base_seed = DEFAULT_BASE_SEED
    set_seed(base_seed)
    cfg = get_cfg()

    print(f"NOTE: Parallel environments set to {cfg.num_envs} for Optimization.")
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(cfg.save_model_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    current_working_dir = os.getcwd()

    runs_base_path = os.path.join(current_working_dir, "runs")
    if not os.path.exists(runs_base_path):
        os.makedirs(runs_base_path)

    tb_log_dir = os.path.join(runs_base_path, timestamp)
    if not os.path.exists(tb_log_dir):
        os.makedirs(tb_log_dir)

    safe_log_dir = tb_log_dir.replace("\\", "/")
    tb_writer = SummaryWriter(log_dir=safe_log_dir)

    cfg_info = (
        f"Actor LR: {cfg.actor_lr}, Critic LR: {cfg.critic_lr}\n"
        f"Epochs: {cfg.n_epoch}, Parallel Envs: {cfg.num_envs}\n"
        f"Encoder Type: {cfg.encoder_type.upper()}"
    )
    print("=== Hyperparameters ===\n" + cfg_info + "\n=======================")

    best_metric = float("inf")

    temp_plate = Plate(id="TMP_000", inbound=1, outbound=2, unitw=1.0)
    temp_plate.from_pile = "TMP_FROM"
    temp_plate.topile = "TMP_TO"

    temp_env = Locating(inbound_plates=[temp_plate],
                        max_stack=50,
                        observed_top_n_plates=cfg.OBSERVED_TOP_N_PLATES,
                        num_summary_stats_deeper=cfg.NUM_SUMMARY_STATS_DEEPER,
                        gamma=cfg.gamma,
                        max_steps=cfg.max_steps)
    actual_pile_feature_dim = temp_env.actual_pile_feature_dim
    del temp_env

    model = SteelPlateConditionalMLPModel(
        embed_dim=cfg.embed_dim, num_actor_layers=cfg.num_actor_layers,
        num_critic_layers=cfg.num_critic_layers, actor_init_std=cfg.actor_init_std,
        critic_init_std=cfg.critic_init_std, pile_feature_dim=actual_pile_feature_dim,
        num_heads=cfg.num_heads,
        encoder_type=cfg.encoder_type
    ).to(device)

    if cfg.load_model and os.path.exists(cfg.model_path):
        try:
            loaded_data = torch.load(cfg.model_path, map_location=device)
            state_dict = loaded_data.get('model_state_dict', loaded_data)
            model.load_state_dict(state_dict)
            print("Model loaded successfully for fine-tuning.")
        except Exception as e:
            print(f"[Warning] Failed to load model: {e}")

    actor_optimizer = optim.Adam(model.actor_parameters(), lr=cfg.actor_lr, weight_decay=cfg.weight_decay)
    critic_optimizer = optim.Adam(model.critic_parameters(), lr=cfg.critic_lr, weight_decay=cfg.weight_decay)

    lr_lambda = lambda epoch: max(0.0, 1.0 - (epoch / float(cfg.n_epoch)))
    actor_lr_sched = LambdaLR(actor_optimizer, lr_lambda=lr_lambda)
    critic_lr_sched = LambdaLR(critic_optimizer, lr_lambda=lr_lambda)

    log_filename = cfg.log_file if cfg.log_file else f"train_log_{timestamp}.csv"
    with open(log_filename, mode="w", newline="") as f:
        csv.writer(f).writerow(
            ["Epoch", "Total_Steps", "Avg_Episode_Reward", "Avg_Loss", "Actor_Loss", "Critic_Loss", "Entropy_Loss",
             "Avg_Final_Reversals", "Avg_Crane_Moves", "Avg_Obstacles"])

    reward_normalizer = RewardNormalizer(num_envs=cfg.num_envs)
    global_step = 0
    envs = None

    # --- 메인 학습 루프 ---
    for epoch in range(cfg.n_epoch):
        if envs is None or epoch % cfg.new_instance_every == 0:
            if envs is not None:
                del envs
                gc.collect()

            envs = []
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Generating Diverse Environments (Mixed: Inbound + Reshuffle)...")

            scenario_counter = Counter()
            s1_counter = Counter()
            ratio_counter = Counter()

            # 정적 중심이므로 inbound는 5%만
            INBOUND_TRAINING_PROB = 0.05

            for i in range(cfg.num_envs):
                while True:
                    try:
                        current_max_stack = 50

                        # ------------------------------------------------
                        # 1) 아주 약한 inbound training branch
                        # ------------------------------------------------
                        is_inbound_training = (random.random() < INBOUND_TRAINING_PROB)

                        if is_inbound_training:
                            scenario_type = "inbound_training"
                            cfg.initial_stock_ratio = 0.7

                            # 쉬운 inbound 상황만 제공
                            n_f = 1
                            n_t = 20
                            ppp = random.randint(15, 30)
                            obs = random.randint(0, 5)

                            s1_strategy = "Random"

                        else:
                            # cfg.initial_stock_ratio = 1.0
                            #
                            # s1_strategy = "Random"
                            # r = random.random()
                            #
                            # r_ratio = random.random()
                            #
                            s1_strategy = random.choices(
                                population=["Random", "H1", "H2", "Reverse_H1", "H3"],
                                weights=[0.30, 0.25, 0.20, 0.15, 0.10],
                                k=1
                            )[0]
                            r = random.random()

                            r_ratio = random.random()

                            if r_ratio < 0.75:
                                cfg.initial_stock_ratio = 1.0
                            elif r_ratio < 0.875:
                                cfg.initial_stock_ratio = 0.9
                            elif r_ratio < 0.95:
                                cfg.initial_stock_ratio = 0.8
                            else:
                                cfg.initial_stock_ratio = 0.7

                            if r < 0.25:
                                scenario_type = "easy"
                                n_f = random.randint(8, 14)
                                n_t = random.randint(14, 22)
                                ppp = random.randint(5, 10)
                                obs = random.randint(0, 10)

                            elif r < 0.70:
                                scenario_type = "medium"
                                n_f = random.randint(14, 22)
                                n_t = random.randint(9, 16)
                                ppp = random.randint(10, 15)
                                obs = random.randint(10, 25)

                            else:
                                scenario_type = "hard"
                                n_f = random.randint(18, 24)
                                n_t = random.randint(7, 12)
                                ppp = random.randint(15, 18)
                                obs = random.randint(15, 30)

                        # 1. 장애물(obs) 오버플로우 방지 (도착지 용량의 80% 제한)
                        total_dest_capacity = n_t * current_max_stack
                        obs = min(obs, int(total_dest_capacity * 0.8))

                        # 2. 이동 강판 수(ppp) 데드락 방지
                        source_safe_cap = max(1, current_max_stack - 1)
                        dest_safe_cap = max(1, int((n_t * current_max_stack - obs) * 0.90))
                        max_ppp_by_dest = max(1, dest_safe_cap // max(n_f, 1))
                        ppp = min(ppp, source_safe_cap, max_ppp_by_dest)

                        # ------------------------------------------------
                        # 3) 실제 데이터 생성
                        # ------------------------------------------------
                        df_plan = generate_reshuffle_plan(
                            rows=['A', 'B'],
                            n_from_piles_reshuffle=n_f,
                            n_to_piles_reshuffle=n_t,
                            n_plates_reshuffle=ppp,
                            safety_margin=cfg.safety_margin,
                            max_stack_override=current_max_stack,
                            fixed_obstacles_count=obs,
                            initial_stock_ratio=cfg.initial_stock_ratio
                        )

                        train_schedule = []
                        for _, row in df_plan.iterrows():
                            p = Plate(
                                id=row['markno'] if 'markno' in df_plan.columns else row['pileno'],
                                inbound=row['inbound'],
                                outbound=row['outbound'],
                                unitw=row['unitw'],
                                planned_outbound=row['planned_outbound'] if 'planned_outbound' in df_plan.columns else
                                row['outbound'],
                                confirmed_outbound=row[
                                    'confirmed_outbound'] if 'confirmed_outbound' in df_plan.columns else row[
                                    'outbound'],
                                confirm_time=row['confirm_time'] if (
                                        'confirm_time' in df_plan.columns and pd.notna(row['confirm_time'])
                                ) else None,
                            )
                            p.from_pile = str(row['pileno']).strip()
                            p.topile = str(row['topile']).strip()
                            train_schedule.append(p)

                        train_schedule = apply_s1_heuristic(
                            train_schedule,
                            n_f,
                            current_max_stack,
                            s1_strategy
                        )

                        envs.append(Locating(
                            max_stack=current_max_stack,
                            inbound_plates=train_schedule,
                            crane_penalty=cfg.crane_penalty,
                            min_obstacles=obs,
                            max_obstacles=obs,
                            observed_top_n_plates=cfg.OBSERVED_TOP_N_PLATES,
                            num_summary_stats_deeper=cfg.NUM_SUMMARY_STATS_DEEPER,
                            gamma=cfg.gamma,
                            max_steps=cfg.max_steps
                        ))

                        scenario_counter[scenario_type] += 1
                        s1_counter[s1_strategy] += 1
                        ratio_counter[str(cfg.initial_stock_ratio)] += 1

                        break

                    except Exception as e:
                        print("[TRAIN GEN ERROR]", e)
                        continue

            random.shuffle(envs)

            if epoch % 10 == 0:
                print("[TRAIN DIST] Scenario:", dict(scenario_counter))
                print("[TRAIN DIST] S1:", dict(s1_counter))
                print("[TRAIN DIST] InitialRatio:", dict(ratio_counter))
        # --- Rollout ---
        rollout_buffer = []

        reward_components = {
            "shape": [],
            "sev": [],
            "future": [],
            "step": [],
        }
        states = torch.stack([env.reset(shuffle_schedule=False) for env in envs]).to(device)
        episode_rewards = torch.zeros(cfg.num_envs, device=device)

        finished_episode_rewards, finished_episode_reversals, finished_episode_crane_moves, finished_episode_obstacles = [], [], [], []
        finished_episode_count = 0
        error_terminated_count = 0
        done_reason_counter = Counter()

        for t in range(cfg.T_horizon):
            global_step += 1
            s_masks_list, d_masks_list = zip(*[env.get_masks() for env in envs])
            source_mask_tensor = torch.stack(s_masks_list).to(device)
            dest_mask_tensor = torch.stack(d_masks_list).to(device)

            s_deadlock = ~source_mask_tensor.any(dim=1)
            d_deadlock = ~dest_mask_tensor.any(dim=1)
            actor_valid_tensor = ~(s_deadlock | d_deadlock)

            if s_deadlock.any(): source_mask_tensor[s_deadlock, 0] = True
            if d_deadlock.any(): dest_mask_tensor[d_deadlock, 0] = True

            with torch.no_grad():
                actions, logprobs, values, _ = model.act_batch(states, source_mask_tensor, dest_mask_tensor,
                                                               greedy=False)

            next_states_list = []
            for i, env in enumerate(envs):
                action_i = (actions[i][0].item(), actions[i][1].item())
                next_state, reward, done, info = env.step(action_i)
                reward_components["shape"].append(info.get("shaping_reward", 0.0))
                reward_components["sev"].append(info.get("severity_penalty", 0.0))
                reward_components["future"].append(info.get("future_conflict_penalty", 0.0))
                reward_components["step"].append(info.get("step_reward_raw", reward))

                rollout_buffer.append({
                    'state': states[i].clone().cpu(),
                    'action': actions[i].clone().cpu(),
                    'logprob': logprobs[i].clone().cpu(),
                    'value': values[i].clone().cpu(),
                    'reward': reward,
                    'done': done,
                    'source_mask': source_mask_tensor[i].clone().cpu(),
                    'dest_mask': dest_mask_tensor[i].clone().cpu(),
                    'actor_valid': actor_valid_tensor[i].clone().cpu()
                })

                episode_rewards[i] += reward
                next_states_list.append(next_state)

                if done:
                    finished_episode_count += 1
                    if isinstance(info, dict) and ("error" in info):
                        error_terminated_count += 1
                    done_reason_counter[str(info.get('episode_end_reason', 'unknown'))] += 1
                    finished_episode_rewards.append(episode_rewards[i].item())
                    final_rev = info.get("final_blocking_metric", None)

                    if final_rev is None:
                        final_rev = info.get("episode_max_blocking_metric", 0)

                    finished_episode_reversals.append(final_rev)
                    finished_episode_crane_moves.append(env.crane_move)
                    finished_episode_obstacles.append(env.num_obstacle_plates)
                    next_states_list[i] = env.reset(shuffle_schedule=False)
                    episode_rewards[i] = 0

            states = torch.stack(next_states_list).to(device)

        # --- GAE 계산 ---
        original_rewards_np = np.array([d['reward'] for d in rollout_buffer])
        reward_normalizer.update(original_rewards_np)
        normalized_rewards_np = reward_normalizer.normalize(original_rewards_np)

        with torch.no_grad():
            last_s_masks_list, last_d_masks_list = zip(*[env.get_masks() for env in envs])
            last_source_mask_tensor = torch.stack(last_s_masks_list).to(device)
            last_dest_mask_tensor = torch.stack(last_d_masks_list).to(device)

            last_s_deadlock = ~last_source_mask_tensor.any(dim=1)
            last_d_deadlock = ~last_dest_mask_tensor.any(dim=1)

            if last_s_deadlock.any():
                last_source_mask_tensor[last_s_deadlock, 0] = True
            if last_d_deadlock.any():
                last_dest_mask_tensor[last_d_deadlock, 0] = True

            _, _, last_values, _ = model.act_batch(
                states,
                last_source_mask_tensor,
                last_dest_mask_tensor,
                greedy=False
            )

        advantages = torch.zeros(len(rollout_buffer), device=device)
        gae = torch.zeros(cfg.num_envs, device=device)

        for t in reversed(range(cfg.T_horizon)):
            for i in range(cfg.num_envs):
                idx = t * cfg.num_envs + i
                is_last_step = (t == cfg.T_horizon - 1)
                next_value = last_values[i] if is_last_step else rollout_buffer[idx + cfg.num_envs]['value'].to(device)
                curr_value = rollout_buffer[idx]['value'].to(device)

                reward = normalized_rewards_np[idx]

                done_mask = 1.0 - rollout_buffer[idx]['done']
                delta = reward + cfg.gamma * next_value * done_mask - curr_value
                gae[i] = delta + cfg.gamma * cfg.lmbda * done_mask * gae[i]
                advantages[idx] = gae[i]

        all_values = torch.stack([rollout_buffer[i]['value'] for i in range(len(rollout_buffer))]).to(device)
        returns = advantages + all_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0.0
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy_loss = 0.0  # raw entropy loss
        total_entropy_contrib = 0.0  # weighted entropy contribution
        num_updates = 0

        # PPO 진단 로그용
        total_kl = 0.0
        total_clip_frac = 0.0
        total_ratio_mean = 0.0
        total_ratio_std = 0.0
        total_adv_std = 0.0
        total_valid_ratio = 0.0

        indices = np.arange(len(rollout_buffer))
        model.train()

        # --- PPO Update ---
        for _ in range(cfg.K_epoch):
            np.random.shuffle(indices)

            for start in range(0, len(rollout_buffer), cfg.mini_batch_size):
                minibatch_indices = indices[start:start + cfg.mini_batch_size]

                s_batch = torch.stack([
                    rollout_buffer[i]["state"] for i in minibatch_indices
                ]).to(device)

                a_batch = torch.stack([
                    rollout_buffer[i]["action"] for i in minibatch_indices
                ]).to(device)

                logprob_batch = torch.stack([
                    rollout_buffer[i]["logprob"] for i in minibatch_indices
                ]).to(device).view(-1)

                adv_batch = advantages[minibatch_indices].to(device).view(-1)
                td_target_batch = returns[minibatch_indices].to(device).view(-1)

                s_mask_batch = torch.stack([
                    rollout_buffer[i]["source_mask"] for i in minibatch_indices
                ]).to(device)

                d_mask_batch = torch.stack([
                    rollout_buffer[i]["dest_mask"] for i in minibatch_indices
                ]).to(device)

                new_logprob, value, entropy = model.evaluate(
                    s_batch,
                    s_mask_batch,
                    d_mask_batch,
                    a_batch
                )

                # -------------------------------------------------
                # 1. PPO 계산 전에 모든 tensor shape을 [B]로 통일
                # -------------------------------------------------
                new_logprob = new_logprob.view(-1)
                logprob_batch = logprob_batch.view(-1)

                value = value.view(-1)
                entropy = entropy.view(-1)

                adv_batch = adv_batch.view(-1)
                td_target_batch = td_target_batch.view(-1)

                # -------------------------------------------------
                # 2. 강제 shape 체크
                # -------------------------------------------------
                assert new_logprob.shape == logprob_batch.shape, \
                    f"logprob shape mismatch: new={new_logprob.shape}, old={logprob_batch.shape}"

                assert value.shape == td_target_batch.shape, \
                    f"value shape mismatch: value={value.shape}, target={td_target_batch.shape}"

                assert adv_batch.shape == logprob_batch.shape, \
                    f"advantage shape mismatch: adv={adv_batch.shape}, logprob={logprob_batch.shape}"

                # -------------------------------------------------
                # 3. ratio 계산
                # -------------------------------------------------
                ratio = torch.exp(new_logprob - logprob_batch)

                with torch.no_grad():
                    approx_kl = (logprob_batch - new_logprob).mean().item()
                    clip_frac = ((ratio - 1.0).abs() > cfg.eps_clip).float().mean().item()

                    total_kl += approx_kl
                    total_clip_frac += clip_frac
                    total_ratio_mean += ratio.mean().item()
                    total_ratio_std += ratio.std().item()
                    total_adv_std += adv_batch.std().item()

                # -------------------------------------------------
                # 4. valid mask
                # -------------------------------------------------
                valid_batch = torch.stack([
                    rollout_buffer[i]["actor_valid"] for i in minibatch_indices
                ]).float().to(device).view(-1)

                total_valid_ratio += valid_batch.mean().item()

                # -------------------------------------------------
                # 5. PPO actor loss
                # -------------------------------------------------
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(
                    ratio,
                    1.0 - cfg.eps_clip,
                    1.0 + cfg.eps_clip
                ) * adv_batch

                actor_loss_each = -torch.min(surr1, surr2)
                actor_loss = (actor_loss_each * valid_batch).sum() / (valid_batch.sum() + 1e-8)

                # -------------------------------------------------
                # 6. Critic / Entropy loss
                # -------------------------------------------------
                critic_loss = torch.nn.functional.smooth_l1_loss(value, td_target_batch)

                entropy_loss = -(entropy * valid_batch).sum() / (valid_batch.sum() + 1e-8)

                loss = (
                        actor_loss
                        + cfg.V_coeff * critic_loss
                        + cfg.E_coeff * entropy_loss
                )

                actor_optimizer.zero_grad(set_to_none=True)
                critic_optimizer.zero_grad(set_to_none=True)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)

                actor_optimizer.step()
                critic_optimizer.step()

                total_loss += loss.item()
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy_loss += entropy_loss.item()
                total_entropy_contrib += (cfg.E_coeff * entropy_loss).item()
                num_updates += 1

        actor_lr_sched.step()
        critic_lr_sched.step()

        avg_total_loss = total_loss / num_updates if num_updates > 0 else 0.0

        avg_actor_loss = total_actor_loss / num_updates if num_updates > 0 else 0.0
        avg_critic_loss = total_critic_loss / num_updates if num_updates > 0 else 0.0
        avg_entropy_loss = total_entropy_loss / num_updates if num_updates > 0 else 0.0

        # 실제 loss에 들어간 가중치 반영값
        avg_critic_contrib = cfg.V_coeff * avg_critic_loss
        avg_entropy_contrib = cfg.E_coeff * avg_entropy_loss

        # 출력 검산용: avg_total_loss와 거의 같아야 함
        avg_loss_check = avg_actor_loss + avg_critic_contrib + avg_entropy_contrib

        avg_kl = total_kl / num_updates if num_updates > 0 else 0.0
        avg_clip_frac = total_clip_frac / num_updates if num_updates > 0 else 0.0
        avg_ratio_mean = total_ratio_mean / num_updates if num_updates > 0 else 0.0
        avg_ratio_std = total_ratio_std / num_updates if num_updates > 0 else 0.0
        avg_adv_std = total_adv_std / num_updates if num_updates > 0 else 0.0
        avg_valid_ratio = total_valid_ratio / num_updates if num_updates > 0 else 0.0

        # --- Logging ---
        avg_ep_reward = np.mean(finished_episode_rewards) if finished_episode_rewards else 0.0
        avg_ep_reversals = np.mean(finished_episode_reversals) if finished_episode_reversals else 0.0
        avg_ep_obstacles = np.mean(finished_episode_obstacles) if finished_episode_obstacles else 0.0

        print(
            f"Epoch {epoch}: Reward={avg_ep_reward:.2f}, Reversals={avg_ep_reversals:.2f}, "
            f"Loss={avg_total_loss:.4f} "
            f"(Act:{avg_actor_loss:.3f} "
            f"CritW:{avg_critic_contrib:.3f} "
            f"EntW:{avg_entropy_contrib:.3f} "
            f"EntRaw:{avg_entropy_loss:.3f} "
            f"Check:{avg_loss_check:.3f}) "
            f"| EpisodesDone={finished_episode_count}, ErrorDone={error_terminated_count}, "
            f"DoneReasons={dict(done_reason_counter)}"
        )
        print(
            f"  PPOStat | "
            f"KL={avg_kl:.6f}, "
            f"ClipFrac={avg_clip_frac:.4f}, "
            f"RatioMean={avg_ratio_mean:.4f}, "
            f"RatioStd={avg_ratio_std:.4f}, "
            f"AdvStd={avg_adv_std:.4f}, "
            f"Valid={avg_valid_ratio:.4f}"
        )

        print(
            f"  RewardComp | "
            f"Shape={np.mean(reward_components['shape']):.4f}, "
            f"Sev={np.mean(reward_components['sev']):.4f}, "
            f"Fut={np.mean(reward_components['future']):.4f}, "
            f"Step={np.mean(reward_components['step']):.4f}"
        )
        tb_writer.add_scalar("Training/AverageReward", avg_ep_reward, epoch)
        tb_writer.add_scalar("Training/AvgFinalReversals", avg_ep_reversals, epoch)
        tb_writer.add_scalar("Training/EpisodesDone", finished_episode_count, epoch)
        tb_writer.add_scalar("Training/ErrorTerminatedEpisodes", error_terminated_count, epoch)
        tb_writer.add_scalar("Training/AvgObstacles", avg_ep_obstacles, epoch)
        tb_writer.add_scalar("Loss/TotalLoss", avg_total_loss, epoch)
        tb_writer.add_scalar("Loss/ActorLossRaw", avg_actor_loss, epoch)
        tb_writer.add_scalar("Loss/CriticLossRaw", avg_critic_loss, epoch)
        tb_writer.add_scalar("Loss/CriticLossWeighted", avg_critic_contrib, epoch)
        tb_writer.add_scalar("Loss/EntropyLossRaw", avg_entropy_loss, epoch)
        tb_writer.add_scalar("Loss/EntropyLossWeighted", avg_entropy_contrib, epoch)
        tb_writer.add_scalar("Loss/TotalLossCheck", avg_loss_check, epoch)
        tb_writer.flush()

        if USE_VESSL:
            vessl.log({"training_reward": avg_ep_reward, "training_reversal": avg_ep_reversals}, step=epoch)

        if epoch > 0 and epoch % cfg.save_every == 0:
            torch.save(model.state_dict(), os.path.join(cfg.save_model_dir, f"checkpoint_epoch_{epoch}.pth"))

        # --- 평가 루프 ---
        if epoch > 0 and epoch % cfg.eval_every == 0:

            train_rng_state = snapshot_rng_state()
            prev_initial_stock_ratio = cfg.initial_stock_ratio

            try:
                print(f"\n--- Evaluation Report (Epoch {epoch}) ---")
                EVAL_STACK_LIMIT = cfg.max_stack
                NUM_EVAL_EPISODES = 5

                eval_ratios = [1.0, 0.7]
                eval_strategies = ["Random"]
                grand_overall_scores = []

                for ratio in eval_ratios:
                    cfg.initial_stock_ratio = ratio

                    for strategy in eval_strategies:
                        combo_name = f"R{ratio}_{strategy}"
                        print(f"\n  >> [Test Combo: {combo_name}]")

                        fixed_overall, fixed_details = run_scenario_benchmark(
                            model=model,
                            device=device,
                            cfg=cfg,
                            stack_limit=EVAL_STACK_LIMIT,
                            scenarios=build_fixed_anchor_eval_scenarios(),
                            episodes=NUM_EVAL_EPISODES,
                            seed_prefix=base_seed + int(ratio * 100),
                            strategy=strategy
                        )

                        for name, nf, nt, obs, real_ppp, avg_score in fixed_details:
                            print(
                                f"    - [{name:<12}] {nf}->{nt} "
                                f"(Obs {obs}, PPP {real_ppp}) | Rev: {avg_score:6.2f}"
                            )
                            tb_writer.add_scalar(
                                f"EvalFixed_{combo_name}/{name}_Reversals",
                                avg_score,
                                epoch
                            )

                        tb_writer.add_scalar(
                            f"EvalFixed_{combo_name}/Overall",
                            fixed_overall,
                            epoch
                        )
                        grand_overall_scores.append(fixed_overall)

                final_grand_avg = np.mean(grand_overall_scores)
                print(f"\n [Grand Eval Average]: {final_grand_avg:.2f}")
                tb_writer.add_scalar(
                    "EvaluationFixed/Grand_Overall_Reversals",
                    final_grand_avg,
                    epoch
                )

                if final_grand_avg < best_metric:
                    print(f"New Best Model! ({best_metric:.2f} -> {final_grand_avg:.2f})")
                    best_metric = final_grand_avg
                    torch.save(
                        model.state_dict(),
                        os.path.join(cfg.save_model_dir, "best_policy.pth")
                    )
                else:
                    print(f"  (Best so far: {best_metric:.2f})")

                tb_writer.flush()

            finally:
                # eval에서 바꾼 seed와 cfg 값을 학습 흐름 기준으로 되돌린다.
                restore_rng_state(train_rng_state)
                cfg.initial_stock_ratio = prev_initial_stock_ratio

            continue

    # --- 최종 테스트 ---
    print("\n--- Final Test Report (Fixed Scenario) ---")
    test_stack_limit = cfg.max_stack
    cfg.initial_stock_ratio = 1.0
    test_episodes = 10
    test_scenarios = build_fixed_anchor_eval_scenarios()
    test_scores = []

    for scenario in test_scenarios:
        name = scenario['name']
        nf, nt, target_ppp, obs = scenario['nf'], scenario['nt'], scenario['ppp'], scenario['obs']
        scenario_seed = base_seed + stable_name_seed(name) % 1000
        scenario_metrics = []
        real_ppp_logged = None

        for ep in range(test_episodes):
            set_seed(scenario_seed + ep)
            test_schedule, real_ppp = generate_schedule_from_scenario(
                cfg=cfg, max_stack_limit=test_stack_limit, nf=nf, nt=nt,
                target_ppp=target_ppp, obs=obs, s1_strategy="Random"
            )
            real_ppp_logged = real_ppp
            if not test_schedule:
                scenario_metrics.append(1000.0)
                continue

            test_env = Locating(
                max_stack=test_stack_limit, inbound_plates=test_schedule,
                min_obstacles=obs, max_obstacles=obs,
                observed_top_n_plates=cfg.OBSERVED_TOP_N_PLATES,
                num_summary_stats_deeper=cfg.NUM_SUMMARY_STATS_DEEPER,
                gamma=cfg.gamma, max_steps=cfg.max_steps
            )
            _, test_metric, _ = evaluate_policy(model, [test_env], device, return_blocked=True)
            scenario_metrics.append(test_metric if test_metric != float('inf') else 1000.0)

        avg_score = np.mean(scenario_metrics) if scenario_metrics else 1000.0
        test_scores.append(avg_score)
        print(f"  - [{name:<12}] {nf}->{nt} (Obs {obs}, PPP {real_ppp_logged}) | Rev: {avg_score:6.2f}")
        tb_writer.add_scalar(f"Test/{name}_Reversals", avg_score, cfg.n_epoch)

    final_test_avg = np.mean(test_scores) if test_scores else float('inf')
    print(f"Final Test Avg Reversals: {final_test_avg:.2f}")
    tb_writer.add_scalar("Test/Overall_Avg_Reversals", final_test_avg, cfg.n_epoch)

    torch.save(model.state_dict(), os.path.join(cfg.save_model_dir, "final_policy.pth"))
    tb_writer.close()
    print("학습 완료.")

if __name__ == "__main__":
    main()
