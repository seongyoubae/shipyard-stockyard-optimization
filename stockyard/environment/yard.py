import copy
import random
import numpy as np
import torch
from stockyard.data.generator import Plate
from stockyard.models.network import MAX_SOURCE, MAX_DEST


# === 키 정규화 함수 ===
def normalize_keys(schedule):
    """스케줄 내 파일(Pile) 키를 정규화하고 정규화된 키 목록 반환"""
    if not schedule: return [], [], []
    original_from_keys_set, original_to_keys_set, valid_schedule_input = set(), set(), []

    for p in schedule:
        if isinstance(p, Plate) and hasattr(p, 'from_pile') and p.from_pile is not None and \
                hasattr(p, 'topile') and p.topile is not None and \
                hasattr(p, 'outbound') and isinstance(getattr(p, 'outbound'), (int, float)):
            valid_schedule_input.append(p)
            original_from_keys_set.add(str(p.from_pile).strip())
            original_to_keys_set.add(str(p.topile).strip())

    if not valid_schedule_input or not original_from_keys_set or not original_to_keys_set:
        return [], [], []

    unique_original_from_list = sorted(list(original_from_keys_set))[:MAX_SOURCE]
    unique_original_to_list = sorted(list(original_to_keys_set))[:MAX_DEST]

    if not unique_original_from_list or not unique_original_to_list:
        unique_original_to_list = ["DEST_DUMMY"]

    while len(unique_original_to_list) < 1:
        unique_original_to_list.append("DEST_DUMMY")

    from_key_map = {key: f"from_{i:02d}" for i, key in enumerate(unique_original_from_list)}
    to_key_map = {key: f"to_{i:02d}" for i, key in enumerate(unique_original_to_list)}

    normalized_schedule_final = []
    default_norm_to_key = list(to_key_map.values())[0]

    for p in valid_schedule_input:
        norm_from = from_key_map.get(str(p.from_pile).strip())
        norm_to = to_key_map.get(str(p.topile).strip())
        if norm_from is not None:
            normalized_p = copy.copy(p)
            setattr(normalized_p, 'from_pile', norm_from)
            setattr(normalized_p, 'topile', norm_to if norm_to is not None else default_norm_to_key)
            normalized_schedule_final.append(normalized_p)

    return normalized_schedule_final, list(from_key_map.values()), list(to_key_map.values())


# === 강화학습 환경 클래스 ===
class Locating(object):
    def __init__(self, max_stack=50, inbound_plates=None, crane_penalty=0.0,
                 min_obstacles=0, max_obstacles=50,
                 observed_top_n_plates=10, num_summary_stats_deeper=4,
                 num_pile_type_features=1, num_blocking_features=1, reward_scale=1.0,
                 max_steps=1000,
                 outbound_update_from_only=True,
                 gamma=0.99,
                 max_lead_time_feature_scale=120.0,
                 reward_w_potential=1.0,
                 reward_w_terminal=0.0,
                 terminal_success_bonus=50.0,
                 terminal_overflow_penalty=-50.0):

        self.max_stack = int(max_stack)
        self.crane_penalty = float(crane_penalty)
        self.reward_scale = reward_scale
        self.reward_w_potential = float(reward_w_potential)
        self.reward_w_terminal = float(reward_w_terminal)
        self.gamma = float(gamma)
        self.max_lead_time_feature_scale = max(1.0, float(max_lead_time_feature_scale))
        self.terminal_success_bonus = float(terminal_success_bonus)
        self.terminal_overflow_penalty = float(terminal_overflow_penalty)

        self.min_obstacles = min_obstacles
        self.max_obstacles = max_obstacles
        self.num_obstacle_plates = 0

        self.OBSERVED_TOP_N_PLATES = observed_top_n_plates
        self.NUM_SUMMARY_STATS_DEEPER = num_summary_stats_deeper
        self.NUM_PILE_TYPE_FEATURES = num_pile_type_features
        self.NUM_BLOCKING_FEATURES = num_blocking_features

        # [next_inbound_ttd_1, next_outbound_ttd_1, next_inbound_ttd_2, next_outbound_ttd_2]
        self.NUM_NEXT_INBOUND_FEATURES = 4
        self.NUM_TIME_FEATURES = 1
        self.config_max_steps = int(max_steps)
        self.max_steps = int(max_steps)
        self.outbound_update_from_only = bool(outbound_update_from_only)

        # Feature Dimension 계산
        self.actual_pile_feature_dim = sum([
            self.OBSERVED_TOP_N_PLATES,
            self.NUM_SUMMARY_STATS_DEEPER,
            self.NUM_NEXT_INBOUND_FEATURES,
            self.NUM_TIME_FEATURES,
            self.NUM_PILE_TYPE_FEATURES,
            self.NUM_BLOCKING_FEATURES
        ])

        self.deeper_stats_start_idx = self.OBSERVED_TOP_N_PLATES
        self.next_inbound_start_idx = self.deeper_stats_start_idx + self.NUM_SUMMARY_STATS_DEEPER
        self.time_feature_idx = self.next_inbound_start_idx + self.NUM_NEXT_INBOUND_FEATURES
        self.pile_type_feature_idx = self.time_feature_idx + self.NUM_TIME_FEATURES
        self.blocking_feature_idx = self.pile_type_feature_idx + self.NUM_PILE_TYPE_FEATURES

        if inbound_plates is None:
            schedule_to_process = []
        else:
            schedule_to_process = copy.deepcopy(inbound_plates)

        for i, p in enumerate(schedule_to_process):
            if not hasattr(p, 'from_pile') or p.from_pile is None: p.from_pile = f"S_{i % MAX_SOURCE}"
            if not hasattr(p, 'topile') or p.topile is None: p.topile = f"D_{i % MAX_DEST}"

        normalized_schedule, self.from_keys, self.to_keys = normalize_keys(schedule_to_process)

        if not self.from_keys or not self.to_keys:
            self.from_keys = [f"from_{i:02d}" for i in range(1)]
            self.to_keys = [f"to_{i:02d}" for i in range(1)]
            normalized_schedule = []

        self.inbound_plates = normalized_schedule
        self.inbound_clone = copy.deepcopy(self.inbound_plates)
        self.all_pile_keys = sorted(list(set(self.from_keys + self.to_keys)))
        self.total_piles = len(self.all_pile_keys)
        self.source_index_to_key = {i: key for i, key in enumerate(self.from_keys)}
        self.dest_index_to_key = {i: key for i, key in enumerate(self.to_keys)}

        all_outbounds = [p.outbound for p in self.inbound_clone if hasattr(p, 'outbound')]
        if all_outbounds:
            self.min_outbound = min(all_outbounds)
            self.max_outbound = max(all_outbounds)
        else:
            self.min_outbound = 0
            self.max_outbound = 1

        self.plates, self.stage, self.crane_move, self.total_plate_count, self.move_data = {}, 0, 0, 0, []
        self.pending_inbound_events = []
        self.pending_inbound_plate = None
        self.pending_outbound_updates = []
        self.current_time = 0
        self.total_departed_count = 0
        self.total_arrived_count = 0
        self.total_outbound_updates = 0
        self.overflowed = False
        self.episode_max_blocking_metric = 0
        self.episode_max_blocked_outbound = 0
        self.episode_sum_blocked_outbound = 0.0

    # 💡 [핵심 패치 1] 절대 Outbound 정규화 함수
    def _normalize_outbound(self, outbound):
        denom = max(1.0, float(self.max_outbound - self.min_outbound))
        value = (float(outbound) - float(self.min_outbound)) / denom
        return float(np.clip(2.0 * value - 1.0, -1.0, 1.0))

    def reset(self, shuffle_schedule=False):
        self.num_obstacle_plates = random.randint(self.min_obstacles, self.max_obstacles)
        schedule = copy.deepcopy(self.inbound_clone)
        if shuffle_schedule:
            random.shuffle(schedule)

        self.max_steps = max(1, int(self.config_max_steps))
        self.plates = {key: [] for key in self.all_pile_keys}

        if self.num_obstacle_plates > 0 and self.to_keys:
            max_task_outbound = 0
            if schedule:
                max_task_outbound = max(p.outbound for p in schedule if hasattr(p, 'outbound'))

            obstacle_plates = []
            base_outbound = max_task_outbound + 1
            for i in range(self.num_obstacle_plates):
                obstacle_plates.append(
                    Plate(id=f"OBS_{i:03d}", inbound=1, outbound=base_outbound + i, unitw=10.0)
                )

            obstacle_plates.sort(key=lambda p: p.outbound, reverse=True)
            for i, p in enumerate(obstacle_plates):
                chosen_dest_key = self.to_keys[i % len(self.to_keys)]
                if len(self.plates[chosen_dest_key]) < self.max_stack:
                    self.plates[chosen_dest_key].append(p)

        self.pending_outbound_updates = []
        self.total_outbound_updates = 0

        for idx, p in enumerate(schedule):
            p._init_idx = idx

        self.pending_inbound_events = sorted(
            [p for p in schedule if hasattr(p, 'inbound')],
            key=lambda x: (x.inbound, getattr(x, '_init_idx', 0))
        )

        self.current_time = int(self.pending_inbound_events[0].inbound) if self.pending_inbound_events else 0

        inbound_added, overflow = self._process_dynamic_inbound()

        self.total_arrived_count = inbound_added
        self.total_departed_count = 0
        self.overflowed = bool(overflow)
        self.episode_max_blocking_metric = 0
        self.episode_max_blocked_outbound = 0
        self.episode_sum_blocked_outbound = 0.0

        self.crane_move, self.stage = 0, 0
        self.total_plate_count = self.total_arrived_count
        self.move_data = []

        return self._get_state()

    def step(self, action):
        from_index, to_index = action
        valid_source_mask, valid_dest_mask = self.get_masks()
        info = {
            "inbound_added": 0, "outbound_updates": 0, "outbound_removed": 0,
            "blocked_outbound": 0, "overflow": False, "current_time": self.current_time,
            "episode_end_reason": None, "idle_wait": False,
        }

        active_source_exists = bool(valid_source_mask[:len(self.from_keys)].any())
        active_dest_exists = bool(valid_dest_mask[:len(self.to_keys)].any())
        step_reward = 0.0

        if active_source_exists and (not active_dest_exists):
            info["overflow"] = True
            info["episode_end_reason"] = "overflow"
            return self._get_state(), float(self.terminal_overflow_penalty * self.reward_scale), True, info

        if active_source_exists:
            if not (0 <= from_index < len(self.from_keys) and valid_source_mask[from_index] and
                    0 <= to_index < len(self.to_keys) and valid_dest_mask[to_index]):
                return self._get_state(), -1.0, True, {"error": "Invalid action"}

            source_key, destination_key = self.from_keys[from_index], self.to_keys[to_index]
            potential_before = -sum(self._get_total_blocking_pairs(self.plates.get(k, [])) for k in self.to_keys)

            if not self.plates.get(source_key):
                return self._get_state(), -1.0, True, {"error": "Source empty"}

            moved_plate = self.plates[source_key].pop()
            self.plates[destination_key].append(moved_plate)

            dest_pile_after = self.plates[destination_key]
            potential_after = -sum(self._get_total_blocking_pairs(self.plates.get(k, [])) for k in self.to_keys)

            severity_penalty = 0.0
            blocking_created = 0

            for plate_below in dest_pile_after[:-1]:
                if plate_below.outbound < moved_plate.outbound:
                    gap = moved_plate.outbound - plate_below.outbound
                    blocking_created += 1
                    severity_penalty += (1.0 + gap * 0.1)

            shaping_reward = potential_after - potential_before
            step_reward = shaping_reward - 0.5 * severity_penalty - 2 * blocking_created

            info["shaping_reward"] = float(shaping_reward)
            info["severity_penalty"] = float(severity_penalty)
            info["blocking_created"] = int(blocking_created)
            info["step_reward_raw"] = float(step_reward)

            self.move_data.append((source_key, destination_key))
            self.crane_move += 1
        else:
            info["idle_wait"] = True
            if self.pending_inbound_events:
                next_inbound_time = int(self.pending_inbound_events[0].inbound)
                if next_inbound_time > self.current_time:
                    self.current_time = next_inbound_time - 1

        self.stage += 1
        self.current_time += 1
        inbound_added, overflow = self._process_dynamic_inbound()

        info["inbound_added"] = inbound_added
        info["overflow"] = overflow
        self.total_arrived_count += inbound_added
        self.total_plate_count = self.total_arrived_count

        done = False
        terminal_reward = 0.0

        if overflow:
            done = True
            info["episode_end_reason"] = "overflow"
            terminal_reward = self.terminal_overflow_penalty
        elif (not self.pending_inbound_events) and self._count_source_plates() == 0:
            done = True
            info["episode_end_reason"] = "from_cleared"
            final_blocking_metric = sum(
                self._get_total_blocking_pairs(self.plates.get(key, [])) for key in self.to_keys)
            info["final_blocking_metric"] = final_blocking_metric

            max_possible_blocking = 0
            for key in self.to_keys:
                pile_len = len(self.plates.get(key, []))
                max_possible_blocking += pile_len * (pile_len - 1) / 2
            max_possible_blocking = max(max_possible_blocking, 1.0)
            final_blocking_ratio = final_blocking_metric / max_possible_blocking

            info["final_blocking_ratio"] = float(final_blocking_ratio)
            terminal_reward = 10.0 - 100.0 * final_blocking_ratio

        elif self.stage >= self.max_steps:
            done = True
            info["episode_end_reason"] = "max_steps"

        info["episode_max_blocking_metric"] = sum(
            self._get_total_blocking_pairs(self.plates.get(key, [])) for key in self.to_keys)
        return self._get_state(), (step_reward + terminal_reward) * self.reward_scale, done, info

    def _get_total_blocking_pairs(self, pile):
        if len(pile) <= 1: return 0
        total_blocking_pairs = 0
        outbounds = [p.outbound for p in pile]
        for i in range(len(outbounds)):
            for j in range(i + 1, len(outbounds)):
                if outbounds[i] < outbounds[j]:
                    total_blocking_pairs += 1
        return total_blocking_pairs

    def _refresh_outbound_bounds(self, plate):
        self.min_outbound = min(self.min_outbound, plate.outbound)
        self.max_outbound = max(self.max_outbound, plate.outbound)

    def _count_source_plates(self):
        return sum(len(self.plates.get(key, [])) for key in self.from_keys)

    def _process_dynamic_inbound(self):
        inbound_added = 0
        overflow = False

        while self.pending_inbound_events and int(self.pending_inbound_events[0].inbound) <= self.current_time:
            plate = self.pending_inbound_events.pop(0)
            is_barge = (len(self.from_keys) == 1)

            available_sources = [
                k for k in self.from_keys
                if len(self.plates.get(k, [])) < self.max_stack or is_barge
            ]

            if not available_sources:
                overflow = True
                self.overflowed = True
                break

            preferred_source = plate.from_pile if plate.from_pile in self.from_keys else None
            if preferred_source in available_sources:
                source_key = preferred_source
            else:
                source_key = random.choice(available_sources)
                plate.from_pile = source_key

            self.plates[source_key].append(plate)
            self._refresh_outbound_bounds(plate)
            inbound_added += 1

        self.pending_inbound_plate = self.pending_inbound_events[0] if self.pending_inbound_events else None
        return inbound_added, overflow

    def _get_state(self):
        state_features = []
        EMPTY_VALUE = 0.0
        padding_feature = [EMPTY_VALUE] * self.actual_pile_feature_dim
        padding_feature[self.blocking_feature_idx] = 0.0

        MAX_BLOCKING_ESTIMATE = 100.0
        MAX_DEEPER_COUNT_ESTIMATE = max(1, self.max_stack - self.OBSERVED_TOP_N_PLATES)
        normalized_time = min(max(float(self.current_time) / max(1.0, float(self.max_steps)), 0.0), 1.0)
        lead_scale = self.max_lead_time_feature_scale

        next_inbound_map = {k: [] for k in self.from_keys}
        for p in self.pending_inbound_events:
            target_pile = getattr(p, 'from_pile', None)
            if target_pile in next_inbound_map and len(next_inbound_map[target_pile]) < 2:
                next_inbound_map[target_pile].append(p)

        for pile_keys, type_val in [(self.from_keys, 1.0), (self.to_keys, 2.0)]:
            max_piles = MAX_SOURCE if type_val == 1.0 else MAX_DEST
            for i in range(max_piles):
                feature_vector_list = list(padding_feature)

                if i < len(pile_keys):
                    pile_key = pile_keys[i]
                    feature_vector_list[self.time_feature_idx] = normalized_time
                    feature_vector_list[self.pile_type_feature_idx] = type_val

                    if type_val == 1.0:
                        future_plates = next_inbound_map.get(pile_key, [])

                        if len(future_plates) > 0:
                            p1 = future_plates[0]
                            in_ttd_1 = float(p1.inbound) - self.current_time
                            feature_vector_list[self.next_inbound_start_idx] = float(
                                np.clip(in_ttd_1 / lead_scale, -1.0, 1.0))
                            feature_vector_list[self.next_inbound_start_idx + 1] = self._normalize_outbound(
                                p1.outbound)
                        else:
                            feature_vector_list[self.next_inbound_start_idx] = EMPTY_VALUE
                            feature_vector_list[self.next_inbound_start_idx + 1] = EMPTY_VALUE

                        if len(future_plates) > 1:
                            p2 = future_plates[1]
                            in_ttd_2 = float(p2.inbound) - self.current_time
                            feature_vector_list[self.next_inbound_start_idx + 2] = float(
                                np.clip(in_ttd_2 / lead_scale, -1.0, 1.0))
                            feature_vector_list[self.next_inbound_start_idx + 3] = self._normalize_outbound(
                                p2.outbound)
                        else:
                            feature_vector_list[self.next_inbound_start_idx + 2] = EMPTY_VALUE
                            feature_vector_list[self.next_inbound_start_idx + 3] = EMPTY_VALUE
                    else:
                        feature_vector_list[
                        self.next_inbound_start_idx:self.next_inbound_start_idx + 4] = [EMPTY_VALUE] * 4

                    pile = self.plates.get(pile_key, [])
                    if pile:
                        outbounds_top = [p.outbound for p in reversed(pile)][:self.OBSERVED_TOP_N_PLATES]

                        for idx, ob in enumerate(outbounds_top):
                            feature_vector_list[idx] = self._normalize_outbound(ob)

                        if len(pile) > self.OBSERVED_TOP_N_PLATES:
                            deeper_plates = pile[:-self.OBSERVED_TOP_N_PLATES]
                            deeper_outbounds = [float(p.outbound) for p in deeper_plates]

                            if deeper_outbounds:
                                start_idx = self.deeper_stats_start_idx
                                feature_vector_list[start_idx] = len(deeper_plates) / MAX_DEEPER_COUNT_ESTIMATE
                                feature_vector_list[start_idx + 1] = self._normalize_outbound(
                                    np.min(deeper_outbounds))
                                feature_vector_list[start_idx + 2] = self._normalize_outbound(
                                    np.max(deeper_outbounds))
                                feature_vector_list[start_idx + 3] = self._normalize_outbound(
                                    np.mean(deeper_outbounds))

                    blocking_pairs = self._get_total_blocking_pairs(pile)
                    feature_vector_list[self.blocking_feature_idx] = min(
                        float(blocking_pairs) / MAX_BLOCKING_ESTIMATE, 1.0)

                state_features.append(feature_vector_list)

        return torch.tensor(state_features, dtype=torch.float)
    def get_masks(self):
        source_flags = [bool(self.plates.get(key)) for key in self.from_keys] + [False] * (
                    MAX_SOURCE - len(self.from_keys))
        dest_flags = [(len(self.plates.get(key, [])) < self.max_stack) for key in self.to_keys] + [False] * (
                    MAX_DEST - len(self.to_keys))
        return torch.tensor(source_flags, dtype=torch.bool), torch.tensor(dest_flags, dtype=torch.bool)
