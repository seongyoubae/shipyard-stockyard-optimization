import argparse
import os
from stockyard.constants import ENCODER_TYPES, ENCODER_ALIASES


def get_cfg():
    parser = argparse.ArgumentParser(
        description="Steel Plate Selection RL Hyperparameters"
    )

    # 시각화 관련
    parser.add_argument(
        "--viz_source",
        type=str,
        default="file",
        choices=["generate", "file"],
        help="Source of data for visualization ('generate' or 'file')",
    )
    parser.add_argument(
        "--viz_data_path",
        type=str,
        default="output/reshuffle_plan(for eval).xlsx",
        help="Path to the Excel file for visualization if --viz_source is 'file'",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="outputs/policy.pt",
        help="Path to the pre-trained model file and visualization log",
    )
    parser.add_argument(
        "--load_model",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to load a pre-trained model",
    )

    # 학습 관련
    parser.add_argument(
        "--n_episode", type=int, default=100000, help="Number of training episodes"
    )
    parser.add_argument(
        "--embed_dim", type=int, default=512, help="Dimension of node embeddings"
    )
    parser.add_argument(
        "--num_heads", type=int, default=8, help="Number of multi-head attention heads"
    )
    parser.add_argument(
        "--encoder_type",
        type=str,
        default="pi_gru_add",
        choices=[*ENCODER_TYPES, *ENCODER_ALIASES],
        help="Type of context encoder for ablation study",
    )
    parser.add_argument(
        "--num_HGT_layers", type=int, default=2, help="Number of HGT layers"
    )
    parser.add_argument(
        "--num_actor_layers",
        type=int,
        default=3,
        help="Number of layers in the actor network",
    )
    parser.add_argument(
        "--num_critic_layers",
        type=int,
        default=4,
        help="Number of layers in the critic network",
    )
    parser.add_argument(
        "--temp_lr",
        type=float,
        default=0.01,
        help="Learning rate for temperature parameter",
    )
    parser.add_argument(
        "--target_entropy",
        type=float,
        default=2.0,
        help="Target entropy for automatic temperature adjustment",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0005,
        help="General learning rate (if not using separate actor/critic lr)",
    )
    parser.add_argument(
        "--actor_lr", type=float, default=0.0001, help="Learning rate for actor network"
    )
    parser.add_argument(
        "--critic_lr",
        type=float,
        default=0.0001,
        help="Learning rate for critic network",
    )
    parser.add_argument(
        "--lr_decay", type=float, default=0.99, help="Learning rate decay factor"
    )
    parser.add_argument(
        "--lr_step",
        type=int,
        default=2000,
        help="Step interval for learning rate decay",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0, help="L2 정규화(가중치 감쇠) 계수"
    )
    parser.add_argument("--gamma", type=float, default=0.995, help="Discount factor")
    parser.add_argument(
        "--lmbda", type=float, default=0.95, help="GAE parameter (lambda)"
    )
    parser.add_argument(
        "--eps_clip", type=float, default=0.25, help="Clipping parameter for PPO"
    )
    parser.add_argument(
        "--T_horizon",
        type=int,
        default=2048,
        help="Number of steps to collect per update, step in episode",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=16,
        help="Number of parallel environments to run",
    )
    parser.add_argument(
        "--mini_batch_size",
        type=int,
        default=1024,
        help="Fixed size of minibatches per PPO update",
    )
    parser.add_argument(
        "--K_epoch",
        type=int,
        default=5,
        help="Number of optimization epochs per update",
    )
    parser.add_argument(
        "--n_epoch", type=int, default=800, help="Total number of epochs"
    )
    parser.add_argument(
        "--P_coeff", type=float, default=1.0, help="Coefficient for policy loss"
    )
    parser.add_argument(
        "--V_coeff", type=float, default=0.5, help="Coefficient for value loss"
    )
    parser.add_argument(
        "--E_coeff", type=float, default=0.01, help="Coefficient for entropy loss"
    )
    parser.add_argument(
        "--eval_every", type=int, default=5, help="Evaluate every x episodes"
    )
    parser.add_argument(
        "--save_every", type=int, default=200, help="Save model every x episodes"
    )
    parser.add_argument(
        "--save_final_state_every",
        type=int,
        default=10000,
        help="Save final state every x episodes",
    )
    parser.add_argument(
        "--new_instance_every",
        type=int,
        default=1,
        help="Generate new scenarios every x episodes",
    )
    parser.add_argument(
        "--value_clip_range",
        type=float,
        default=0.25,
        help="Value clipping range for critic update",
    )
    parser.add_argument(
        "--grad_clip_norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm for model parameters",
    )
    parser.add_argument(
        "--shaping_reward_scale",
        type=float,
        default=1.0,
        help="Potential Reward Scaling Factor",
    )
    parser.add_argument(
        "--target_metric_value",
        type=float,
        default=0.0,
        help="Target metric value to stop training early (lower is better)",
    )
    parser.add_argument(
        "--OBSERVED_TOP_N_PLATES",
        type=int,
        default=10,
        help="Number of top plates' outbound values to observe in detail for state representation",
    )
    parser.add_argument(
        "--NUM_SUMMARY_STATS_DEEPER",
        type=int,
        default=4,
        help="Number of summary stats for plates deeper than OBSERVED_TOP_N_PLATES (e.g., count, min, max, mean outbound)",
    )

    # 파일 및 저장 관련
    parser.add_argument(
        "--plates_data_path",
        type=str,
        default="output/reshuffle_plan.xlsx",
        help="Path to the plates data Excel file",
    )
    parser.add_argument(
        "--evaluation_plates_data_path",
        type=str,
        default="output/reshuffle_plan(for eval1).xlsx",
        help="Path to the plates eval data Excel file",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="training_log.csv",
        help="Path for training log CSV file",
    )
    parser.add_argument(
        "--save_model_dir",
        type=str,
        default="saved_models",
        help="Directory to save models",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save output Excel files",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run the model on (cuda or cpu)",
    )
    parser.add_argument(
        "--tensorboard_dir",
        type=str,
        default="./runs",
        help="Directory to save Tensorboard logs",
    )

    # 환경 관련 파라미터
    parser.add_argument(
        "--num_pile",
        type=int,
        default=20,
        help="Number of piles (files) in the environment/action space",
    )
    parser.add_argument(
        "--max_stack", type=int, default=50, help="Maximum stack height for each pile"
    )  # 학습시 50으로 했고, TEST 시에는 30으로 하였음 강건성 검증을 위하여
    parser.add_argument(
        "--crane_penalty", type=float, default=0, help="Crane movement penalty"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=1000,
        help="Maximum steps per episode in dynamic mode",
    )
    parser.add_argument(
        "--outbound_update_from_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply outbound-date confirmation updates only while plate is in FROM (or pending inbound).",
    )
    #  초기 도착지 채움 비율
    parser.add_argument(
        "--initial_dest_fill_ratio",
        type=float,
        default=0.4,
        help="Ratio of plates to pre-fill in destination piles at reset (0.0 to 1.0)",
    )
    #  초기 무질서도 비율
    parser.add_argument(
        "--initial_disorder_ratio",
        type=float,
        default=0.0,
        help="Ratio of swaps to introduce disorder in the initial sorted state (0.0 to 1.0)",
    )
    parser.add_argument(
        "--min_obstacles", type=int, default=0, help="리셋 시 생성될 최소 방해물 개수"
    )
    parser.add_argument(
        "--max_obstacles", type=int, default=50, help="리셋 시 생성될 최대 방해물 개수"
    )
    parser.add_argument(
        "--num_obstacle_plates",
        type=int,
        default=40,
        help="[For Visualization] Specific number of obstacle plates in the environment.",
    )
    # 데이터 생성 관련
    parser.add_argument(
        "--num_plates",
        type=int,
        default=400,
        help="Number of plates for schedule generation",
    )
    parser.add_argument(
        "--inbound_min", type=int, default=1, help="Minimum inbound value"
    )
    parser.add_argument(
        "--inbound_max", type=int, default=10, help="Maximum inbound value"
    )
    parser.add_argument(
        "--initial_stock_ratio",
        type=float,
        default=0.7,
        help="Fraction of plates available at initial reset time",
    )
    parser.add_argument(
        "--initial_stock_window",
        type=int,
        default=1,
        help="Initial-stock inbound day spread from inbound_min",
    )
    parser.add_argument(
        "--apply_initial_stock_ratio_on_input_sample",
        action="store_true",
        help="If set, enforce initial_stock_ratio even when plate triplets are sampled from private input (excluded)",
    )
    parser.add_argument(
        "--dynamic_inbound_end",
        type=int,
        default=300,
        help="Latest inbound day target for dynamic arrivals",
    )
    parser.add_argument(
        "--lead_time_min",
        type=int,
        default=1,
        help="Minimum lead time days (outbound - inbound)",
    )
    parser.add_argument(
        "--lead_time_max",
        type=int,
        default=120,
        help="Maximum lead time days (outbound - inbound)",
    )
    parser.add_argument(
        "--lead_time_scale",
        type=float,
        default=1.0,
        help="Global scale factor for empirical lead-time sampling",
    )
    parser.add_argument(
        "--outbound_shock_ratio",
        type=float,
        default=-1.0,
        help="Extra stochastic outbound-change ratio; set -1 to auto-use generic synthetic setting",
    )
    parser.add_argument(
        "--outbound_shock_max_days",
        type=int,
        default=14,
        help="Max absolute days for additional outbound shock (e.g., 14 means about 2 weeks)",
    )
    parser.add_argument(
        "--apply_outbound_shock_on_input_sample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply additional outbound shock when sampling plate triplets from private input (excluded) (default: on). Use --no-apply_outbound_shock_on_input_sample to disable.",
    )
    parser.add_argument(
        "--outbound_extra_min",
        type=int,
        default=1,
        help="Minimum extra for outbound relative to inbound",
    )
    parser.add_argument(
        "--outbound_extra_max",
        type=int,
        default=20,
        help="Maximum extra for outbound relative to inbound",
    )
    parser.add_argument(
        "--unitw_min", type=float, default=1.0, help="Minimum unit weight"
    )
    parser.add_argument(
        "--unitw_max", type=float, default=10.0, help="Maximum unit weight"
    )

    # 재배치 계획 관련
    parser.add_argument(
        "--n_from_piles_reshuffle",
        type=int,
        default=30,
        help="Number of source piles for reshuffle",
    )
    parser.add_argument(
        "--n_to_piles_reshuffle",
        type=int,
        default=30,
        help="Number of destination piles for reshuffle",
    )
    parser.add_argument(
        "--n_plates_reshuffle",
        type=int,
        default=20,
        help="Average number of plates to reshuffle per pile",
    )
    parser.add_argument(
        "--safety_margin", type=int, default=0, help="Safety margin for reshuffle plan"
    )

    # 네트워크 초기화 관련
    parser.add_argument(
        "--actor_init_std",
        type=float,
        default=0.01,
        help="Standard deviation for actor head initialization",
    )
    parser.add_argument(
        "--critic_init_std",
        type=float,
        default=1.0,
        help="Standard deviation for critic head initialization",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="ELU",
        help="Activation function to use (e.g., elu, relu)",
    )

    # 타깃 critic 업데이트 관련 (추가)
    parser.add_argument(
        "--update_target_interval",
        type=int,
        default=10,
        help="Interval (in minibatches) for updating target critic network",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.01,
        help="Soft update coefficient for target critic network",
    )

    # Use parse_known_args so IDE/runtime-added unknown args do not break execution.
    # This avoids abrupt argparse exits in PyCharm/launcher environments.
    args, unknown_args = parser.parse_known_args()
    if unknown_args:
        print(f"[Info] Ignoring unknown args: {unknown_args}")

    # Vessl Web UI에서 설정된 하이퍼파라미터로 덮어쓰기
    for key, value in vars(args).items():
        env_key = f"VESSEL_PARAM_{key}"
        env_value = os.getenv(env_key)
        if env_value is not None:
            try:
                if isinstance(value, bool):
                    setattr(args, key, env_value.lower() == "true")
                elif isinstance(value, int):
                    setattr(args, key, int(env_value))
                elif isinstance(value, float):
                    setattr(args, key, float(env_value))
                else:
                    setattr(args, key, env_value)
            except Exception as e:
                print(
                    f"[Warning] Could not convert {env_key}='{env_value}' to type {type(value)}: {e}"
                )

    return args
