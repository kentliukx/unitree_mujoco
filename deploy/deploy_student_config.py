from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


ROOT = Path(__file__).resolve().parents[1]
STUDENT_DIR = ROOT / "deploy" / "student"


@dataclass
class DeployStudentConfig:
    checkpoint: Optional[Path] = STUDENT_DIR / "model.pt"
    device: str = "auto"
    inference_backend: str = "tensorrt"
    tensorrt_engine: Optional[Path] = None
    tensorrt_depth_engine: Optional[Path] = None
    tensorrt_core_engine: Optional[Path] = None
    # Target time from policy.act() start to the active LowCmd DDS Write.
    tensorrt_post_inference_delay_s: float = 0.008

    # Keep this layout identical to Legged Gym's StudentActorCritic slices.
    obs_dim: int = 2714
    act_dim: int = 12
    proprio_history_len: int = 10
    height_dim: int = 21 * 11
    clip_observations: float = 100.0
    clip_actions: float = 10.0
    actor_hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    critic_hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    activation: str = "elu"
    history_latent_dim: int = 32
    depth_latent_dim: int = 32
    mixer_latent_dim: int = 45
    rnn_type: str = "gru"
    rnn_hidden_size: int = 45
    rnn_num_layers: int = 1
    rnn_latent_dim: int = 32
    ladder_obs_dim: int = 13

    mujoco_domain_id: int = 1
    mujoco_interface: str = "lo"
    real_domain_id: int = 0
    control_dt: float = 0.02
    startup_timeout_s: float = 5.0
    status_print_interval_s: float = 1.0
    inference_print_interval_s: float = 0.5
    obs_debug_print_interval_s: float = 0.5
    height_scan_rows: int = 21
    height_scan_cols: int = 11
    auto_stop_after_s: Optional[float] = -1
    fall_stop_enabled: bool = True
    fall_stop_projected_gravity_z_min: float = 0.0
    obs_debug: bool = False
    record: bool = False
    record_dir: Path = ROOT / "deploy" / "records"
    visualize_depth: bool = False
    release_motion_on_start: bool = True
    release_motion_timeout_s: float = 5.0
    release_motion_retry_s: float = 1.0
    release_motion_max_attempts: int = 5
    require_realsense_on_start: bool = True
    require_tactile_on_start: bool = True
    tactile_port: Optional[str] = None
    tactile_baud: int = 115200

    lowcmd_topic: str = "rt/lowcmd"
    lowstate_topic: str = "rt/lowstate"
    depth_topic: str = "rt/depthimage"
    reset_topic: str = "rt/mujoco_reset"
    clock_topic: str = "rt/mujoco_clock"

    goal_forward_m: float = 2.0
    goal_lateral_m: float = 0.0
    keyboard_goal_scale: float = 1.0
    keyboard_lateral_goal_scale: float = 0.2
    joystick_deadzone: float = 0.1
    joystick_print_interval_s: float = 0.5
    depth_min: float = 0.1
    depth_max: float = 2.0
    depth_width: int = 54
    depth_height: int = 36
    depth_pitch_deg: float = 15.0
    depth_camera_pos: List[float] = field(default_factory=lambda: [0.35, 0.0, 0.08])
    realsense_width: int = 480
    realsense_height: int = 270
    realsense_crop_cols_each_side: int = 5
    realsense_center_sample_size: int = 2
    realsense_downsample_mode: str = "min"
    realsense_fps: int = 30
    realsense_publish_every_n_frames: int = 1
    realsense_serial: Optional[str] = None

    action_scale: float = 1
    kp: float = 25.0
    kd: float = 0.5
    num_motor_idl_go: int = 20

    joint_order: List[str] = field(default_factory=lambda: [
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",
        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",
        "RL_hip_joint",
        "RL_thigh_joint",
        "RL_calf_joint",
        "RR_hip_joint",
        "RR_thigh_joint",
        "RR_calf_joint",
    ])
    sdk_motor_order: List[str] = field(default_factory=lambda: [
        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",
        "RR_hip_joint",
        "RR_thigh_joint",
        "RR_calf_joint",
        "RL_hip_joint",
        "RL_thigh_joint",
        "RL_calf_joint",
    ])
    default_joint_angles: Dict[str, float] = field(default_factory=lambda: {
        "FL_hip_joint": 0.1,
        "FL_thigh_joint": 0.7,
        "FL_calf_joint": -1.7,
        "FR_hip_joint": -0.1,
        "FR_thigh_joint": 0.7,
        "FR_calf_joint": -1.7,
        "RL_hip_joint": 0.1,
        "RL_thigh_joint": 0.7,
        "RL_calf_joint": -1.7,
        "RR_hip_joint": -0.1,
        "RR_thigh_joint": 0.7,
        "RR_calf_joint": -1.7,
    })
    sdk_to_urdf_offsets: Dict[str, float] = field(default_factory=lambda: {
        "FL_hip_joint": 0.0,
        "FL_thigh_joint": 0.0,
        "FL_calf_joint": 0.0,
        "FR_hip_joint": 0.0,
        "FR_thigh_joint": 0.0,
        "FR_calf_joint": 0.0,
        "RL_hip_joint": 0.0,
        "RL_thigh_joint": 0.0,
        "RL_calf_joint": 0.0,
        "RR_hip_joint": 0.0,
        "RR_thigh_joint": 0.0,
        "RR_calf_joint": 0.0,
    })


CONFIG = DeployStudentConfig()
