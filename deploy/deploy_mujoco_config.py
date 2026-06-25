from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEGGED_GYM_ROOT = ROOT.parent / "legged_gym"


@dataclass
class DeployMujocoConfig:
    scene: Path = ROOT / "deploy/go2/scene_terrain_leggedgym.xml"
    policy: str = "teacher"  # teacher | student
    checkpoint: Path | None = None
    device: str = "cpu"

    spawn_pos: list[float] = field(default_factory=lambda: [-2.0, 0.0, 0.35])
    goal: list[float] = field(default_factory=lambda: [2.361, 0.0, 1.268])
    goal_height: float = 0.918
    friction: float = 1

    enable_height_scan: bool = True
    enable_depth: bool = True
    depth_min: float = 0.1
    depth_max: float = 3.0
    depth_hfov_deg: float = 87.0
    camera_pitch_deg: float = 15.0

    ladder_bar_spacing: float = 0.2
    ladder_angle_deg: float = 35.0
    ladder_bar_y_scale: float = 0.65


CONFIG = DeployMujocoConfig()
