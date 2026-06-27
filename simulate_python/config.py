from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

ROBOT = "go2" # Robot name, "go2", "b2", "b2w", "h1", "go2w", "g1"
ROBOT_SCENE = str(ROOT / "deploy" / "go2" / "scene_terrain_leggedgym.xml") # Robot scene
DOMAIN_ID = 1 # Domain id
INTERFACE = "lo" # Interface 

USE_JOYSTICK = 0 # Simulate Unitree WirelessController using a gamepad
JOYSTICK_TYPE = "xbox" # support "xbox" and "switch" gamepad layout
JOYSTICK_DEVICE = 0 # Joystick number

PRINT_SCENE_INFORMATION = True # Print link, joint and sensors information of robot
ENABLE_ELASTIC_BAND = False # Virtual spring band, used for lifting h1

SIMULATE_DT = 0.001  # Need to be larger than the runtime of viewer.sync()
VIEWER_DT = 0.02  # 50 fps for viewer

DEPTH_TOPIC = "rt/depthimage"
RESET_TOPIC = "rt/mujoco_reset"
CLOCK_TOPIC = "rt/mujoco_clock"
CLOCK_UPDATE_DT = 0.02
DEPTH_WIDTH = 64
DEPTH_HEIGHT = 36
DEPTH_HFOV_DEG = 87.0
DEPTH_UPDATE_DT = 0.1
DEPTH_CAMERA_POS = [0.33, 0.0, 0.08]
DEPTH_CAMERA_PITCH_DEG = 15.0
