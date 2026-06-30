import sys
import multiprocessing as mp
import queue
import struct
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

try:
    import torch
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
    from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import HeightMap_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
    from unitree_sdk2py.utils.crc import CRC
    try:
        from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
        from unitree_sdk2py.go2.sport.sport_client import SportClient
    except ImportError:
        MotionSwitcherClient = None
        SportClient = None
except ImportError as exc:
    raise SystemExit(
        "Missing runtime dependency. Run this script in the deployment environment "
        "with `torch` and `unitree_sdk2py` installed."
    ) from exc


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
RSL_RL_CANDIDATES = (ROOT / "rsl_rl", ROOT.parent / "rsl_rl")
STUDENT_DIR = ROOT / "deploy" / "student"

for path in (SCRIPT_DIR, *RSL_RL_CANDIDATES):
    if not path.exists():
        continue
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from deploy_student_config import CONFIG
from rsl_rl.modules.actor_critic_recurrent import ActorCriticRecurrent


def show_depth_camera(frame_queue, min_depth, max_depth, height, width):
    import matplotlib.pyplot as plt

    plt.ion()
    figure, axis = plt.subplots(num="Student depth camera")
    image = axis.imshow(
        np.zeros((height, width), dtype=np.float32),
        cmap="turbo",
        vmin=min_depth,
        vmax=max_depth,
    )
    axis.set_title("Forward depth [m]")
    axis.axis("off")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.show()

    while plt.fignum_exists(figure.number):
        try:
            frame = frame_queue.get(timeout=0.05)
        except queue.Empty:
            plt.pause(0.001)
            continue
        if frame is None:
            break
        image.set_data(frame)
        image.set_clim(vmin=min_depth, vmax=max_depth)
        axis.set_title(f"Forward depth [m]  min={frame.min():.2f} max={frame.max():.2f}")
        figure.canvas.draw_idle()
        figure.canvas.flush_events()
    plt.close(figure)


def quat_conjugate(quat):
    return np.array([quat[0], -quat[1], -quat[2], -quat[3]], dtype=np.float64)


def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def quat_rotate(quat, vec):
    qvec = np.array([0.0, vec[0], vec[1], vec[2]], dtype=np.float64)
    return quat_multiply(quat_multiply(quat, qvec), quat_conjugate(quat))[1:]


def quat_rotate_inverse(quat, vec):
    return quat_rotate(quat_conjugate(quat), vec)


def default_checkpoint():
    model_path = STUDENT_DIR / "model.pt"
    if model_path.exists():
        return model_path
    checkpoints = sorted(
        STUDENT_DIR.glob("*.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if checkpoints:
        return checkpoints[0]
    raise FileNotFoundError(
        f"No student checkpoint found in {STUDENT_DIR}. "
        "Copy your student .pt there, preferably as model.pt."
    )


def resolve_config():
    cfg = CONFIG
    if cfg.checkpoint is None:
        cfg.checkpoint = default_checkpoint()
    cfg.checkpoint = Path(cfg.checkpoint).resolve()
    if not cfg.checkpoint.exists():
        fallback = default_checkpoint()
        if cfg.checkpoint == fallback.resolve():
            raise FileNotFoundError(
                f"No student checkpoint found at {cfg.checkpoint}. "
                "Copy your student .pt to deploy/student/model.pt."
            )
        cfg.checkpoint = fallback.resolve()
    return cfg


class LatestBuffer:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest = None
        self.event = threading.Event()

    def handler(self, msg):
        with self.lock:
            self.latest = msg
        self.event.set()

    def get(self):
        with self.lock:
            return self.latest


class RealSenseDepthReader:
    def __init__(self, cfg, update_callback):
        self.cfg = cfg
        self.update_callback = update_callback
        self.stop_event = threading.Event()
        self.thread = None
        self.error = None

    def start(self):
        if self.thread is not None:
            return
        self.thread = threading.Thread(target=self._run, name="RealSenseDepthReader", daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=1.0)

    def _run(self):
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            self.error = exc
            print("[depth] pyrealsense2 is not installed; real depth will stay at max range.", flush=True)
            return

        pipeline = rs.pipeline()
        config = rs.config()
        if self.cfg.realsense_serial:
            config.enable_device(str(self.cfg.realsense_serial))
        config.enable_stream(
            rs.stream.depth,
            int(self.cfg.realsense_width),
            int(self.cfg.realsense_height),
            rs.format.z16,
            int(self.cfg.realsense_fps),
        )

        try:
            profile = pipeline.start(config)
            depth_sensor = profile.get_device().first_depth_sensor()
            depth_scale = float(depth_sensor.get_depth_scale())
            device = profile.get_device()
            try:
                name = device.get_info(rs.camera_info.name)
                serial = device.get_info(rs.camera_info.serial_number)
                print(f"[depth] RealSense started: {name} serial={serial} scale={depth_scale:g}", flush=True)
            except Exception:
                print(f"[depth] RealSense started: scale={depth_scale:g}", flush=True)

            frame_count = 0
            publish_every = max(int(self.cfg.realsense_publish_every_n_frames), 1)
            while not self.stop_event.is_set():
                try:
                    frames = pipeline.wait_for_frames(timeout_ms=1000)
                except RuntimeError:
                    continue
                depth_frame = frames.get_depth_frame()
                if not depth_frame:
                    continue
                frame_count += 1
                if frame_count % publish_every != 0:
                    continue
                depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale
                depth = self._resize_depth(depth)
                depth = np.nan_to_num(
                    depth,
                    nan=self.cfg.depth_max,
                    posinf=self.cfg.depth_max,
                    neginf=self.cfg.depth_max,
                )
                depth[depth <= 0.0] = self.cfg.depth_max
                depth = np.clip(depth, self.cfg.depth_min, self.cfg.depth_max)
                self.update_callback(depth.reshape(-1).astype(np.float32))
        except Exception as exc:
            self.error = exc
            print(f"[depth] RealSense thread stopped with error: {exc}", flush=True)
        finally:
            try:
                pipeline.stop()
            except Exception:
                pass

    def _resize_depth(self, depth):
        src_h, src_w = depth.shape
        dst_h = int(self.cfg.depth_height)
        dst_w = int(self.cfg.depth_width)
        if src_h == dst_h and src_w == dst_w:
            return depth

        if src_h % dst_h == 0 and src_w % dst_w == 0:
            block_h = src_h // dst_h
            block_w = src_w // dst_w
            blocks = depth.reshape(dst_h, block_h, dst_w, block_w)
            center_y0 = max(block_h // 2 - 1, 0)
            center_y1 = min(center_y0 + 2, block_h)
            center_x0 = max(block_w // 2 - 1, 0)
            center_x1 = min(center_x0 + 2, block_w)
            center = blocks[:, center_y0:center_y1, :, center_x0:center_x1]
            valid = center > 0.0
            summed = np.where(valid, center, 0.0).sum(axis=(1, 3))
            counts = valid.sum(axis=(1, 3))
            return np.where(counts > 0, summed / np.maximum(counts, 1), self.cfg.depth_max)

        y_idx = np.linspace(0, src_h - 1, dst_h).astype(np.int32)
        x_idx = np.linspace(0, src_w - 1, dst_w).astype(np.int32)
        return depth[np.ix_(y_idx, x_idx)]


class StudentPolicy:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.module = ActorCriticRecurrent(
            num_actor_obs=cfg.obs_dim,
            num_critic_obs=cfg.obs_dim,
            num_actions=cfg.act_dim,
            actor_hidden_dims=cfg.actor_hidden_dims,
            critic_hidden_dims=cfg.critic_hidden_dims,
            activation=cfg.activation,
            history_length=cfg.proprio_history_len,
            history_latent_dim=cfg.history_latent_dim,
            depth_latent_dim=cfg.depth_latent_dim,
            mixer_latent_dim=cfg.mixer_latent_dim,
            rnn_type=cfg.rnn_type,
            rnn_hidden_size=cfg.rnn_hidden_size,
            rnn_num_layers=cfg.rnn_num_layers,
        ).to(self.device)
        checkpoint = torch.load(cfg.checkpoint, map_location="cpu")
        self.module.load_state_dict(checkpoint["model_state_dict"], strict=False)
        self.module.eval()

    def reset(self):
        self.module.reset()

    @torch.inference_mode()
    def act(self, obs_np):
        obs = torch.from_numpy(obs_np).to(self.device).unsqueeze(0)
        action = self.module.act_inference(obs)
        return action.squeeze(0).detach().cpu().numpy()


class GoalCommandSource:
    def __init__(self, cfg, source):
        self.cfg = cfg
        self.source = source
        self.goal = np.array([cfg.goal_forward_m, cfg.goal_lateral_m], dtype=np.float32)
        self.reached_goal = 0.0
        self.pygame = None
        self.reset_requested = False
        self.stop_requested = False
        self._reset_key_was_down = False
        self._select_was_down = False
        self.last_joystick_print_time = 0.0
        self.latest_buttons = {}
        if self.source == "keyboard":
            import pygame

            self.pygame = pygame
            pygame.init()
            pygame.display.set_mode((360, 120))
            pygame.display.set_caption("Goal keyboard control: W/S A/D")
        elif self.source != "joystick":
            raise ValueError(f"Unsupported goal source: {self.source}")

    def update_from_low_state(self, low_state):
        if self.source == "keyboard":
            self._update_keyboard()
        else:
            self._update_joystick(low_state)

    def close(self):
        if self.pygame is not None:
            self.pygame.quit()

    def _update_keyboard(self):
        pygame = self.pygame
        pygame.event.pump()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.goal[:] = 0.0
                self.reached_goal = 1.0
                return

        keys = pygame.key.get_pressed()
        forward = bool(keys[pygame.K_w])
        backward = bool(keys[pygame.K_s])
        left = bool(keys[pygame.K_a])
        right = bool(keys[pygame.K_d])
        reset_key = bool(keys[pygame.K_r])

        x_scale = float(self.cfg.keyboard_goal_scale)
        y_scale = float(self.cfg.keyboard_lateral_goal_scale)
        x = x_scale * (float(forward) - float(backward))
        y = y_scale * (float(left) - float(right))
        any_pressed = forward or backward or left or right

        self.goal[:] = (x, y)
        self.reached_goal = 0.0 if any_pressed else 1.0
        self.reset_requested = reset_key and not self._reset_key_was_down
        self._reset_key_was_down = reset_key

    def _update_joystick(self, low_state=None):
        if low_state is not None:
            raw_lateral_axis, raw_forward_axis = self._parse_low_state_joystick(low_state)
        else:
            self.goal[:] = 0.0
            self.reached_goal = 1.0
            self.reset_requested = False
            return

        deadzone = float(self.cfg.joystick_deadzone)
        lateral_axis = raw_lateral_axis
        forward_axis = raw_forward_axis
        if abs(forward_axis) < deadzone:
            forward_axis = 0.0
        if abs(lateral_axis) < deadzone:
            lateral_axis = 0.0

        x = float(self.cfg.keyboard_goal_scale) * forward_axis
        y = -float(self.cfg.keyboard_lateral_goal_scale) * lateral_axis
        self.goal[:] = (x, y)
        self.reached_goal = 1.0 if x == 0.0 and y == 0.0 else 0.0
        self.reset_requested = False
        select_down = bool(self.latest_buttons.get("Select", 0))
        if select_down and not self._select_was_down:
            self.stop_requested = not self.stop_requested
            mode = "entering" if self.stop_requested else "leaving"
            print(f"[student] Select pressed: {mode} stop mode", flush=True)
        self._select_was_down = select_down
        self._maybe_print_joystick(raw_lateral_axis, raw_forward_axis, lateral_axis, forward_axis, x, y)

    def _maybe_print_joystick(self, raw_lateral_axis, raw_forward_axis, lateral_axis, forward_axis, x, y):
        now = time.perf_counter()
        if now - self.last_joystick_print_time < float(self.cfg.joystick_print_interval_s):
            return
        self.last_joystick_print_time = now
        print(
            "[joystick] "
            f"raw_lx={raw_lateral_axis:.3f} raw_ly={raw_forward_axis:.3f} "
            f"lateral={lateral_axis:.3f} forward={forward_axis:.3f} "
            f"goal=({x:.3f},{y:.3f}) "
            f"buttons={self._format_buttons(self.latest_buttons)}",
            flush=True,
        )

    def _parse_low_state_joystick(self, low_state):
        data = bytes(low_state.wireless_remote)
        if len(data) < 24:
            return 0.0, 0.0
        try:
            lx = struct.unpack_from("<f", data, 4)[0]
            ly = struct.unpack_from("<f", data, 20)[0]
        except struct.error:
            return 0.0, 0.0
        self.latest_buttons = self._parse_remote_buttons(data)
        return float(np.clip(lx, -1.0, 1.0)), float(np.clip(ly, -1.0, 1.0))

    @staticmethod
    def _parse_remote_buttons(data):
        if len(data) < 4:
            return {}
        data1 = data[2]
        data2 = data[3]
        return {
            "R1": (data1 >> 0) & 1,
            "L1": (data1 >> 1) & 1,
            "Start": (data1 >> 2) & 1,
            "Select": (data1 >> 3) & 1,
            "R2": (data1 >> 4) & 1,
            "L2": (data1 >> 5) & 1,
            "F1": (data1 >> 6) & 1,
            "F3": (data1 >> 7) & 1,
            "A": (data2 >> 0) & 1,
            "B": (data2 >> 1) & 1,
            "X": (data2 >> 2) & 1,
            "Y": (data2 >> 3) & 1,
            "Up": (data2 >> 4) & 1,
            "Right": (data2 >> 5) & 1,
            "Down": (data2 >> 6) & 1,
            "Left": (data2 >> 7) & 1,
        }

    @staticmethod
    def _format_buttons(buttons):
        if not buttons:
            return "none"
        pressed = [name for name, value in buttons.items() if value]
        return ",".join(pressed) if pressed else "none"


class StudentDeploy:
    def __init__(self, cfg, mode, interface, load_policy=True):
        self.cfg = cfg
        self.mode = mode
        self.interface = interface
        self.policy = StudentPolicy(cfg) if load_policy else None
        self.crc = CRC()

        self.low_state_buffer = LatestBuffer()
        self.depth_buffer = LatestBuffer()
        self.clock_buffer = LatestBuffer()
        self.low_cmd_pub = None
        self.reset_pub = None

        self.default_dof_pos = np.array(
            [cfg.default_joint_angles[name] for name in cfg.joint_order], dtype=np.float32
        )
        self.sdk_to_urdf_offset = np.array(
            [cfg.sdk_to_urdf_offsets[name] for name in cfg.joint_order], dtype=np.float32
        )
        self.policy_to_sdk = np.array(
            [cfg.sdk_motor_order.index(name) for name in cfg.joint_order], dtype=np.int32
        )
        self.sdk_to_policy = np.array(
            [cfg.joint_order.index(name) for name in cfg.sdk_motor_order], dtype=np.int32
        )
        self.gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        self.last_action = np.zeros(cfg.act_dim, dtype=np.float32)
        self.proprio_history = deque(maxlen=cfg.proprio_history_len)
        self.latest_depth = np.full(cfg.depth_height * cfg.depth_width, cfg.depth_max, dtype=np.float32)
        self.depth_lock = threading.Lock()
        self.realsense_reader = None
        self.last_status_time = 0.0
        self.last_inference_print_time = 0.0
        self.last_obs_debug_print_time = 0.0
        self.depth_process = None
        self.depth_frame_queue = None
        goal_source = "keyboard" if self.mode == "mujoco" else "joystick"
        self.goal_source = GoalCommandSource(cfg, goal_source)

    def setup_lowstate_channel_only(self):
        if self.mode == "mujoco":
            ChannelFactoryInitialize(self.cfg.mujoco_domain_id, self.cfg.mujoco_interface)
        else:
            ChannelFactoryInitialize(self.cfg.real_domain_id, self.interface)
        low_state_sub = ChannelSubscriber(self.cfg.lowstate_topic, LowState_)
        low_state_sub.Init(self.low_state_buffer.handler, 10)
        self.low_state_sub = low_state_sub

    def setup_channels(self):
        if self.mode == "mujoco":
            ChannelFactoryInitialize(self.cfg.mujoco_domain_id, self.cfg.mujoco_interface)
        else:
            ChannelFactoryInitialize(self.cfg.real_domain_id, self.interface)
            self.release_motion_mode()

        low_state_sub = ChannelSubscriber(self.cfg.lowstate_topic, LowState_)
        low_state_sub.Init(self.low_state_buffer.handler, 10)
        self.low_state_sub = low_state_sub

        if self.mode == "mujoco":
            depth_sub = ChannelSubscriber(self.cfg.depth_topic, HeightMap_)
            depth_sub.Init(self.depth_buffer.handler, 2)
            self.depth_sub = depth_sub
            clock_sub = ChannelSubscriber(self.cfg.clock_topic, String_)
            clock_sub.Init(self.clock_buffer.handler, 5)
            self.clock_sub = clock_sub

        self.low_cmd_pub = ChannelPublisher(self.cfg.lowcmd_topic, LowCmd_)
        self.low_cmd_pub.Init()
        if self.mode == "mujoco":
            self.reset_pub = ChannelPublisher(self.cfg.reset_topic, String_)
            self.reset_pub.Init()

    def wait_for_inputs(self):
        if not self.low_state_buffer.event.wait(self.cfg.startup_timeout_s):
            if self.mode == "mujoco":
                raise TimeoutError("No LowState received. Start simulate_python/unitree_mujoco.py first.")
            raise TimeoutError("No LowState received from robot.")
        if self.mode == "mujoco" and not self.depth_buffer.event.wait(self.cfg.startup_timeout_s):
            raise TimeoutError("No depth image received on rt/depthimage.")
        if self.mode == "mujoco" and not self.clock_buffer.event.wait(self.cfg.startup_timeout_s):
            raise TimeoutError("No MuJoCo clock received on rt/mujoco_clock.")

    def release_motion_mode(self):
        if not self.cfg.release_motion_on_start:
            return
        if MotionSwitcherClient is None or SportClient is None:
            raise RuntimeError(
                "unitree_sdk2py motion switcher clients are unavailable. "
                "Update unitree_sdk2py or set release_motion_on_start=False after manually turning off sport_mode."
            )

        sport_client = SportClient()
        sport_client.SetTimeout(float(self.cfg.release_motion_timeout_s))
        sport_client.Init()

        motion_switcher_client = MotionSwitcherClient()
        motion_switcher_client.SetTimeout(float(self.cfg.release_motion_timeout_s))
        motion_switcher_client.Init()

        for attempt in range(1, int(self.cfg.release_motion_max_attempts) + 1):
            status, result = motion_switcher_client.CheckMode()
            mode_name = self._motion_mode_name(result)
            if not mode_name:
                print("[student] no active sport motion mode", flush=True)
                return

            print(
                f"[student] releasing active sport motion mode '{mode_name}' "
                f"({attempt}/{self.cfg.release_motion_max_attempts})",
                flush=True,
            )
            sport_client.StandDown()
            motion_switcher_client.ReleaseMode()
            time.sleep(float(self.cfg.release_motion_retry_s))

        status, result = motion_switcher_client.CheckMode()
        mode_name = self._motion_mode_name(result)
        if mode_name:
            raise RuntimeError(
                f"Failed to release active sport motion mode '{mode_name}'. "
                "Use the Unitree app or robot shell to turn off sport_mode before deploying."
            )

    @staticmethod
    def _motion_mode_name(result):
        if isinstance(result, dict):
            return str(result.get("name", "") or "")
        return ""

    def _auto_stop_due(self, control_start_time, auto_stop_triggered):
        if auto_stop_triggered:
            return False
        if self.cfg.auto_stop_after_s is None:
            return False
        auto_stop_after_s = float(self.cfg.auto_stop_after_s)
        if auto_stop_after_s <= 0.0:
            return False
        return time.perf_counter() - control_start_time >= auto_stop_after_s

    def _fall_stop_due(self, proprio, fall_stop_triggered):
        if fall_stop_triggered:
            return False
        if not self.cfg.fall_stop_enabled:
            return False
        projected_gravity_z = float(proprio[5])
        return projected_gravity_z > float(self.cfg.fall_stop_projected_gravity_z_min)

    def seed_history(self):
        state = self.low_state_buffer.get()
        proprio = self._get_current_proprio(state)
        self.proprio_history.clear()
        for _ in range(self.cfg.proprio_history_len):
            self.proprio_history.append(proprio.copy())
        self.policy.reset()

    def run_joint_debug(self):
        self.setup_lowstate_channel_only()
        if not self.low_state_buffer.event.wait(self.cfg.startup_timeout_s):
            raise TimeoutError("No LowState received from robot.")
        print(
            "[joint-debug] reading LowState only; no LowCmd is published. "
            "Move the robot by hand and compare q-policy/default/obs.",
            flush=True,
        )
        while True:
            low_state = self.low_state_buffer.get()
            if low_state is None:
                time.sleep(0.05)
                continue
            q_policy, dq_policy = self._get_dof_state(low_state)
            q_sdk = np.zeros(self.cfg.act_dim, dtype=np.float32)
            dq_sdk = np.zeros(self.cfg.act_dim, dtype=np.float32)
            for i in range(self.cfg.act_dim):
                q_sdk[i] = float(low_state.motor_state[i].q)
                dq_sdk[i] = float(low_state.motor_state[i].dq)
            obs_q = q_policy - self.default_dof_pos
            zero_action_target_policy = self.default_dof_pos.copy()
            zero_action_target_sdk = self._policy_q_to_sdk_q(zero_action_target_policy)
            print("\n[joint-debug] sdk raw order:", flush=True)
            print(self._format_joint_values(self.cfg.sdk_motor_order, q_sdk), flush=True)
            print("[joint-debug] policy/obs order q:", flush=True)
            print(self._format_joint_values(self.cfg.joint_order, q_policy), flush=True)
            print("[joint-debug] policy/obs order q-default:", flush=True)
            print(self._format_joint_values(self.cfg.joint_order, obs_q), flush=True)
            print("[joint-debug] policy/obs order dq:", flush=True)
            print(self._format_joint_values(self.cfg.joint_order, dq_policy), flush=True)
            print("[joint-debug] zero action target q in sdk order:", flush=True)
            print(self._format_joint_values(self.cfg.sdk_motor_order, zero_action_target_sdk), flush=True)
            time.sleep(0.5)

    @staticmethod
    def _format_joint_values(names, values):
        return "  ".join(f"{name}={float(value):+.3f}" for name, value in zip(names, values))

    def run(self):
        self.setup_channels()
        try:
            self.start_realsense_depth()
            self.wait_for_inputs()
            self.seed_history()
            self.start_depth_viewer()

            next_tick = time.perf_counter()
            control_start_time = next_tick
            next_mujoco_time = None
            stop_was_active = False
            auto_stop_triggered = False
            fall_stop_triggered = False
            while True:
                if self.mode == "mujoco":
                    sim_time = self._wait_for_mujoco_policy_tick(next_mujoco_time)
                    next_mujoco_time = sim_time + self.cfg.control_dt

                low_state = self.low_state_buffer.get()
                if low_state is None:
                    time.sleep(self.cfg.control_dt)
                    continue

                depth_msg = self.depth_buffer.get()
                if depth_msg is not None:
                    self._set_latest_depth(self._decode_depth(depth_msg))
                    self.update_depth_viewer()
                elif self.mode == "real":
                    self.update_depth_viewer()

                proprio = self._get_current_proprio(low_state)
                self.proprio_history.append(proprio.copy())
                self.goal_source.update_from_low_state(low_state)
                if self._fall_stop_due(proprio, fall_stop_triggered):
                    self.goal_source.stop_requested = True
                    fall_stop_triggered = True
                    projected_gravity = proprio[3:6]
                    print(
                        "[student] fall stop: projected_gravity="
                        f"{self._format_array(projected_gravity)}; entering stop mode",
                        flush=True,
                    )
                if self._auto_stop_due(control_start_time, auto_stop_triggered):
                    self.goal_source.stop_requested = True
                    auto_stop_triggered = True
                    print(
                        f"[student] auto stop after {self.cfg.auto_stop_after_s:.1f}s: entering stop mode",
                        flush=True,
                    )
                if self.mode == "mujoco" and self.goal_source.reset_requested:
                    self._publish_reset_command()
                    self._reset_policy_state(low_state)
                    next_mujoco_time = None
                if self.goal_source.stop_requested:
                    stop_was_active = True
                    self.last_action[:] = 0.0
                    self._publish_zero_cmd()
                    self._maybe_print_stop_status()
                    if self.mode != "mujoco":
                        next_tick += self.cfg.control_dt
                        sleep_s = next_tick - time.perf_counter()
                        if sleep_s > 0.0:
                            time.sleep(sleep_s)
                        else:
                            next_tick = time.perf_counter()
                    continue
                if stop_was_active:
                    self._reset_policy_state(low_state)
                    stop_was_active = False

                obs = self._build_observation(low_state)
                self._maybe_print_obs_debug(low_state, obs)
                inference_start = time.perf_counter()
                policy_action = self.policy.act(obs)
                inference_ms = (time.perf_counter() - inference_start) * 1000.0
                self._maybe_print_inference_time(inference_ms)
                action = np.clip(
                    policy_action.astype(np.float32),
                    -self.cfg.clip_actions,
                    self.cfg.clip_actions,
                )
                self.last_action[:] = action

                target_q = self._compute_target_q(action)
                self._publish_low_cmd(target_q)
                self._maybe_print_status(action, target_q)

                if self.mode != "mujoco":
                    next_tick += self.cfg.control_dt
                    sleep_s = next_tick - time.perf_counter()
                    if sleep_s > 0.0:
                        time.sleep(sleep_s)
                    else:
                        next_tick = time.perf_counter()
        finally:
            self.stop_realsense_depth()
            self.stop_depth_viewer()
            self.goal_source.close()

    def _publish_reset_command(self):
        if self.reset_pub is None:
            return
        self.reset_pub.Write(String_("reset"))
        print("[student] requested mujoco reset", flush=True)

    def _wait_for_mujoco_policy_tick(self, next_mujoco_time):
        while True:
            msg = self.clock_buffer.get()
            if msg is None:
                self.clock_buffer.event.wait(timeout=0.01)
                continue
            try:
                sim_time = float(msg.data)
            except (TypeError, ValueError):
                time.sleep(0.001)
                continue
            if next_mujoco_time is None or sim_time + 1e-9 >= next_mujoco_time:
                return sim_time
            time.sleep(0.001)

    def _reset_policy_state(self, low_state):
        self.last_action[:] = 0.0
        proprio = self._get_current_proprio(low_state)
        self.proprio_history.clear()
        for _ in range(self.cfg.proprio_history_len):
            self.proprio_history.append(proprio.copy())
        self.policy.reset()

    def start_depth_viewer(self):
        if not self.cfg.visualize_depth:
            return
        mp_context = mp.get_context("spawn")
        self.depth_frame_queue = mp_context.Queue(maxsize=1)
        self.depth_process = mp_context.Process(
            target=show_depth_camera,
            args=(
                self.depth_frame_queue,
                float(self.cfg.depth_min),
                float(self.cfg.depth_max),
                int(self.cfg.depth_height),
                int(self.cfg.depth_width),
            ),
            daemon=True,
        )
        self.depth_process.start()

    def update_depth_viewer(self):
        if self.depth_frame_queue is None or self.depth_process is None:
            return
        if not self.depth_process.is_alive():
            return
        frame = self._get_latest_depth().reshape(self.cfg.depth_height, self.cfg.depth_width)
        try:
            self.depth_frame_queue.put_nowait(frame)
        except queue.Full:
            pass

    def stop_depth_viewer(self):
        if self.depth_frame_queue is not None:
            try:
                self.depth_frame_queue.put_nowait(None)
            except queue.Full:
                pass
        if self.depth_process is not None and self.depth_process.is_alive():
            self.depth_process.join(timeout=0.5)

    def start_realsense_depth(self):
        if self.mode != "real":
            return
        self.realsense_reader = RealSenseDepthReader(self.cfg, self._set_latest_depth)
        self.realsense_reader.start()

    def stop_realsense_depth(self):
        if self.realsense_reader is not None:
            self.realsense_reader.stop()
            self.realsense_reader = None

    def _set_latest_depth(self, depth):
        with self.depth_lock:
            self.latest_depth[:] = depth

    def _get_latest_depth(self):
        with self.depth_lock:
            return self.latest_depth.copy()

    def _build_observation(self, low_state):
        curr_proprio_clean = self.proprio_history[-1]
        curr_proprio_noisy = curr_proprio_clean.copy()
        proprio_history = np.concatenate(list(self.proprio_history), axis=0)

        obs = np.concatenate(
            [
                self.goal_source.goal.astype(np.float32),
                np.array([self.goal_source.reached_goal], dtype=np.float32),
                curr_proprio_clean,
                curr_proprio_noisy,
                proprio_history,
                np.zeros(3, dtype=np.float32),
                np.zeros(4, dtype=np.float32),
                np.zeros(1, dtype=np.float32),
                np.zeros(1, dtype=np.float32),
                np.ones(1, dtype=np.float32),
                np.ones(1, dtype=np.float32),
                np.zeros(3, dtype=np.float32),
                np.zeros(3, dtype=np.float32),
                np.zeros(4, dtype=np.float32),
                np.zeros(4, dtype=np.float32),
                np.zeros(self.cfg.height_dim, dtype=np.float32),
                np.zeros(5, dtype=np.float32),
                self._get_latest_depth().astype(np.float32),
            ]
        )
        if obs.shape[0] != self.cfg.obs_dim:
            raise RuntimeError(f"Unexpected observation size {obs.shape[0]}, expected {self.cfg.obs_dim}")
        return np.clip(
            obs,
            -self.cfg.clip_observations,
            self.cfg.clip_observations,
        ).astype(np.float32)

    def _decode_depth(self, msg):
        expected = self.cfg.depth_width * self.cfg.depth_height
        if int(msg.width) != self.cfg.depth_width or int(msg.height) != self.cfg.depth_height:
            return self._get_latest_depth()
        if len(msg.data) != expected:
            return self._get_latest_depth()
        depth = np.array(msg.data, dtype=np.float32)
        depth = np.nan_to_num(
            depth,
            nan=self.cfg.depth_max,
            posinf=self.cfg.depth_max,
            neginf=self.cfg.depth_max,
        )
        return np.clip(depth, self.cfg.depth_min, self.cfg.depth_max)

    def _get_current_proprio(self, low_state):
        quat = self._get_base_quat(low_state)
        ang_vel_body = np.array(low_state.imu_state.gyroscope, dtype=np.float64)
        projected_gravity = quat_rotate_inverse(quat, self.gravity_world)
        dof_pos, dof_vel = self._get_dof_state(low_state)
        return np.concatenate(
            [
                (ang_vel_body * 0.25).astype(np.float32),
                projected_gravity.astype(np.float32),
                (dof_pos - self.default_dof_pos).astype(np.float32),
                (dof_vel * 0.05).astype(np.float32),
                self.last_action.astype(np.float32),
            ]
        )

    def _get_base_quat(self, low_state):
        quat = np.array(low_state.imu_state.quaternion, dtype=np.float64)
        norm = np.linalg.norm(quat)
        if norm < 1e-6:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return quat / norm

    def _get_dof_state(self, low_state):
        q_sdk = np.zeros(self.cfg.act_dim, dtype=np.float32)
        dq_sdk = np.zeros(self.cfg.act_dim, dtype=np.float32)
        for i in range(self.cfg.act_dim):
            q_sdk[i] = float(low_state.motor_state[i].q)
            dq_sdk[i] = float(low_state.motor_state[i].dq)
        q_policy = q_sdk[self.sdk_to_policy] + self.sdk_to_urdf_offset
        dq_policy = dq_sdk[self.sdk_to_policy]
        return q_policy, dq_policy

    def _policy_q_to_sdk_q(self, q_policy):
        q_sdk_order = q_policy - self.sdk_to_urdf_offset
        return q_sdk_order[self.policy_to_sdk]

    def _compute_target_q(self, action):
        return (action * self.cfg.action_scale + self.default_dof_pos).astype(np.float32)

    def _publish_low_cmd(self, target_q):
        cmd = unitree_go_msg_dds__LowCmd_()
        cmd.head[0] = 0xFE
        cmd.head[1] = 0xEF
        cmd.level_flag = 0xFF
        cmd.gpio = 0
        for i in range(self.cfg.num_motor_idl_go):
            cmd.motor_cmd[i].mode = 0x01
            cmd.motor_cmd[i].q = 0.0
            cmd.motor_cmd[i].kp = 0.0
            cmd.motor_cmd[i].dq = 0.0
            cmd.motor_cmd[i].kd = 0.0
            cmd.motor_cmd[i].tau = 0.0
        target_q_sdk = self._policy_q_to_sdk_q(target_q)
        for i in range(self.cfg.act_dim):
            cmd.motor_cmd[i].q = float(target_q_sdk[i])
            cmd.motor_cmd[i].kp = float(self.cfg.kp)
            cmd.motor_cmd[i].kd = float(self.cfg.kd)
        cmd.crc = self.crc.Crc(cmd)
        self.low_cmd_pub.Write(cmd)

    def _publish_zero_cmd(self):
        cmd = unitree_go_msg_dds__LowCmd_()
        cmd.head[0] = 0xFE
        cmd.head[1] = 0xEF
        cmd.level_flag = 0xFF
        cmd.gpio = 0
        for i in range(self.cfg.num_motor_idl_go):
            cmd.motor_cmd[i].mode = 0x00
            cmd.motor_cmd[i].q = 0.0
            cmd.motor_cmd[i].kp = 0.0
            cmd.motor_cmd[i].dq = 0.0
            cmd.motor_cmd[i].kd = 0.0
            cmd.motor_cmd[i].tau = 0.0
        cmd.crc = self.crc.Crc(cmd)
        self.low_cmd_pub.Write(cmd)

    def _maybe_print_stop_status(self):
        now = time.perf_counter()
        if now - self.last_status_time < self.cfg.status_print_interval_s:
            return
        self.last_status_time = now
        print("[student] STOP mode active: publishing zero LowCmd", flush=True)

    def _maybe_print_inference_time(self, inference_ms):
        now = time.perf_counter()
        if now - self.last_inference_print_time < self.cfg.inference_print_interval_s:
            return
        self.last_inference_print_time = now
        print(f"[policy] forward_time={inference_ms:.2f}ms", flush=True)

    def _maybe_print_obs_debug(self, low_state, obs):
        if not self.cfg.obs_debug:
            return
        now = time.perf_counter()
        if now - self.last_obs_debug_print_time < self.cfg.obs_debug_print_interval_s:
            return
        self.last_obs_debug_print_time = now
        q_urdf, dq_urdf = self._get_dof_state(low_state)
        q_minus_default = q_urdf - self.default_dof_pos
        proprio = self._get_current_proprio(low_state)
        ang_vel_scaled = proprio[0:3]
        projected_gravity = proprio[3:6]
        depth = self._get_latest_depth()
        print("\n[obs-debug] summary", flush=True)
        print(
            f"goal={self.goal_source.goal.tolist()} reached={self.goal_source.reached_goal:.0f} "
            f"obs_shape={obs.shape[0]} obs_minmax=({obs.min():+.3f},{obs.max():+.3f})",
            flush=True,
        )
        print(
            f"ang_vel_scaled={self._format_array(ang_vel_scaled)} "
            f"projected_gravity={self._format_array(projected_gravity)}",
            flush=True,
        )
        print("[obs-debug] q_urdf:", flush=True)
        print(self._format_joint_values(self.cfg.joint_order, q_urdf), flush=True)
        print("[obs-debug] q_minus_default:", flush=True)
        print(self._format_joint_values(self.cfg.joint_order, q_minus_default), flush=True)
        print("[obs-debug] dq_urdf:", flush=True)
        print(self._format_joint_values(self.cfg.joint_order, dq_urdf), flush=True)
        print("[obs-debug] last_action:", flush=True)
        print(self._format_joint_values(self.cfg.joint_order, self.last_action), flush=True)
        print(
            f"[obs-debug] depth min={depth.min():.3f} max={depth.max():.3f} mean={depth.mean():.3f}",
            flush=True,
        )

    @staticmethod
    def _format_array(values):
        return "[" + ", ".join(f"{float(value):+.3f}" for value in values) + "]"

    def _maybe_print_status(self, action, target_q):
        now = time.perf_counter()
        if now - self.last_status_time < self.cfg.status_print_interval_s:
            return
        self.last_status_time = now
        target_delta = target_q - self.default_dof_pos
        print(
            "[student] "
            f"mode={self.mode} "
            f"goal=({self.goal_source.goal[0]:.1f},{self.goal_source.goal[1]:.1f}) "
            f"reached={self.goal_source.reached_goal:.0f} "
            f"depth=({self._get_latest_depth().min():.2f},{self._get_latest_depth().max():.2f}) "
            f"action=({action.min():+.3f},{action.max():+.3f}) "
            f"dq_cmd=({target_delta.min():+.3f},{target_delta.max():+.3f}) "
            f"q0={target_q[0]:+.3f}",
            flush=True,
        )

def parse_args():
    args = list(sys.argv[1:])
    if "--camera-debug" in args:
        args.remove("--camera-debug")
        visualize_depth = True
    else:
        visualize_depth = False

    if "--joint-debug" in args:
        args.remove("--joint-debug")
        joint_debug = True
    else:
        joint_debug = False

    if "--obs-debug" in args:
        args.remove("--obs-debug")
        obs_debug = True
    else:
        obs_debug = False

    unknown_flags = [arg for arg in args if arg.startswith("--")]
    if unknown_flags:
        raise SystemExit(f"Unsupported option(s): {' '.join(unknown_flags)}")

    if len(args) > 1:
        raise SystemExit(
            "Usage: python deploy/deploy_student.py [mujoco|<network_interface>] "
            "[--camera-debug] [--joint-debug] [--obs-debug]"
        )
    if not args or args[0] == "mujoco":
        return "mujoco", "lo", visualize_depth, joint_debug, obs_debug
    return "real", args[0], visualize_depth, joint_debug, obs_debug


def main():
    cfg = resolve_config()
    mode, interface, visualize_depth, joint_debug, obs_debug = parse_args()
    cfg.visualize_depth = visualize_depth
    cfg.obs_debug = obs_debug
    deploy = StudentDeploy(cfg, mode, interface, load_policy=not joint_debug)
    if joint_debug:
        deploy.run_joint_debug()
    else:
        deploy.run()


if __name__ == "__main__":
    main()
