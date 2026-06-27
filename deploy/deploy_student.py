import sys
import multiprocessing as mp
import queue
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
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
    from unitree_sdk2py.utils.crc import CRC
except ImportError as exc:
    raise SystemExit(
        "Missing runtime dependency. Run this script in the deployment environment "
        "with `torch` and `unitree_sdk2py` installed."
    ) from exc


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
RSL_RL_ROOT = ROOT.parent / "rsl_rl"
STUDENT_DIR = ROOT / "deploy" / "student"

for path in (SCRIPT_DIR, RSL_RL_ROOT):
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
            valid = blocks > 0.0
            nearest = np.where(valid, blocks, self.cfg.depth_max).min(axis=(1, 3))
            has_valid = valid.any(axis=(1, 3))
            return np.where(has_valid, nearest, self.cfg.depth_max)

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
    def __init__(self, cfg):
        self.cfg = cfg
        self.source = cfg.goal_source
        self.goal = np.array([cfg.goal_forward_m, cfg.goal_lateral_m], dtype=np.float32)
        self.reached_goal = 0.0
        self.pygame = None
        self.reset_requested = False
        self._reset_key_was_down = False
        if self.source == "keyboard":
            import pygame

            self.pygame = pygame
            pygame.init()
            pygame.display.set_mode((360, 120))
            pygame.display.set_caption("Goal keyboard control: W/S A/D")
        elif self.source == "joystick":
            self._setup_joystick_placeholder()

    def update(self):
        if self.source == "keyboard":
            self._update_keyboard()
        elif self.source == "joystick":
            self._update_joystick_placeholder()
        elif self.source == "fixed":
            self.goal[:] = (self.cfg.goal_forward_m, self.cfg.goal_lateral_m)
            self.reached_goal = 0.0
        else:
            raise ValueError(f"Unsupported goal source: {self.source}")

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

    def _setup_joystick_placeholder(self):
        self.goal[:] = 0.0
        self.reached_goal = 1.0
        self.reset_requested = False

    def _update_joystick_placeholder(self):
        self.goal[:] = 0.0
        self.reached_goal = 1.0
        self.reset_requested = False


class StudentDeploy:
    def __init__(self, cfg, mode, interface):
        self.cfg = cfg
        self.mode = mode
        self.interface = interface
        self.policy = StudentPolicy(cfg)
        self.crc = CRC()

        self.low_state_buffer = LatestBuffer()
        self.high_state_buffer = LatestBuffer()
        self.depth_buffer = LatestBuffer()
        self.clock_buffer = LatestBuffer()
        self.low_cmd_pub = None
        self.reset_pub = None

        self.default_dof_pos = np.array(
            [cfg.default_joint_angles[name] for name in cfg.joint_order], dtype=np.float32
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
        self.depth_process = None
        self.depth_frame_queue = None
        self.goal_source = GoalCommandSource(cfg)

    def setup_channels(self):
        if self.mode == "mujoco":
            ChannelFactoryInitialize(self.cfg.mujoco_domain_id, self.cfg.mujoco_interface)
        else:
            ChannelFactoryInitialize(self.cfg.real_domain_id, self.interface)

        low_state_sub = ChannelSubscriber(self.cfg.lowstate_topic, LowState_)
        low_state_sub.Init(self.low_state_buffer.handler, 10)
        self.low_state_sub = low_state_sub

        high_state_sub = ChannelSubscriber(self.cfg.sportstate_topic, SportModeState_)
        high_state_sub.Init(self.high_state_buffer.handler, 10)
        self.high_state_sub = high_state_sub

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

    def seed_history(self):
        state = self.low_state_buffer.get()
        proprio = self._get_current_proprio(state)
        self.proprio_history.clear()
        for _ in range(self.cfg.proprio_history_len):
            self.proprio_history.append(proprio.copy())
        self.policy.reset()

    def run(self):
        self.setup_channels()
        self.start_realsense_depth()
        self.wait_for_inputs()
        self.seed_history()
        self.start_depth_viewer()

        next_tick = time.perf_counter()
        next_mujoco_time = None
        try:
            while True:
                if self.mode == "mujoco":
                    sim_time = self._wait_for_mujoco_policy_tick(next_mujoco_time)
                    if sim_time is None:
                        continue
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
                self.goal_source.update()
                if self.mode == "mujoco" and self.goal_source.reset_requested:
                    self._publish_reset_command()
                    self._reset_policy_state(low_state)
                    next_mujoco_time = None

                obs = self._build_observation(low_state)
                action = np.clip(
                    self.policy.act(obs).astype(np.float32),
                    -self.cfg.clip_actions,
                    self.cfg.clip_actions,
                )
                self.last_action[:] = action

                torques = self._compute_torques(action, low_state)
                self._publish_low_cmd(torques)
                self._maybe_print_status(action, torques)

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
        return q_sdk[self.sdk_to_policy], dq_sdk[self.sdk_to_policy]

    def _compute_torques(self, action, low_state):
        dof_pos, dof_vel = self._get_dof_state(low_state)
        desired_q = action * self.cfg.action_scale + self.default_dof_pos
        torques = self.cfg.kp * (desired_q - dof_pos) - self.cfg.kd * dof_vel
        same_direction = (dof_vel * torques) > 0.0
        max_torque = np.where(
            same_direction,
            self.cfg.torque_limit_same_direction,
            self.cfg.torque_limit_opposite_direction,
        )
        speed = np.abs(dof_vel)
        decay_torque = max_torque * (self.cfg.motor_velocity_x2 - speed) / max(
            self.cfg.motor_velocity_x2 - self.cfg.motor_velocity_x1, 1e-6
        )
        torque_limit = np.where(
            speed < self.cfg.motor_velocity_x1, max_torque, np.clip(decay_torque, 0.0, None)
        )
        torques = np.clip(torques, -torque_limit, torque_limit)
        friction = (
            self.cfg.motor_static_friction * np.tanh(dof_vel / self.cfg.motor_friction_activation_velocity)
            + self.cfg.motor_dynamic_friction * dof_vel
        )
        return np.clip(
            torques - friction,
            -self.cfg.torque_limit_opposite_direction,
            self.cfg.torque_limit_opposite_direction,
        ).astype(np.float32)

    def _publish_low_cmd(self, torques):
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
        torques_sdk = torques[self.policy_to_sdk]
        for i in range(self.cfg.act_dim):
            cmd.motor_cmd[i].tau = float(torques_sdk[i])
        cmd.crc = self.crc.Crc(cmd)
        self.low_cmd_pub.Write(cmd)

    def _maybe_print_status(self, action, torques):
        now = time.perf_counter()
        if now - self.last_status_time < self.cfg.status_print_interval_s:
            return
        self.last_status_time = now
        print(
            "[student] "
            f"mode={self.mode} "
            f"goal=({self.goal_source.goal[0]:.1f},{self.goal_source.goal[1]:.1f}) "
            f"reached={self.goal_source.reached_goal:.0f} "
            f"depth=({self._get_latest_depth().min():.2f},{self._get_latest_depth().max():.2f}) "
            f"action0={action[0]:.3f} torque0={torques[0]:.3f}",
            flush=True,
        )

def parse_args():
    args = list(sys.argv[1:])
    visualize_depth = False
    if "--camera-debug" in args:
        visualize_depth = True
        args.remove("--camera-debug")
    goal_source = CONFIG.goal_source
    if "--goal-source" in args:
        source_index = args.index("--goal-source")
        if source_index + 1 >= len(args):
            raise SystemExit("--goal-source requires one of: keyboard, joystick, fixed")
        goal_source = args[source_index + 1]
        del args[source_index:source_index + 2]
    if len(args) < 1 or args[0] == "mujoco":
        return "mujoco", "lo", visualize_depth, goal_source
    return "real", args[0], visualize_depth, goal_source


def main():
    cfg = resolve_config()
    mode, interface, visualize_depth, goal_source = parse_args()
    cfg.visualize_depth = visualize_depth
    cfg.goal_source = goal_source
    deploy = StudentDeploy(cfg, mode, interface)
    deploy.run()


if __name__ == "__main__":
    main()
