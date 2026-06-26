import math
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

try:
    import mujoco
    import mujoco.viewer
    import torch
except ImportError as exc:
    raise SystemExit(
        "Missing runtime dependency. Run this script in the deployment environment "
        "with `mujoco` and `torch` installed."
    ) from exc


ROOT = Path(__file__).resolve().parents[1]
LEGGED_GYM_ROOT = ROOT.parent / "legged_gym"
RSL_RL_ROOT = ROOT.parent / "rsl_rl"

if str(LEGGED_GYM_ROOT) not in sys.path:
    sys.path.insert(0, str(LEGGED_GYM_ROOT))
if str(RSL_RL_ROOT) not in sys.path:
    sys.path.insert(0, str(RSL_RL_ROOT))

from rsl_rl.modules.actor_critic_recurrent import ActorCriticRecurrent, TeacherPolicy
from deploy_mujoco_config import CONFIG


OBS_DIM = 3072
ACT_DIM = 12
PROPRIO_DIM = 42
PROPRIO_HISTORY_LEN = 10
HEIGHT_SHAPE = (21, 11)
HEIGHT_DIM = HEIGHT_SHAPE[0] * HEIGHT_SHAPE[1]
DEPTH_SHAPE = (36, 64)
DEPTH_DIM = DEPTH_SHAPE[0] * DEPTH_SHAPE[1]
SIM_DT = 0.005
CONTROL_DECIMATION = 4
ACTION_SCALE = 1.0
CLIP_OBSERVATIONS = 100.0
CLIP_ACTIONS = 10.0
KP = 25.0
KD = 0.5
TORQUE_LIMIT_SAME = 20.2
TORQUE_LIMIT_OPPOSITE = 23.4
MOTOR_VEL_X1 = 13.5
MOTOR_VEL_X2 = 30.0
MOTOR_STATIC_FRICTION = 0.0
MOTOR_DYNAMIC_FRICTION = 0.0
MOTOR_FRICTION_ACTIVATION_VELOCITY = 0.01
GOAL_RADIUS = 0.15

JOINT_ORDER = [
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
]

ACTUATOR_ORDER = [
    "FL_hip",
    "FL_thigh",
    "FL_calf",
    "FR_hip",
    "FR_thigh",
    "FR_calf",
    "RL_hip",
    "RL_thigh",
    "RL_calf",
    "RR_hip",
    "RR_thigh",
    "RR_calf",
]

FOOT_GEOMS = ["FL", "FR", "RL", "RR"]

DEFAULT_JOINT_ANGLES = {
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
}

HEIGHT_POINTS_X = np.array(
    [
        -0.8,
        -0.72,
        -0.64,
        -0.56,
        -0.48,
        -0.4,
        -0.32,
        -0.24,
        -0.16,
        -0.08,
        0.0,
        0.08,
        0.16,
        0.24,
        0.32,
        0.4,
        0.48,
        0.56,
        0.64,
        0.72,
        0.8,
    ],
    dtype=np.float64,
)
HEIGHT_POINTS_Y = np.array(
    [-0.4, -0.32, -0.24, -0.16, -0.08, 0.0, 0.08, 0.16, 0.24, 0.32, 0.4],
    dtype=np.float64,
)


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


def quat_from_euler_xyz(roll, pitch, yaw):
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float64,
    )


def wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


class PolicyWrapper:
    def __init__(self, cfg):
        self.device = torch.device(cfg.device)
        self.policy_kind = cfg.policy
        self.module = self._load_policy(cfg).to(self.device)
        self.module.eval()

    def _load_policy(self, cfg):
        if cfg.policy == "teacher":
            module = TeacherPolicy(
                num_actions=ACT_DIM,
                actor_hidden_dims=[256, 128, 64],
                activation="elu",
            )
            module.load(str(cfg.checkpoint))
            return module
        if cfg.policy == "student":
            module = ActorCriticRecurrent(
                num_actor_obs=OBS_DIM,
                num_critic_obs=OBS_DIM,
                num_actions=ACT_DIM,
                actor_hidden_dims=[256, 128, 64],
                critic_hidden_dims=[512, 256, 128],
                activation="elu",
                history_length=PROPRIO_HISTORY_LEN,
                history_latent_dim=32,
                depth_latent_dim=32,
                mixer_latent_dim=32,
                rnn_type="gru",
                rnn_hidden_size=32,
                rnn_num_layers=1,
            )
            checkpoint = torch.load(cfg.checkpoint, map_location="cpu")
            module.load_state_dict(checkpoint["model_state_dict"], strict=False)
            return module
        raise ValueError(f"Unsupported policy type: {cfg.policy}")

    def reset(self):
        if hasattr(self.module, "reset"):
            self.module.reset()
        if hasattr(self.module, "reset_memory"):
            self.module.reset_memory()

    @torch.inference_mode()
    def act(self, obs_np):
        obs = torch.from_numpy(obs_np).to(device_for_tensor(self.device))
        obs = obs.unsqueeze(0)
        if self.policy_kind == "teacher":
            action = self.module.act_inference(obs)
        else:
            action = self.module.act_inference(obs)
        return action.squeeze(0).detach().cpu().numpy()


def device_for_tensor(device):
    return device if device.type != "cuda" else torch.device(device)


class Go2LadderDeploy:
    def __init__(self, cfg):
        self.cfg = cfg
        self.policy = PolicyWrapper(cfg)
        self.model = mujoco.MjModel.from_xml_path(str(cfg.scene))
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = SIM_DT

        self.joint_qpos_ids = np.array(
            [
                self.model.jnt_qposadr[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                ]
                for name in JOINT_ORDER
            ],
            dtype=np.int32,
        )
        self.joint_qvel_ids = np.array(
            [
                self.model.jnt_dofadr[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                ]
                for name in JOINT_ORDER
            ],
            dtype=np.int32,
        )
        self.actuator_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                for name in ACTUATOR_ORDER
            ],
            dtype=np.int32,
        )
        self.use_actuators = self.model.nu > 0 and np.all(self.actuator_ids >= 0)
        if self.use_actuators:
            ctrlrange = self.model.actuator_ctrlrange[self.actuator_ids]
            self.torque_limits = np.maximum(np.abs(ctrlrange[:, 0]), np.abs(ctrlrange[:, 1])).astype(np.float32)
        else:
            self.torque_limits = np.full(ACT_DIM, TORQUE_LIMIT_OPPOSITE, dtype=np.float32)
        self.imu_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "imu")
        self.foot_geom_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
                for name in FOOT_GEOMS
            ],
            dtype=np.int32,
        )
        self.default_dof_pos = np.array(
            [DEFAULT_JOINT_ANGLES[name] for name in JOINT_ORDER], dtype=np.float32
        )
        self.proprio_history = deque(maxlen=PROPRIO_HISTORY_LEN)
        self.last_action = np.zeros(ACT_DIM, dtype=np.float32)
        self.feet_air_time = np.zeros(4, dtype=np.float32)
        self.feet_ground_time = np.zeros(4, dtype=np.float32)
        self.prev_contacts = np.zeros(4, dtype=bool)
        self.height_scan = np.zeros(HEIGHT_DIM, dtype=np.float32)
        self.depth_image = np.full(DEPTH_DIM, cfg.depth_max, dtype=np.float32)
        self.goal_world = np.array(cfg.goal, dtype=np.float64)
        self.goal_world[2] = float(cfg.goal_height)
        self.gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        self.camera_offset = np.array([0.33, 0.0, 0.08], dtype=np.float64)
        self.camera_rot = quat_from_euler_xyz(0.0, math.radians(cfg.camera_pitch_deg), 0.0)
        self.reset()

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.data.qpos[0:3] = np.array(self.cfg.spawn_pos, dtype=np.float64)
        self.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.data.qpos[self.joint_qpos_ids] = self.default_dof_pos
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        self.data.qfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.proprio_history.clear()
        seed_prop = self._get_current_proprio()
        for _ in range(PROPRIO_HISTORY_LEN):
            self.proprio_history.append(seed_prop.copy())
        self.last_action[:] = 0.0
        self.feet_air_time[:] = 0.0
        self.feet_ground_time[:] = 0.0
        self.prev_contacts[:] = False
        self.policy.reset()

    def run(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            step_count = 0
            action = np.zeros(ACT_DIM, dtype=np.float32)
            while viewer.is_running():
                if step_count % CONTROL_DECIMATION == 0:
                    self._update_observation_terms()
                    obs = self._build_observation()
                    action = self.policy.act(obs).astype(np.float32)
                    self.last_action[:] = action
                torques = self._compute_torques(action)
                if self.use_actuators:
                    self.data.ctrl[self.actuator_ids] = torques
                else:
                    self.data.qfrc_applied[:] = 0.0
                    self.data.qfrc_applied[self.joint_qvel_ids] = torques
                mujoco.mj_step(self.model, self.data)
                viewer.sync()
                if self._should_reset():
                    self.reset()
                time.sleep(max(0.0, SIM_DT - 0.0005))
                step_count += 1

    def _should_reset(self):
        base_z = float(self.data.qpos[2])
        quat = self.data.qpos[3:7].copy()
        projected_gravity = quat_rotate_inverse(quat, self.gravity_world)
        upside_down = projected_gravity[2] > -0.2
        return base_z < 0.12 or upside_down

    def _get_current_proprio(self):
        quat = self.data.qpos[3:7].copy()
        ang_vel_world = self.data.qvel[3:6].copy()
        ang_vel_body = quat_rotate_inverse(quat, ang_vel_world)
        projected_gravity = quat_rotate_inverse(quat, self.gravity_world)
        dof_pos = self.data.qpos[self.joint_qpos_ids].copy()
        dof_vel = self.data.qvel[self.joint_qvel_ids].copy()
        return np.concatenate(
            [
                ang_vel_body.astype(np.float32) * 0.25,
                projected_gravity.astype(np.float32),
                (dof_pos - self.default_dof_pos).astype(np.float32),
                (dof_vel * 0.05).astype(np.float32),
                self.last_action.astype(np.float32),
            ]
        )

    def _update_observation_terms(self):
        proprio = self._get_current_proprio()
        self.proprio_history.append(proprio.copy())
        contacts = self._foot_contacts()
        self._update_contact_timers(contacts)
        self.height_scan[:] = self._sample_height_scan()
        # self.depth_image[:] = self._sample_depth_image()

    def _build_observation(self):
        quat = self.data.qpos[3:7].copy()
        rel_goal_world = self.goal_world - self.data.qpos[0:3]
        rel_goal_body = quat_rotate_inverse(quat, rel_goal_world)
        goal_dist = np.linalg.norm(rel_goal_world)
        reached_goal = np.array([1.0 if goal_dist < GOAL_RADIUS else 0.0], dtype=np.float32)
        goal = np.array([rel_goal_body[0], rel_goal_body[1], reached_goal[0]], dtype=np.float32)

        curr_proprio_clean = self.proprio_history[-1]
        curr_proprio_noisy = curr_proprio_clean.copy()
        proprio_history = np.concatenate(list(self.proprio_history), axis=0)

        base_lin_vel_body = quat_rotate_inverse(quat, self.data.qvel[0:3].copy()).astype(np.float32)
        contacts = self._foot_contacts().astype(np.float32)
        privileged = np.concatenate(
            [
                base_lin_vel_body,
                contacts,
                np.array([self.cfg.friction], dtype=np.float32),
                np.array([0.0], dtype=np.float32),
                np.array([1.0], dtype=np.float32),
                np.array([1.0], dtype=np.float32),
                np.zeros(3, dtype=np.float32),
                np.zeros(3, dtype=np.float32),
                self.feet_air_time.copy(),
                self.feet_ground_time.copy(),
            ]
        )

        forward_world = quat_rotate(quat, np.array([1.0, 0.0, 0.0], dtype=np.float64))
        base_heading = math.atan2(forward_world[1], forward_world[0])
        ladder_up_yaw_rel = wrap_to_pi(-base_heading)

        ladder_obs = np.array(
            [
                self.cfg.ladder_bar_spacing,
                self.cfg.ladder_bar_spacing * math.sin(math.radians(self.cfg.ladder_angle_deg)),
                self.cfg.ladder_bar_spacing * math.cos(math.radians(self.cfg.ladder_angle_deg)),
                ladder_up_yaw_rel,
                self.cfg.ladder_bar_y_scale,
            ],
            dtype=np.float32,
        )

        obs = np.concatenate(
            [
                goal,
                curr_proprio_clean,
                curr_proprio_noisy,
                proprio_history,
                privileged,
                self.height_scan.astype(np.float32),
                ladder_obs,
                self.depth_image.astype(np.float32),
            ]
        )
        if obs.shape[0] != OBS_DIM:
            raise RuntimeError(f"Unexpected observation size {obs.shape[0]}, expected {OBS_DIM}")
        return np.clip(obs, -CLIP_OBSERVATIONS, CLIP_OBSERVATIONS).astype(np.float32)

    def _foot_contacts(self):
        contact = np.zeros(4, dtype=bool)
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            for foot_idx, geom_id in enumerate(self.foot_geom_ids):
                if con.geom1 == geom_id or con.geom2 == geom_id:
                    contact[foot_idx] = True
        return contact

    def _update_contact_timers(self, contacts):
        self.feet_air_time[~contacts] += SIM_DT
        self.feet_ground_time[contacts] += SIM_DT
        self.feet_air_time[contacts & ~self.prev_contacts] = 0.0
        self.feet_ground_time[~contacts & self.prev_contacts] = 0.0
        self.prev_contacts[:] = contacts

    def _compute_torques(self, action):
        action = np.clip(action, -CLIP_ACTIONS, CLIP_ACTIONS)
        dof_pos = self.data.qpos[self.joint_qpos_ids]
        dof_vel = self.data.qvel[self.joint_qvel_ids]
        desired_q = action * ACTION_SCALE + self.default_dof_pos
        torques = KP * (desired_q - dof_pos) - KD * dof_vel
        same_direction = (dof_vel * torques) > 0.0
        max_torque = np.where(same_direction, TORQUE_LIMIT_SAME, TORQUE_LIMIT_OPPOSITE)
        speed = np.abs(dof_vel)
        decay_torque = max_torque * (MOTOR_VEL_X2 - speed) / max(MOTOR_VEL_X2 - MOTOR_VEL_X1, 1e-6)
        torque_limit = np.where(speed < MOTOR_VEL_X1, max_torque, np.clip(decay_torque, 0.0, None))
        torque_limit = np.minimum(torque_limit, self.torque_limits)
        torques = np.clip(torques, -torque_limit, torque_limit)
        friction = (
            MOTOR_STATIC_FRICTION * np.tanh(dof_vel / MOTOR_FRICTION_ACTIVATION_VELOCITY)
            + MOTOR_DYNAMIC_FRICTION * dof_vel
        )
        return np.clip(torques - friction, -self.torque_limits, self.torque_limits).astype(np.float32)

    def _sample_height_scan(self):
        if not self.cfg.enable_height_scan:
            return np.zeros(HEIGHT_DIM, dtype=np.float32)
        base_pos = self.data.qpos[0:3].copy()
        quat = self.data.qpos[3:7].copy()
        forward_world = quat_rotate(quat, np.array([1.0, 0.0, 0.0], dtype=np.float64))
        base_heading = math.atan2(forward_world[1], forward_world[0])
        cos_yaw = math.cos(base_heading)
        sin_yaw = math.sin(base_heading)
        samples = np.full(HEIGHT_DIM, 1.0, dtype=np.float32)
        idx = 0
        for x in HEIGHT_POINTS_X:
            for y in HEIGHT_POINTS_Y:
                world_point = np.array(
                    [
                        base_pos[0] + cos_yaw * x - sin_yaw * y,
                        base_pos[1] + sin_yaw * x + cos_yaw * y,
                        base_pos[2],
                    ],
                    dtype=np.float64,
                )
                origin = world_point + np.array([0.0, 0.0, 2.0], dtype=np.float64)
                dist = self._ray_distance(origin, np.array([0.0, 0.0, -1.0], dtype=np.float64))
                if dist is None:
                    height = base_pos[2] - 0.5
                else:
                    hit_z = origin[2] - dist
                    height = base_pos[2] - 0.5 - hit_z
                samples[idx] = np.clip(height, -1.0, 1.0)
                idx += 1
        return samples

    def _sample_depth_image(self):
        if not self.cfg.enable_depth:
            return np.full(DEPTH_DIM, self.cfg.depth_max, dtype=np.float32)
        quat = self.data.qpos[3:7].copy()
        base_pos = self.data.qpos[0:3].copy()
        cam_quat = quat_multiply(quat, self.camera_rot)
        cam_pos = base_pos + quat_rotate(quat, self.camera_offset)
        h, w = DEPTH_SHAPE
        hfov = math.radians(self.cfg.depth_hfov_deg)
        vfov = 2.0 * math.atan(math.tan(hfov * 0.5) * (h / w))
        depth = np.full((h, w), self.cfg.depth_max, dtype=np.float32)
        for v in range(h):
            pitch = (0.5 - (v + 0.5) / h) * vfov
            for u in range(w):
                yaw = ((u + 0.5) / w - 0.5) * hfov
                ray_local = np.array(
                    [
                        1.0,
                        math.tan(yaw),
                        math.tan(pitch),
                    ],
                    dtype=np.float64,
                )
                ray_world = quat_rotate(cam_quat, ray_local / np.linalg.norm(ray_local))
                dist = self._ray_distance(cam_pos, ray_world)
                if dist is not None:
                    depth[v, u] = np.clip(dist, self.cfg.depth_min, self.cfg.depth_max)
        return depth.reshape(-1)

    def _ray_distance(self, origin, direction):
        try:
            geomgroup = np.ones(6, dtype=np.uint8)
            result = mujoco.mj_ray(
                self.model,
                self.data,
                origin,
                direction,
                geomgroup,
                1,
                -1,
            )
        except Exception:
            return None
        if isinstance(result, tuple):
            dist = result[0]
        else:
            dist = result
        if dist < 0:
            return None
        return float(dist)


def default_checkpoint(policy_kind):
    if policy_kind == "teacher":
        return LEGGED_GYM_ROOT / "logs/go2_ladder/Teacher/model_30000.pt"
    if policy_kind == "student":
        return LEGGED_GYM_ROOT / "logs/go2_ladder/Teacher_stage2/model_30000.pt"
    raise ValueError(f"Unsupported policy type: {policy_kind}")


def resolve_config():
    cfg = CONFIG
    if cfg.checkpoint is None:
        cfg.checkpoint = default_checkpoint(cfg.policy)
    cfg.scene = Path(cfg.scene).resolve()
    cfg.checkpoint = Path(cfg.checkpoint).resolve()
    return cfg


def main():
    cfg = resolve_config()
    deploy = Go2LadderDeploy(cfg)
    deploy.run()


if __name__ == "__main__":
    main()
