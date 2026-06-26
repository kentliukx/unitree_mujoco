import math
import time
import mujoco
import mujoco.viewer
import numpy as np
from threading import Thread
import threading

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import HeightMap_
from unitree_sdk2py_bridge import UnitreeSdk2Bridge, ElasticBand

import config


locker = threading.Lock()
reset_requested = threading.Event()

mj_model = mujoco.MjModel.from_xml_path(config.ROBOT_SCENE)
mj_data = mujoco.MjData(mj_model)

DEFAULT_JOINT_QPOS = (
    0.1, 0.7, -1.7,
    -0.1, 0.7, -1.7,
    0.1, 0.7, -1.7,
    -0.1, 0.7, -1.7,
)


def QuatConjugate(quat):
    return np.array([quat[0], -quat[1], -quat[2], -quat[3]], dtype=np.float64)


def QuatMultiply(q1, q2):
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


def QuatRotate(quat, vec):
    qvec = np.array([0.0, vec[0], vec[1], vec[2]], dtype=np.float64)
    return QuatMultiply(QuatMultiply(quat, qvec), QuatConjugate(quat))[1:]


def QuatFromEulerXYZ(roll, pitch, yaw):
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


def ResetRobotPose():
    if mj_model.nkey > 0:
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
    else:
        mujoco.mj_resetData(mj_model, mj_data)
    mj_data.qpos[0:3] = (-2.0, 0.0, 0.35)
    mj_data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    mj_data.qpos[7:19] = DEFAULT_JOINT_QPOS
    mj_data.qvel[:] = 0.0
    if mj_model.nu > 0:
        mj_data.ctrl[:] = 0.0
    mujoco.mj_forward(mj_model, mj_data)


ResetRobotPose()


def ResetCommandHandler(msg: String_):
    if msg.data == "reset":
        reset_requested.set()


class DepthCameraPublisher:
    def __init__(self, mj_model, mj_data):
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.width = int(config.DEPTH_WIDTH)
        self.height = int(config.DEPTH_HEIGHT)
        self.camera_offset = np.array(config.DEPTH_CAMERA_POS, dtype=np.float64)
        self.camera_rot = QuatFromEulerXYZ(
            0.0, math.radians(float(config.DEPTH_CAMERA_PITCH_DEG)), 0.0
        )
        self.geomgroup = np.ones(6, dtype=np.uint8)
        self.ray_geomid = np.zeros(1, dtype=np.int32)
        self.publisher = ChannelPublisher(config.DEPTH_TOPIC, HeightMap_)
        self.publisher.Init()

    def Publish(self):
        depth = self.RenderDepth()
        msg = HeightMap_(
            time.time(),
            "mujoco_depth_camera",
            0.0,
            self.width,
            self.height,
            [0.0, 0.0],
            depth.reshape(-1).astype(np.float32).tolist(),
        )
        self.publisher.Write(msg)

    def RenderDepth(self):
        quat = self.mj_data.qpos[3:7].copy()
        base_pos = self.mj_data.qpos[0:3].copy()
        cam_quat = QuatMultiply(quat, self.camera_rot)
        cam_pos = base_pos + QuatRotate(quat, self.camera_offset)
        hfov = math.radians(float(config.DEPTH_HFOV_DEG))
        vfov = 2.0 * math.atan(math.tan(hfov * 0.5) * (self.height / self.width))
        depth = np.full((self.height, self.width), np.inf, dtype=np.float32)

        for v in range(self.height):
            pitch = (0.5 - (v + 0.5) / self.height) * vfov
            for u in range(self.width):
                yaw = ((u + 0.5) / self.width - 0.5) * hfov
                ray_local = np.array([1.0, math.tan(yaw), math.tan(pitch)], dtype=np.float64)
                ray_world = QuatRotate(cam_quat, ray_local / np.linalg.norm(ray_local))
                dist = mujoco.mj_ray(
                    self.mj_model,
                    self.mj_data,
                    cam_pos,
                    ray_world,
                    self.geomgroup,
                    True,
                    -1,
                    self.ray_geomid,
                )
                if isinstance(dist, tuple):
                    dist = dist[0]
                if dist >= 0.0:
                    depth[v, u] = float(dist)
        return depth


if config.ENABLE_ELASTIC_BAND:
    elastic_band = ElasticBand()
    if config.ROBOT == "h1" or config.ROBOT == "g1":
        band_attached_link = mj_model.body("torso_link").id
    else:
        band_attached_link = mj_model.body("base_link").id
    viewer = mujoco.viewer.launch_passive(
        mj_model, mj_data, key_callback=elastic_band.MujuocoKeyCallback
    )
else:
    viewer = mujoco.viewer.launch_passive(mj_model, mj_data)

mj_model.opt.timestep = config.SIMULATE_DT
num_motor_ = mj_model.nu
dim_motor_sensor_ = 3 * num_motor_

time.sleep(0.2)


def SimulationThread():
    global mj_data, mj_model

    ChannelFactoryInitialize(config.DOMAIN_ID, config.INTERFACE)
    unitree = UnitreeSdk2Bridge(mj_model, mj_data)
    reset_subscriber = ChannelSubscriber(config.RESET_TOPIC, String_)
    reset_subscriber.Init(ResetCommandHandler, 1)
    depth_camera = DepthCameraPublisher(mj_model, mj_data)
    next_depth_publish_time = 0.0

    if config.USE_JOYSTICK:
        unitree.SetupJoystick(device_id=0, js_type=config.JOYSTICK_TYPE)
    if config.PRINT_SCENE_INFORMATION:
        unitree.PrintSceneInformation()

    while viewer.is_running():
        step_start = time.perf_counter()

        locker.acquire()

        if reset_requested.is_set():
            ResetRobotPose()
            reset_requested.clear()
            next_depth_publish_time = mj_data.time

        if config.ENABLE_ELASTIC_BAND:
            if elastic_band.enable:
                mj_data.xfrc_applied[band_attached_link, :3] = elastic_band.Advance(
                    mj_data.qpos[:3], mj_data.qvel[:3]
                )
        mujoco.mj_step(mj_model, mj_data)
        if mj_data.time >= next_depth_publish_time:
            depth_camera.Publish()
            next_depth_publish_time = mj_data.time + config.DEPTH_UPDATE_DT

        locker.release()

        time_until_next_step = mj_model.opt.timestep - (
            time.perf_counter() - step_start
        )
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)


def PhysicsViewerThread():
    while viewer.is_running():
        locker.acquire()
        viewer.sync()
        locker.release()
        time.sleep(config.VIEWER_DT)


if __name__ == "__main__":
    viewer_thread = Thread(target=PhysicsViewerThread)
    sim_thread = Thread(target=SimulationThread)

    viewer_thread.start()
    sim_thread.start()
