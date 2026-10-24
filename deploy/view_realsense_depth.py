import argparse
from datetime import datetime
from pathlib import Path
import time

import numpy as np

from deploy_student_config import DeployStudentConfig
from depth_processing import preprocess_depth


def colorize_depth(depth_m, min_depth, max_depth, cv2):
    """Apply one fixed metric-depth color scale to every saved image."""
    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    clipped = np.clip(depth_m, min_depth, max_depth)
    normalized = (clipped - min_depth) / (max_depth - min_depth)
    depth_u8 = np.round(255.0 * normalized).astype(np.uint8)
    depth_u8[~valid] = 0
    return cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)


def save_record_pair(record_dir, index, raw_color, cnn_color, cv2):
    raw_path = record_dir / f"frame_{index:06d}_raw.png"
    cnn_path = record_dir / f"frame_{index:06d}_cnn.png"
    if not cv2.imwrite(str(raw_path), raw_color):
        raise RuntimeError(f"Failed to save {raw_path}")
    if not cv2.imwrite(str(cnn_path), cnn_color):
        raise RuntimeError(f"Failed to save {cnn_path}")


def main():
    defaults = DeployStudentConfig()
    parser = argparse.ArgumentParser(description="Show RealSense depth image in a window.")
    parser.add_argument("--width", type=int, default=defaults.realsense_width)
    parser.add_argument("--height", type=int, default=defaults.realsense_height)
    parser.add_argument("--fps", type=int, default=defaults.realsense_fps)
    parser.add_argument("--min-depth", type=float, default=defaults.depth_min)
    parser.add_argument("--max-depth", type=float, default=defaults.depth_max)
    parser.add_argument("--serial", default=None)
    parser.add_argument(
        "--record",
        action="store_true",
        help="Save raw-resolution and CNN-input depth images at a fixed interval.",
    )
    parser.add_argument(
        "--record-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "records" / "realsense_depth",
        help="Base output directory; a timestamped session directory is created inside it.",
    )
    parser.add_argument("--record-interval", type=float, default=0.1)
    parser.add_argument("--list-profiles", action="store_true", help="Print supported depth profiles and exit.")
    args = parser.parse_args()

    if args.min_depth >= args.max_depth:
        parser.error("--min-depth must be smaller than --max-depth")
    if args.record_interval <= 0.0:
        parser.error("--record-interval must be positive")

    depth_cfg = DeployStudentConfig()
    depth_cfg.realsense_width = args.width
    depth_cfg.realsense_height = args.height
    depth_cfg.realsense_fps = args.fps
    depth_cfg.depth_min = args.min_depth
    depth_cfg.depth_max = args.max_depth

    import cv2
    import pyrealsense2 as rs

    if args.list_profiles:
        context = rs.context()
        devices = list(context.query_devices())
        if not devices:
            raise SystemExit("No RealSense device found.")
        for device in devices:
            name = device.get_info(rs.camera_info.name)
            serial = device.get_info(rs.camera_info.serial_number)
            print(f"{name} serial={serial}")
            profiles = set()
            for sensor in device.query_sensors():
                if not sensor.is_depth_sensor():
                    continue
                for profile in sensor.get_stream_profiles():
                    video_profile = profile.as_video_stream_profile()
                    if video_profile.stream_type() == rs.stream.depth and video_profile.format() == rs.format.z16:
                        profiles.add((video_profile.width(), video_profile.height(), video_profile.fps()))
            for width, height, fps in sorted(profiles, key=lambda item: (item[0] * item[1], item[2])):
                print(f"  {width}x{height}@{fps}")
        return

    pipeline = rs.pipeline()
    config = rs.config()
    if args.serial:
        config.enable_device(str(args.serial))
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

    window_name = "RealSense depth"
    mouse = {"x": None, "y": None}
    record_dir = None
    record_index = 0
    next_record_time = None
    if args.record:
        record_dir = args.record_dir.expanduser() / datetime.now().strftime("%Y%m%d_%H%M%S")
        record_dir.mkdir(parents=True, exist_ok=False)
        next_record_time = time.monotonic()
        print(f"[depth-view] recording to {record_dir}", flush=True)

    def on_mouse(event, x, y, flags, param):
        mouse["x"] = x
        mouse["y"] = y

    started = False
    try:
        profile = pipeline.start(config)
        started = True
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        print("[depth-view] press q or Esc to quit", flush=True)
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, on_mouse)

        while True:
            frames = pipeline.wait_for_frames(timeout_ms=1000)
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                continue

            depth_m = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale
            raw_color = colorize_depth(depth_m, args.min_depth, args.max_depth, cv2)
            cnn_depth = preprocess_depth(depth_m, depth_cfg)
            cnn_color = colorize_depth(cnn_depth, args.min_depth, args.max_depth, cv2)

            now = time.monotonic()
            if record_dir is not None and now >= next_record_time:
                save_record_pair(record_dir, record_index, raw_color, cnn_color, cv2)
                if record_index == 0:
                    print(
                        f"[depth-view] raw={depth_m.shape[1]}x{depth_m.shape[0]} "
                        f"cnn={cnn_depth.shape[1]}x{cnn_depth.shape[0]} "
                        f"range=[{args.min_depth:.2f}, {args.max_depth:.2f}]m",
                        flush=True,
                    )
                record_index += 1
                while next_record_time <= now:
                    next_record_time += args.record_interval

            depth_color = raw_color.copy()

            x = mouse["x"]
            y = mouse["y"]
            if x is not None and y is not None and 0 <= x < depth_m.shape[1] and 0 <= y < depth_m.shape[0]:
                depth_value = float(depth_m[y, x])
                if depth_value > 0.0:
                    text = f"x={x} y={y} depth={depth_value:.3f}m"
                else:
                    text = f"x={x} y={y} depth=invalid"
                cv2.circle(depth_color, (x, y), 4, (255, 255, 255), 1)
            else:
                text = "move mouse over image"

            cv2.rectangle(depth_color, (0, 0), (430, 30), (0, 0, 0), -1)
            cv2.putText(
                depth_color,
                text,
                (8, 21),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.setWindowTitle(window_name, text)
            cnn_display_width = round(cnn_color.shape[1] * depth_color.shape[0] / cnn_color.shape[0])
            cnn_display = cv2.resize(
                cnn_color,
                (cnn_display_width, depth_color.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            combined = np.concatenate((depth_color, cnn_display), axis=1)
            cv2.imshow(window_name, combined)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        if started:
            pipeline.stop()
        cv2.destroyAllWindows()
        if record_dir is not None:
            print(f"[depth-view] saved {record_index} image pairs to {record_dir}", flush=True)


if __name__ == "__main__":
    main()
