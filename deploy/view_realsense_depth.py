import argparse

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Show RealSense depth image in a window.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-depth", type=float, default=2.0)
    parser.add_argument("--serial", default=None)
    args = parser.parse_args()

    import cv2
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    config = rs.config()
    if args.serial:
        config.enable_device(str(args.serial))
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

    window_name = "RealSense depth"
    mouse = {"x": None, "y": None}

    def on_mouse(event, x, y, flags, param):
        mouse["x"] = x
        mouse["y"] = y

    try:
        profile = pipeline.start(config)
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
            depth_norm = np.clip(depth_m / float(args.max_depth), 0.0, 1.0)
            depth_u8 = (255.0 * (1.0 - depth_norm)).astype(np.uint8)
            depth_u8[depth_m <= 0.0] = 0
            depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)

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
            cv2.imshow(window_name, depth_color)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
