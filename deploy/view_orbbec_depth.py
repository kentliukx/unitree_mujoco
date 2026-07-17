#!/usr/bin/env python3
"""Preview metric depth from an Orbbec Femto Bolt camera.

Move the mouse over the image to inspect a pixel's depth. Press q or Esc to
quit. This uses the default depth profile advertised by the connected camera.
"""

import argparse

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Show Orbbec depth in a window.")
    parser.add_argument("--max-depth", type=float, default=4.0, help="Display range in metres.")
    parser.add_argument("--timeout-ms", type=int, default=1000)
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("OpenCV is missing. Install it with: python3 -m pip install opencv-python") from exc
    try:
        from pyorbbecsdk import Pipeline
    except ImportError as exc:
        raise SystemExit(
            "Orbbec SDK is missing. Install it with: python3 -m pip install --upgrade pyorbbecsdk2"
        ) from exc

    pipeline = Pipeline()
    started = False
    window_name = "Orbbec Femto Bolt depth"
    mouse = {"x": None, "y": None}

    def on_mouse(_event, x, y, _flags, _param):
        mouse["x"] = x
        mouse["y"] = y

    try:
        pipeline.start()
        started = True
        print(
            f"[orbbec-depth] depth display range 0-{args.max_depth:g}m; press q or Esc to quit",
            flush=True,
        )
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, on_mouse)

        while True:
            frames = pipeline.wait_for_frames(args.timeout_ms)
            if frames is None:
                continue
            depth_frame = frames.get_depth_frame()
            if depth_frame is None:
                continue

            depth_raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(
                depth_frame.get_height(), depth_frame.get_width()
            )
            depth_m = depth_raw.astype(np.float32) * float(depth_frame.get_depth_scale()) * 0.001
            valid = depth_m > 0.0

            depth_norm = np.clip(depth_m / args.max_depth, 0.0, 1.0)
            depth_u8 = (255.0 * (1.0 - depth_norm)).astype(np.uint8)
            depth_u8[~valid] = 0
            display = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)

            x, y = mouse["x"], mouse["y"]
            if x is not None and y is not None and 0 <= x < depth_m.shape[1] and 0 <= y < depth_m.shape[0]:
                value = float(depth_m[y, x])
                text = f"x={x} y={y} depth={value:.3f}m" if value > 0.0 else f"x={x} y={y} depth=invalid"
                cv2.circle(display, (x, y), 4, (255, 255, 255), 1)
            else:
                text = "move mouse over image"

            cv2.rectangle(display, (0, 0), (460, 30), (0, 0, 0), -1)
            cv2.putText(display, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.setWindowTitle(window_name, text)
            cv2.imshow(window_name, display)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    finally:
        if started:
            pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
