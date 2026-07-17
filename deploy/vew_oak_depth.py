#!/usr/bin/env python3
"""Preview metric stereo depth from an OAK-D camera.

Move the mouse over the image to inspect a pixel's depth. Press q or Esc to
quit. Supports both the legacy DepthAI v2 API and the current v3 API.
"""

import argparse

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Show OAK stereo depth in a window.")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--resolution", type=int, choices=(400, 480, 720, 800), default=400)
    parser.add_argument("--max-depth", type=float, default=4.0, help="Display range in metres.")
    parser.add_argument("--extended-disparity", action="store_true", help="Improve close-range depth.")
    parser.add_argument("--subpixel", action="store_true", help="Improve long-range depth precision.")
    parser.add_argument("--no-lr-check", action="store_true", help="Disable left-right consistency checking.")
    return parser.parse_args()


def start_v2_pipeline(dai, args):
    pipeline = dai.Pipeline()
    left = pipeline.create(dai.node.MonoCamera)
    right = pipeline.create(dai.node.MonoCamera)
    stereo = pipeline.create(dai.node.StereoDepth)
    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("depth")

    resolution_names = {
        400: dai.MonoCameraProperties.SensorResolution.THE_400_P,
        480: dai.MonoCameraProperties.SensorResolution.THE_480_P,
        720: dai.MonoCameraProperties.SensorResolution.THE_720_P,
        800: dai.MonoCameraProperties.SensorResolution.THE_800_P,
    }
    for camera, name in ((left, "left"), (right, "right")):
        camera.setCamera(name)
        camera.setResolution(resolution_names[args.resolution])
        camera.setFps(args.fps)

    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setLeftRightCheck(not args.no_lr_check)
    stereo.setExtendedDisparity(args.extended_disparity)
    stereo.setSubpixel(args.subpixel)
    left.out.link(stereo.left)
    right.out.link(stereo.right)
    stereo.depth.link(xout.input)

    device = dai.Device(pipeline)
    queue = device.getOutputQueue(name="depth", maxSize=2, blocking=False)
    return queue, device.close, "DepthAI v2"


def start_v3_pipeline(dai, args):
    pipeline = dai.Pipeline()
    left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
    stereo = pipeline.create(dai.node.StereoDepth)

    left.requestFullResolutionOutput().link(stereo.left)
    right.requestFullResolutionOutput().link(stereo.right)
    stereo.setRectification(True)
    stereo.setLeftRightCheck(not args.no_lr_check)
    stereo.setExtendedDisparity(args.extended_disparity)
    stereo.setSubpixel(args.subpixel)
    queue = stereo.depth.createOutputQueue()
    pipeline.start()

    def close():
        pipeline.stop()

    return queue, close, "DepthAI v3"


def main():
    args = parse_args()
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("OpenCV is missing. Install it with: python3 -m pip install opencv-python") from exc
    try:
        import depthai as dai
    except ImportError as exc:
        raise SystemExit("DepthAI is missing. Install it with: python3 -m pip install depthai") from exc

    start_pipeline = start_v3_pipeline if hasattr(dai.node, "Camera") else start_v2_pipeline
    try:
        depth_queue, close_pipeline, api_name = start_pipeline(dai, args)
    except Exception as exc:
        raise SystemExit(f"Could not start OAK depth pipeline: {exc}") from exc

    window_name = "OAK depth"
    mouse = {"x": None, "y": None}

    def on_mouse(_event, x, y, _flags, _param):
        mouse["x"] = x
        mouse["y"] = y

    print(
        f"[oak-depth] {api_name}, depth display range 0-{args.max_depth:g}m; press q or Esc to quit",
        flush=True,
    )
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)
    try:
        while True:
            frame = depth_queue.get()
            depth_mm = frame.getFrame().astype(np.float32)
            depth_m = depth_mm * 0.001
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
        close_pipeline()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
