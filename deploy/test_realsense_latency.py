import argparse
import statistics
import time

import numpy as np


def percentile(values, ratio):
    if not values:
        return float("nan")
    values = sorted(values)
    index = int(round((len(values) - 1) * ratio))
    return values[index]


def format_stats(name, values, unit="ms"):
    if not values:
        return f"{name}: n=0"
    return (
        f"{name}: n={len(values)} "
        f"mean={statistics.mean(values):.2f}{unit} "
        f"p50={percentile(values, 0.50):.2f}{unit} "
        f"p95={percentile(values, 0.95):.2f}{unit} "
        f"max={max(values):.2f}{unit}"
    )


def timestamp_domain_name(rs, domain):
    mapping = {
        rs.timestamp_domain.hardware_clock: "hardware_clock",
        rs.timestamp_domain.system_time: "system_time",
        rs.timestamp_domain.global_time: "global_time",
    }
    return mapping.get(domain, str(domain))


def frame_metadata(frame, metadata_value):
    """Return a metadata value, or None when this camera/driver does not expose it."""
    if not frame.supports_frame_metadata(metadata_value):
        return None
    return float(frame.get_frame_metadata(metadata_value))


def main():
    parser = argparse.ArgumentParser(
        description="Measure RealSense depth frame timing without touching Unitree SDK."
    )
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=270)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--serial", default=None)
    parser.add_argument("--copy-depth", action="store_true", help="Also time numpy depth copy.")
    args = parser.parse_args()

    import pyrealsense2 as rs

    context = rs.context()
    devices = list(context.query_devices())
    if not devices:
        raise SystemExit("[depth-latency] no RealSense device found")

    print("[depth-latency] devices:")
    for device in devices:
        name = device.get_info(rs.camera_info.name)
        serial = device.get_info(rs.camera_info.serial_number)
        print(f"  {name} serial={serial}")

    pipeline = rs.pipeline()
    config = rs.config()
    if args.serial:
        config.enable_device(str(args.serial))
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    if depth_sensor.supports(rs.option.global_time_enabled):
        try:
            depth_sensor.set_option(rs.option.global_time_enabled, 1.0)
            print("[depth-latency] enabled RealSense global_time")
        except RuntimeError as exc:
            print(f"[depth-latency] could not enable global_time: {exc}")
    depth_scale = depth_sensor.get_depth_scale()
    print(
        f"[depth-latency] stream={args.width}x{args.height}@{args.fps} "
        f"depth_scale={depth_scale:g}"
    )

    wait_ms_values = []
    interarrival_ms_values = []
    sensor_interval_ms_values = []
    age_ms_values = []
    copy_ms_values = []
    exposure_ms_values = []
    exposure_mid_to_usb_ms_values = []
    frame_gaps = 0
    samples = 0
    last_arrival_perf = None
    last_frame_ts = None
    last_frame_number = None
    start_perf = None
    domain_text = "unknown"

    try:
        print(f"[depth-latency] warming up {args.warmup} frames...")
        for _ in range(max(args.warmup, 0)):
            pipeline.wait_for_frames(timeout_ms=1000)

        print("[depth-latency] measuring...")
        start_perf = time.perf_counter()
        next_print = start_perf + 1.0
        while time.perf_counter() - start_perf < args.seconds:
            wait_start_perf = time.perf_counter()
            frames = pipeline.wait_for_frames(timeout_ms=1000)
            wait_end_perf = time.perf_counter()
            arrival_time_ms = time.time() * 1000.0
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                continue

            samples += 1
            wait_ms_values.append((wait_end_perf - wait_start_perf) * 1000.0)

            frame_ts = float(depth_frame.get_timestamp())
            frame_number = int(depth_frame.get_frame_number())
            domain = depth_frame.get_frame_timestamp_domain()
            domain_text = timestamp_domain_name(rs, domain)

            if last_arrival_perf is not None:
                interarrival_ms_values.append((wait_end_perf - last_arrival_perf) * 1000.0)
            last_arrival_perf = wait_end_perf

            if last_frame_ts is not None:
                sensor_interval_ms_values.append(frame_ts - last_frame_ts)
            last_frame_ts = frame_ts

            if last_frame_number is not None and frame_number > last_frame_number + 1:
                frame_gaps += frame_number - last_frame_number - 1
            last_frame_number = frame_number

            if domain in (rs.timestamp_domain.system_time, rs.timestamp_domain.global_time):
                age_ms_values.append(arrival_time_ms - frame_ts)

            # These metadata timestamps use the camera clock, so their difference is
            # independent of the host/global timestamp clock conversion.
            exposure_us = frame_metadata(depth_frame, rs.frame_metadata_value.actual_exposure)
            sensor_ts_us = frame_metadata(depth_frame, rs.frame_metadata_value.sensor_timestamp)
            usb_start_ts_us = frame_metadata(depth_frame, rs.frame_metadata_value.frame_timestamp)
            if exposure_us is not None:
                exposure_ms_values.append(exposure_us / 1000.0)
            if sensor_ts_us is not None and usb_start_ts_us is not None:
                mid_exposure_to_usb_ms = (usb_start_ts_us - sensor_ts_us) / 1000.0
                if mid_exposure_to_usb_ms >= 0.0:
                    exposure_mid_to_usb_ms_values.append(mid_exposure_to_usb_ms)

            if args.copy_depth:
                copy_start = time.perf_counter()
                _ = np.asanyarray(depth_frame.get_data()).copy()
                copy_ms_values.append((time.perf_counter() - copy_start) * 1000.0)

            now = time.perf_counter()
            if now >= next_print:
                fps = samples / max(now - start_perf, 1e-6)
                age_text = ""
                if age_ms_values:
                    age_text = f" age_latest={age_ms_values[-1]:.2f}ms"
                camera_text = ""
                if exposure_ms_values:
                    camera_text += f" exposure_latest={exposure_ms_values[-1]:.2f}ms"
                if exposure_mid_to_usb_ms_values:
                    camera_text += (
                        " mid_exposure_to_usb_latest="
                        f"{exposure_mid_to_usb_ms_values[-1]:.2f}ms"
                    )
                print(
                    f"[depth-latency] samples={samples} fps={fps:.1f} "
                    f"domain={domain_text} dropped={frame_gaps}{age_text}{camera_text}",
                    flush=True,
                )
                next_print = now + 1.0
    finally:
        pipeline.stop()

    elapsed = max(time.perf_counter() - start_perf, 1e-6) if start_perf is not None else 0.0
    print("\n[depth-latency] summary")
    print(f"elapsed={elapsed:.2f}s samples={samples} fps={samples / elapsed:.2f} dropped_frames={frame_gaps}")
    print(f"timestamp_domain={domain_text}")
    print(format_stats("wait_for_frames_block", wait_ms_values))
    print(format_stats("host_interarrival", interarrival_ms_values))
    print(format_stats("sensor_timestamp_interval", sensor_interval_ms_values))
    if age_ms_values:
        print(format_stats("estimated_frame_age", age_ms_values))
    else:
        print(
            "estimated_frame_age: unavailable because timestamps are not in system/global time"
        )
    if args.copy_depth:
        print(format_stats("numpy_depth_copy", copy_ms_values))
    if exposure_ms_values:
        print(format_stats("actual_exposure", exposure_ms_values))
    else:
        print("actual_exposure: unavailable in this camera/driver metadata")
    if exposure_mid_to_usb_ms_values:
        print(
            format_stats(
                "mid_exposure_to_usb_start",
                exposure_mid_to_usb_ms_values,
            )
        )
        print(
            "mid_exposure_to_usb_start includes the remaining half exposure, "
            "sensor readout, and D4 depth pipeline before USB transmission."
        )
    else:
        print("mid_exposure_to_usb_start: unavailable in this camera/driver metadata")


if __name__ == "__main__":
    main()
