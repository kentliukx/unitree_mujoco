#!/usr/bin/env python3
"""Read four precise foot-contact booleans from the tactile-sensor MCU.

Frame format, repeated without a checksum:
    0xA5, FL, FR, BL, BR
Each contact byte is interpreted as false when zero and true otherwise.
"""

import argparse
import glob
import sys
import time


SOF = 0xA5
FRAME_SIZE = 5
STM32_VID = 0x0483
STM32_PID = 0x5740
CONTACT_NAMES = ("FL", "FR", "BL", "BR")


class TactileFrameDecoder:
    """Incrementally recover fixed-size contact frames from a serial byte stream."""

    def __init__(self):
        self.buffer = bytearray()

    def feed(self, data):
        self.buffer.extend(data)
        frames = []
        while True:
            try:
                sof_index = self.buffer.index(SOF)
            except ValueError:
                self.buffer.clear()
                return frames
            if sof_index:
                del self.buffer[:sof_index]
            if len(self.buffer) < FRAME_SIZE:
                return frames
            frame = self.buffer[:FRAME_SIZE]
            del self.buffer[:FRAME_SIZE]
            frames.append(tuple(value != 0 for value in frame[1:]))


def find_port(requested_port):
    if requested_port:
        return requested_port

    try:
        from serial.tools import list_ports
    except ImportError as error:
        raise RuntimeError("Missing pyserial. Install it with: python -m pip install pyserial") from error

    matches = [
        port.device
        for port in list_ports.comports()
        if port.vid == STM32_VID and port.pid == STM32_PID
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(f"Multiple STM32 virtual serial ports found: {matches}. Use --port.")

    by_id_matches = glob.glob("/dev/serial/by-id/*STMicroelectronics*")
    if len(by_id_matches) == 1:
        return by_id_matches[0]
    raise RuntimeError(
        "STM32 Virtual COM Port (VID:PID 0483:5740) was not found. "
        "Check `ls /dev/ttyACM*` or provide --port /dev/ttyACM0."
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", help="Serial device; auto-detect STM32 0483:5740 when omitted.")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate (default: 115200).")
    parser.add_argument("--verbose", action="store_true", help="Print every received frame, not only changes.")
    parser.add_argument("--list", action="store_true", help="List serial ports and exit.")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        import serial
        from serial.tools import list_ports
    except ImportError as error:
        raise SystemExit("Missing pyserial. Install it with: python -m pip install pyserial") from error

    if args.list:
        for port in list_ports.comports():
            print(f"{port.device}: vid={port.vid!s} pid={port.pid!s} {port.description}")
        return

    try:
        port = find_port(args.port)
        connection = serial.Serial(port, args.baud, timeout=0.1)
    except Exception as error:
        raise SystemExit(f"Could not open tactile serial port: {error}") from error

    print(f"[tactile] connected port={port} baud={args.baud}", flush=True)
    decoder = TactileFrameDecoder()
    previous = None
    try:
        while True:
            data = connection.read(connection.in_waiting or 1)
            if not data:
                continue
            for contacts in decoder.feed(data):
                if args.verbose or contacts != previous:
                    state = " ".join(
                        f"{name}={'contact' if contact else 'air'}"
                        for name, contact in zip(CONTACT_NAMES, contacts)
                    )
                    print(f"[tactile] {state}", flush=True)
                previous = contacts
    except KeyboardInterrupt:
        print("\n[tactile] stopped", flush=True)
    finally:
        connection.close()


if __name__ == "__main__":
    main()
