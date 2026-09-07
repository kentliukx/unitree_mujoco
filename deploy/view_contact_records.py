#!/usr/bin/env python3
"""Plot durable tactile-contact CSV logs produced by --record-contact."""

import argparse
import csv
import os
import select
import sys
import termios
import time
import tty
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECORD_DIR = ROOT / "deploy" / "records"
CONTACT_NAMES = ("FL", "FR", "RL", "RR")
PROBABILITY_NAMES = tuple(f"pred_{name}" for name in CONTACT_NAMES)


def read_terminal_key():
    key = os.read(sys.stdin.fileno(), 1).decode(errors="ignore")
    if key != "\x1b":
        return key.lower()
    deadline = time.monotonic() + 0.2
    while time.monotonic() < deadline:
        ready, _, _ = select.select([sys.stdin], [], [], max(deadline - time.monotonic(), 0.0))
        if not ready:
            break
        key += os.read(sys.stdin.fileno(), 1).decode(errors="ignore")
        if key[-1:] in "ABCD":
            break
    return {"\x1b[A": "up", "\x1b[B": "down"}.get(key, key.lower())


def select_log(record_dir):
    files = sorted(record_dir.glob("contact_*.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No contact_*.csv files found in {record_dir}")
    if not sys.stdin.isatty():
        raise RuntimeError("Contact-log selection requires an interactive terminal. Use --file instead.")

    selected = 0
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        while True:
            print("\033[2J\033[H", end="")
            print(
                f"[contact-viewer] {selected + 1}/{len(files)}  "
                "Up/Down (or K/J) select, Enter open, Q quit\n"
            )
            for index, path in enumerate(files):
                print(f"{'>' if index == selected else ' '} {path.name}")
            key = read_terminal_key()
            if key in ("up", "k"):
                selected = max(selected - 1, 0)
            elif key in ("down", "j"):
                selected = min(selected + 1, len(files) - 1)
            elif key in ("\r", "\n"):
                return files[selected]
            elif key == "q":
                raise SystemExit(0)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


def load_log(path):
    rows = []
    with Path(path).open("r", encoding="ascii", newline="") as file:
        for row in csv.DictReader(file):
            try:
                probabilities = None
                if all(row.get(name) not in (None, "") for name in PROBABILITY_NAMES):
                    probabilities = [float(row[name]) for name in PROBABILITY_NAMES]
                rows.append((float(row["time_s"]), [int(row[name]) for name in CONTACT_NAMES], probabilities))
            except (KeyError, TypeError, ValueError):
                # A sudden power loss can leave only the final CSV line partial.
                continue
    if not rows:
        raise RuntimeError(f"No complete contact samples found in {path}")
    probabilities = None
    if all(row[2] is not None for row in rows):
        probabilities = np.asarray([row[2] for row in rows], dtype=np.float32)
    return (
        np.asarray([row[0] for row in rows]),
        np.asarray([row[1] for row in rows], dtype=np.uint8),
        probabilities,
    )


def main():
    parser = argparse.ArgumentParser(description="Visualize durable tactile-contact CSV logs.")
    parser.add_argument("--file", type=Path, help="Contact CSV to open directly.")
    parser.add_argument("--record-dir", type=Path, default=DEFAULT_RECORD_DIR)
    args = parser.parse_args()

    path = args.file if args.file is not None else select_log(args.record_dir)
    times, contacts, probabilities = load_log(path)

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 6), num=f"Tactile contacts: {path.name}")
    for index, (name, axis) in enumerate(zip(CONTACT_NAMES, axes)):
        axis.step(times, contacts[:, index], where="post", linewidth=1.0, label="sensor")
        if probabilities is not None:
            axis.plot(times, probabilities[:, index], linewidth=1.0, label="estimated probability")
        axis.set_ylabel(name)
        axis.set_ylim(-0.15, 1.15)
        axis.set_yticks((0, 1))
        axis.grid(axis="x", alpha=0.3)
        axis.legend(loc="upper right")
    axes[-1].set_xlabel("time [s]")
    figure.suptitle(f"{path.name}  |  samples={len(times)}  duration={times[-1]:.3f}s")
    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
