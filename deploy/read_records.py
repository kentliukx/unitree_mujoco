import select
import os
import sys
import termios
import time
import tty
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECORD_DIR = ROOT / "deploy" / "records"


def read_terminal_key():
    """Read a complete cursor-key escape sequence from local or remote terminals."""
    key = os.read(sys.stdin.fileno(), 1).decode(errors="ignore")
    if key != "\x1b":
        return key
    deadline = time.monotonic() + 0.2
    while len(key) < 8:
        timeout = max(deadline - time.monotonic(), 0.0)
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        if not ready:
            break
        key += os.read(sys.stdin.fileno(), 1).decode(errors="ignore")
        if key.startswith(("\x1b[", "\x1bO")) and key[-1:] in "ABCDFH":
            break
    return key


def normalize_terminal_key(key):
    if key.startswith("\x1b"):
        cursor_keys = {
            "A": "up",
            "B": "down",
            "C": "right",
            "D": "left",
            "H": "home",
            "F": "end",
        }
        return cursor_keys.get(key[-1:], key)
    return key.lower()


def select_record(record_dir):
    files = sorted(record_dir.glob("*.npz"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No .npz records found in {record_dir}")
    if not sys.stdin.isatty():
        raise RuntimeError("Record selection requires an interactive terminal.")

    selected = 0
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        while True:
            print("\033[2J\033[H", end="")
            print(
                f"[record-viewer] {selected + 1}/{len(files)}  "
                "Up/Down (or K/J) select, Enter open, Q quit\n"
            )
            for index, path in enumerate(files):
                marker = ">" if index == selected else " "
                print(f"{marker} {path.name}")
            key = normalize_terminal_key(read_terminal_key())
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


def format_array(values, precision=4, full=False):
    threshold = values.size + 1 if full else 24
    return np.array2string(
        values,
        precision=precision,
        suppress_small=False,
        threshold=threshold,
        max_line_width=160,
    )


class RecordViewer:
    def __init__(self, path):
        self.path = Path(path)
        self.data = np.load(self.path, allow_pickle=False)
        self.index = 0
        self.num_frames = int(self.data["time_s"].shape[0])
        reconstructed = self.data["reconstructed_info"]
        if reconstructed.shape != (self.num_frames, 28):
            raise RuntimeError(
                f"Expected reconstructed_info shape ({self.num_frames}, 28), got {reconstructed.shape}"
            )

        import matplotlib.pyplot as plt

        self.plt = plt
        self.figure, self.axis = plt.subplots(
            num=f"Record viewer: {self.path.name}", figsize=(12, 5)
        )
        self.bars = self.axis.bar(np.arange(28), np.zeros(28, dtype=np.float32))
        finite = reconstructed[np.isfinite(reconstructed)]
        limit = max(1.0, float(np.max(np.abs(finite))) * 1.1) if finite.size else 1.0
        self.axis.set_ylim(-limit, limit)
        self.axis.set_xlim(-0.75, 27.75)
        self.axis.set_xlabel("reconstructed dimension")
        self.axis.set_ylabel("value")
        self.axis.set_xticks(np.arange(28))
        self.axis.grid(axis="y", alpha=0.3)
        self.figure.canvas.mpl_connect("key_press_event", self.on_key)
        self.figure.tight_layout()

    def run(self):
        print(f"[record-viewer] file={self.path}", flush=True)
        print(
            "[record-viewer] keys in terminal or figure: "
            "Left/A = previous, Right/D = next, Home = first, End = last, Q/Esc = quit",
            flush=True,
        )
        self.show_current()
        self.figure.show()
        if sys.stdin.isatty():
            self.terminal_key_loop()
        else:
            self.plt.show()

    def on_key(self, event):
        key = (event.key or "").lower()
        self.handle_key(key)

    def terminal_key_loop(self):
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            while self.plt.fignum_exists(self.figure.number):
                ready, _, _ = select.select([sys.stdin], [], [], 0.05)
                if ready:
                    if self.handle_key(normalize_terminal_key(read_terminal_key())):
                        break
                self.plt.pause(0.001)
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    def handle_key(self, key):
        if key in ("right", "d"):
            self.move(1)
        elif key in ("left", "a"):
            self.move(-1)
        elif key == "home":
            self.index = 0
            self.show_current()
        elif key == "end":
            self.index = self.num_frames - 1
            self.show_current()
        elif key in ("q", "escape"):
            self.plt.close(self.figure)
            return True
        return False

    def move(self, delta):
        next_index = int(np.clip(self.index + delta, 0, self.num_frames - 1))
        if next_index == self.index:
            return
        self.index = next_index
        self.show_current()

    def show_current(self):
        i = self.index
        reconstructed = self.data["reconstructed_info"][i]
        for bar, value in zip(self.bars, reconstructed):
            bar.set_height(float(value))
        contacts = self.data["sensor_contacts"][i]
        self.figure.suptitle(
            f"frame {i + 1}/{self.num_frames}  "
            f"t={float(self.data['time_s'][i]):.3f}s  "
            f"sensor FL={int(contacts[0])} FR={int(contacts[1])}"
        )
        self.figure.canvas.draw_idle()
        self.print_frame(i)

    def print_frame(self, i):
        print("\n" + "=" * 100, flush=True)
        print(f"[record] frame {i + 1}/{self.num_frames} file={self.path.name}", flush=True)
        print(
            f"time_s={float(self.data['time_s'][i]):.6f} "
            f"wall_time_s={float(self.data['wall_time_s'][i]):.6f} "
            f"record_interval_s={float(self.data['record_interval_s']):.3f}",
            flush=True,
        )
        contacts = self.data["sensor_contacts"][i]
        reconstructed = self.data["reconstructed_info"][i]
        print(f"sensor_contacts FL={int(contacts[0])} FR={int(contacts[1])}", flush=True)
        print("estimated (0:15):", flush=True)
        names = (
            ("base_lin_vel", slice(0, 3)),
            ("contact_logits", slice(3, 7)),
            ("friction", slice(7, 8)),
            ("added_mass", slice(8, 9)),
            ("applied_force", slice(9, 12)),
            ("applied_torque", slice(12, 15)),
        )
        for name, section in names:
            print(f"{name}={format_array(reconstructed[section], precision=4, full=True)}", flush=True)
        print(
            f"reconstructed_ladder (15:28)="
            f"{format_array(reconstructed[15:28], precision=4, full=True)}",
            flush=True,
        )


def main():
    record_path = select_record(DEFAULT_RECORD_DIR)
    viewer = RecordViewer(record_path)
    viewer.run()


if __name__ == "__main__":
    main()
