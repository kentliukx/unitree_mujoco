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
        self.depth_shape = tuple(int(v) for v in self.data["depth_shape"])
        self.height_shape = tuple(int(v) for v in self.data["height_scan_shape"])
        self.current_depth = np.zeros(self.depth_shape, dtype=np.float32)

        import matplotlib.pyplot as plt

        self.plt = plt
        self.figure, (self.depth_axis, self.height_axis) = plt.subplots(
            1,
            2,
            num=f"Record viewer: {self.path.name}",
            figsize=(10, 5),
        )

        self.depth_image = self.depth_axis.imshow(
            np.zeros(self.depth_shape, dtype=np.float32),
            cmap="turbo",
            vmin=0.1,
            vmax=2.0,
        )
        self.height_image = self.height_axis.imshow(
            np.zeros(self.height_shape, dtype=np.float32),
            cmap="viridis",
            origin="lower",
            vmin=-1.0,
            vmax=1.0,
        )
        self.depth_axis.set_title("Depth [m]")
        self.height_axis.set_title("Reconstructed height")
        for axis in (self.depth_axis, self.height_axis):
            axis.set_xlabel("col")
            axis.set_ylabel("row")
        self.figure.colorbar(self.depth_image, ax=self.depth_axis, fraction=0.046, pad=0.04)
        self.figure.colorbar(self.height_image, ax=self.height_axis, fraction=0.046, pad=0.04)
        self.depth_cursor_text = self.depth_axis.annotate(
            "",
            xy=(0, 0),
            xytext=(8, 8),
            textcoords="offset points",
            color="white",
            bbox={"boxstyle": "round", "fc": "black", "alpha": 0.75},
        )
        self.depth_cursor_text.set_visible(False)
        self.figure.canvas.mpl_connect("key_press_event", self.on_key)
        self.figure.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
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

    def on_mouse_move(self, event):
        if event.inaxes != self.depth_axis or event.xdata is None or event.ydata is None:
            if self.depth_cursor_text.get_visible():
                self.depth_cursor_text.set_visible(False)
                self.figure.canvas.draw_idle()
            return

        col = int(event.xdata)
        row = int(event.ydata)
        if not (0 <= row < self.depth_shape[0] and 0 <= col < self.depth_shape[1]):
            return
        depth_m = float(self.current_depth[row, col])
        self.depth_cursor_text.xy = (col, row)
        self.depth_cursor_text.set_text(f"({col}, {row})  {depth_m:.3f} m")
        self.depth_cursor_text.set_visible(True)
        self.figure.canvas.draw_idle()

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
        depth = self.data["depth"][i].reshape(self.depth_shape)
        reconstructed_height = self.data["reconstructed_height"][i].reshape(self.height_shape)

        self.current_depth = depth
        self.depth_image.set_data(depth)
        self.depth_image.set_clim(vmin=float(np.nanmin(depth)), vmax=float(np.nanmax(depth)))
        self.height_image.set_data(reconstructed_height)
        self.figure.suptitle(
            f"frame {i + 1}/{self.num_frames}  "
            f"t={float(self.data['time_s'][i]):.3f}s  "
            f"goal={format_array(self.data['goal'][i], precision=3, full=True)}  "
            f"ladder={format_array(self.data['reconstructed_ladder'][i], precision=3, full=True)}"
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
        print("current_proprio:", flush=True)
        self.print_proprio(i)
        print(f"command_obs={format_array(self.data['goal'][i], precision=4, full=True)}", flush=True)
        print(f"reached_goal={float(self.data['reached_goal'][i]):.0f}", flush=True)
        if "contact_precision" in self.data:
            print(
                f"contact_precision={format_array(self.data['contact_precision'][i], precision=4, full=True)}",
                flush=True,
            )
        print("estimated:", flush=True)
        self.print_estimated(i)
        print(
            f"ladder_obs={format_array(self.data['reconstructed_ladder'][i], precision=4, full=True)}",
            flush=True,
        )

    def print_estimated(self, i):
        estimated = self.data["estimated"][i]
        names = (
            ("base_lin_vel", slice(0, 3)),
            ("friction", slice(3, 4)),
            ("added_mass", slice(4, 5)),
            ("applied_force", slice(5, 8)),
            ("applied_torque", slice(8, 11)),
        )
        for name, section in names:
            print(f"{name}={format_array(estimated[section], precision=4, full=True)}", flush=True)

    def print_proprio(self, i):
        proprio = self.data["proprio"][i]
        sections = (
            ("ang_vel_scaled", proprio[0:3]),
            ("projected_gravity", proprio[3:6]),
            ("q_minus_default", proprio[6:18]),
            ("dq_scaled", proprio[18:30]),
            ("last_action", proprio[30:42]),
        )
        for name, values in sections:
            print(f"{name}={format_array(values, precision=4, full=True)}", flush=True)


def main():
    record_path = select_record(DEFAULT_RECORD_DIR)
    viewer = RecordViewer(record_path)
    viewer.run()


if __name__ == "__main__":
    main()
