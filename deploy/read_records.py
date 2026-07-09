import argparse
import select
import sys
import termios
import tty
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECORD_DIR = ROOT / "deploy" / "records"


def latest_record(record_dir):
    files = sorted(record_dir.glob("*.npz"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No .npz records found in {record_dir}")
    return files[0]


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
                    key = sys.stdin.read(1)
                    if key == "\x1b":
                        key += sys.stdin.read(2)
                    if self.handle_key(self.normalize_terminal_key(key)):
                        break
                self.plt.pause(0.001)
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    @staticmethod
    def normalize_terminal_key(key):
        mapping = {
            "\x1b[C": "right",
            "\x1b[D": "left",
            "\x1b[H": "home",
            "\x1b[F": "end",
            "\x1b": "escape",
        }
        return mapping.get(key, key.lower())

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
            ("foot_contacts", slice(3, 7)),
            ("applied_force", slice(7, 10)),
            ("applied_torque", slice(10, 13)),
            ("friction", slice(13, 14)),
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
    parser = argparse.ArgumentParser(
        description="Read a deploy --record .npz file. Use left/right keys to inspect each 0.5s sample."
    )
    parser.add_argument("record", nargs="?", type=Path, help="Path to record .npz. Defaults to latest in deploy/records.")
    parser.add_argument("--record-dir", type=Path, default=DEFAULT_RECORD_DIR)
    args = parser.parse_args()

    record_path = args.record if args.record is not None else latest_record(args.record_dir)
    viewer = RecordViewer(record_path)
    viewer.run()


if __name__ == "__main__":
    main()
