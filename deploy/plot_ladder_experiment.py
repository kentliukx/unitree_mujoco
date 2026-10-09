#!/usr/bin/env python3
"""Plot tactile and reconstructed ladder quantities from a durable CSV log."""

import argparse
import csv
from pathlib import Path

import numpy as np


# Plot switches: set an item to False to omit that subplot.
PLOT_FL_CONTACT = True
PLOT_FR_CONTACT = True
PLOT_INCLINATION = True
PLOT_SPACING = True
PLOT_FRICTION = False
PLOT_HEIGHT_IN = 0.45
FIXED_VERTICAL_SPACE_IN = 0.15
FIGURE_WIDTH_IN = 7.2
NOMINAL_INCLINATION_DEG = 60.0
NOMINAL_SPACING_M = 0.20

RECONSTRUCTED_NAMES = tuple(f"reconstructed_{index:02d}" for index in range(28))


def load_window(path, start_s, end_s):
    rows = []
    with path.open("r", encoding="ascii", newline="") as file:
        for row in csv.DictReader(file):
            try:
                time_s = float(row["time_s"])
                if start_s <= time_s <= end_s:
                    reconstructed = [float(row[name]) for name in RECONSTRUCTED_NAMES]
                    rows.append((time_s, int(row["FL"]), int(row["FR"]), reconstructed))
            except (KeyError, TypeError, ValueError):
                # A sudden power loss may leave the final CSV row incomplete.
                continue
    if not rows:
        raise RuntimeError(f"No complete samples in [{start_s}, {end_s}] s: {path}")
    return (
        np.asarray([row[0] for row in rows], dtype=np.float64),
        np.asarray([[row[1], row[2]] for row in rows], dtype=np.uint8),
        np.asarray([row[3] for row in rows], dtype=np.float32),
    )


def sigmoid(values):
    values = np.clip(values, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-values))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--start", type=float, default=43.0)
    parser.add_argument("--end", type=float, default=55.0)
    parser.add_argument("--inclination-deg", type=float, default=NOMINAL_INCLINATION_DEG)
    parser.add_argument("--spacing-m", type=float, default=NOMINAL_SPACING_M)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    output = args.output
    if output is None:
        output = args.csv.with_name(
            f"{args.csv.stem}_{args.start:g}s_{args.end:g}s_analysis.png"
        )

    time_s, contacts, reconstructed = load_window(args.csv, args.start, args.end)
    relative_time_s = time_s - args.start
    contact_probability = sigmoid(reconstructed[:, 3:7])
    # Training supervises the sampled friction, while simulation combines it
    # with the foot material coefficient 1.0 using averaging mode.
    effective_friction = (reconstructed[:, 7] + 1.0) * 0.5
    spacing_cm = reconstructed[:, 23] * 100.0
    inclination_deg = np.rad2deg(reconstructed[:, 25])

    import matplotlib.pyplot as plt

    plot_specs = []
    if PLOT_FL_CONTACT:
        plot_specs.append(("contact", 0, "FL"))
    if PLOT_FR_CONTACT:
        plot_specs.append(("contact", 1, "FR"))
    if PLOT_INCLINATION:
        plot_specs.append(("inclination",))
    if PLOT_SPACING:
        plot_specs.append(("spacing",))
    if PLOT_FRICTION:
        plot_specs.append(("friction",))
    if not plot_specs:
        raise RuntimeError("At least one PLOT_* option must be True.")

    figure, axes_grid = plt.subplots(
        len(plot_specs),
        1,
        sharex=True,
        squeeze=False,
        figsize=(
            FIGURE_WIDTH_IN,
            FIXED_VERTICAL_SPACE_IN + PLOT_HEIGHT_IN * len(plot_specs),
        ),
        constrained_layout=True,
    )
    axes = axes_grid[:, 0]
    legend_style = {
        "fontsize": 5.8,
        "frameon": False,
        "handlelength": 1.2,
        "handletextpad": 0.4,
        "labelspacing": 0.2,
        "borderaxespad": 0.2,
    }

    actual_spacing_cm = args.spacing_m * 100.0
    for axis, spec in zip(axes, plot_specs):
        kind = spec[0]
        if kind == "contact":
            _, index, name = spec
            axis.step(
                relative_time_s,
                contacts[:, index],
                where="post",
                color="black",
                linewidth=0.8,
                label=f"{name} sensor",
            )
            axis.plot(
                relative_time_s,
                contact_probability[:, index],
                color="#147d92",
                linewidth=0.9,
                label=f"reconstructed {name} contact precision",
            )
            axis.set_ylim(-0.15, 1.15)
            axis.set_yticks((0.0, 0.5, 1.0))
            axis.legend(loc="upper right", **legend_style)
        elif kind == "inclination":
            axis.axhline(args.inclination_deg, color="#bd3f31", linestyle="--")
            axis.plot(
                relative_time_s,
                inclination_deg,
                color="#1f5aa6",
                linewidth=0.9,
                label="reconstructed inclination",
            )
            axis.set_ylim(15.0, 75.0)
            axis.legend(loc="lower right", **legend_style)
        elif kind == "spacing":
            axis.axhline(actual_spacing_cm, color="#bd3f31", linestyle="--")
            axis.plot(
                relative_time_s,
                spacing_cm,
                color="#1f5aa6",
                linewidth=0.9,
                label="reconstructed spacing",
            )
            axis.legend(loc="lower right", **legend_style)
        elif kind == "friction":
            axis.plot(
                relative_time_s,
                effective_friction,
                color="#805515",
                linewidth=0.9,
                label="reconstructed friction",
            )
            axis.set_ylim(0.25, 1.5)
            axis.legend(loc="upper right", **legend_style)

    axes[-1].set_xlabel("time [s]", fontsize=7)

    for axis in axes:
        axis.set_xlim(0.0, args.end - args.start)
        axis.grid(True, alpha=0.25)
        axis.tick_params(labelsize=5.8)
        axis.yaxis.label.set_size(6.5)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220)
    plt.close(figure)

    print(f"[plot] source={args.csv}")
    print(f"[plot] requested_window=({args.start:.3f}, {args.end:.3f}) s")
    print(f"[plot] samples={len(time_s)} observed_window=({time_s[0]:.3f}, {time_s[-1]:.3f}) s")
    print(f"[plot] output={output}")


if __name__ == "__main__":
    main()
