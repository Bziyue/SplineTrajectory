#!/usr/bin/env python3
"""Run and visualize the convex-hull trajectory optimization demo."""

from __future__ import annotations

import argparse
import csv
import subprocess
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle


ROOT = Path(__file__).resolve().parents[1]


def load_numeric_csv(path: Path) -> np.ndarray:
    return np.genfromtxt(path, delimiter=",", names=True)


def load_controls(path: Path) -> dict[int, np.ndarray]:
    groups: dict[int, list[tuple[float, float]]] = defaultdict(list)
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            groups[int(row["piece"])].append((float(row["x"]), float(row["y"])))
    return {key: np.asarray(value) for key, value in groups.items()}


def load_control_pieces(path: Path) -> list[tuple[int, np.ndarray]]:
    groups: dict[int, list[tuple[float, float]]] = defaultdict(list)
    source_segments: dict[int, int] = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            piece = int(row["piece"])
            source_segments[piece] = int(row["source_segment"])
            groups[piece].append((float(row["x"]), float(row["y"])))
    return [
        (source_segments[piece], np.asarray(groups[piece]))
        for piece in sorted(groups)
    ]


def convex_hull(points: np.ndarray) -> np.ndarray:
    unique = sorted(set(map(tuple, np.asarray(points, dtype=float))))
    if len(unique) <= 2:
        return np.asarray(unique)

    def cross(origin, a, b):
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (
            a[1] - origin[1]
        ) * (b[0] - origin[0])

    lower = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1])


def draw_control_hulls(
    ax, controls: dict[int, np.ndarray], color: str, alpha: float, label: str
) -> None:
    first = True
    for points in controls.values():
        hull = convex_hull(points)
        if len(hull) >= 3:
            closed = np.vstack([hull, hull[0]])
            ax.fill(
                closed[:, 0],
                closed[:, 1],
                color=color,
                alpha=alpha,
                linewidth=0.5,
                edgecolor=color,
                label=label if first else None,
            )
        ax.plot(points[:, 0], points[:, 1], ".", color=color, ms=2.0, alpha=0.7)
        first = False


def draw_environment(ax, path: Path) -> None:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return
    if rows[0]["type"] == "circle":
        for row in rows:
            center = (float(row["cx"]), float(row["cy"]))
            radius = float(row["radius"])
            clearance = float(row["clearance"])
            ax.add_patch(Circle(center, radius, color="#d63031", alpha=0.38))
            ax.add_patch(
                Circle(
                    center,
                    radius + clearance,
                    fill=False,
                    color="#d63031",
                    linestyle="--",
                    linewidth=1.2,
                )
            )
    else:
        for index, row in enumerate(rows):
            xmin, ymin = float(row["xmin"]), float(row["ymin"])
            xmax, ymax = float(row["xmax"]), float(row["ymax"])
            ax.add_patch(
                Rectangle(
                    (xmin, ymin),
                    xmax - xmin,
                    ymax - ymin,
                    facecolor=plt.cm.Set3(index / max(1, len(rows) - 1)),
                    edgecolor="#636e72",
                    alpha=0.20,
                    linewidth=1.0,
                )
            )


def draw_scenario(ax, data_dir: Path, name: str, title: str) -> None:
    initial = load_numeric_csv(data_dir / f"{name}_initial_trajectory.csv")
    trajectory = load_numeric_csv(data_dir / f"{name}_trajectory.csv")
    bezier = load_controls(data_dir / f"{name}_bezier.csv")
    minvo = load_controls(data_dir / f"{name}_minvo.csv")

    draw_environment(ax, data_dir / f"{name}_environment.csv")
    draw_control_hulls(ax, bezier, "#0984e3", 0.075, "subdivided Bezier hull")
    draw_control_hulls(ax, minvo, "#e17055", 0.045, "MINVO hull")
    ax.plot(initial["x"], initial["y"], "--", color="#636e72", lw=1.4, label="initial")
    ax.plot(
        trajectory["x"],
        trajectory["y"],
        color="#2d3436",
        lw=2.4,
        label="optimized MINCO",
    )
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.18)
    ax.legend(fontsize=7, ncol=2, loc="upper center")


def draw_bezier_subdivision_stack(data_dir: Path, image_dir: Path) -> Path:
    trajectory = load_numeric_csv(
        data_dir / "bezier_subdivision_trajectory.csv"
    )
    segment_colors = ("#0984e3", "#00a878", "#e17055")
    fig, axes = plt.subplots(
        5,
        1,
        figsize=(13.5, 16.5),
        dpi=180,
        sharex=True,
        sharey=True,
    )

    all_points = np.column_stack((trajectory["x"], trajectory["y"]))
    padding = 0.08 * np.ptp(all_points, axis=0)
    padding = np.maximum(padding, np.array([0.15, 0.15]))
    lower = np.min(all_points, axis=0) - padding
    upper = np.max(all_points, axis=0) + padding

    for depth, ax in enumerate(axes):
        pieces = load_control_pieces(
            data_dir / f"bezier_subdivision_depth_{depth}.csv"
        )
        for source_segment, points in pieces:
            color = segment_colors[source_segment]
            hull = convex_hull(points)
            if len(hull) >= 3:
                closed = np.vstack([hull, hull[0]])
                ax.fill(
                    closed[:, 0],
                    closed[:, 1],
                    facecolor=color,
                    edgecolor=color,
                    linewidth=1.0,
                    alpha=0.18,
                    zorder=1,
                )
            ax.plot(
                points[:, 0],
                points[:, 1],
                "--o",
                color=color,
                linewidth=0.75,
                markersize=2.2,
                alpha=0.82,
                zorder=6,
            )

        ax.plot(
            trajectory["x"],
            trajectory["y"],
            color="#1f2328",
            linewidth=1.8,
            label="trajectory",
            zorder=4,
        )
        pieces_per_source = 1 << depth
        ax.set_title(
            f"Subdivision depth {depth} — "
            f"{pieces_per_source} piece"
            f"{'s' if pieces_per_source > 1 else ''} per source segment, "
            f"{3 * pieces_per_source} hulls total",
            fontsize=12,
        )
        ax.set_ylabel("y")
        ax.set_xlim(lower[0], upper[0])
        ax.set_ylim(lower[1], upper[1])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.18)

    axes[-1].set_xlabel("x")
    legend_handles = [
        plt.Line2D(
            [0], [0], color="#1f2328", linewidth=2.1, label="trajectory"
        )
    ]
    for segment, color in enumerate(segment_colors, start=1):
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                color=color,
                linestyle="--",
                marker="o",
                markersize=3,
                linewidth=1.0,
                label=f"source segment {segment}: control polygon / hull",
            )
        )
    axes[0].legend(
        handles=legend_handles,
        fontsize=8,
        ncol=2,
        loc="upper center",
    )
    fig.suptitle(
        "Bezier convex-hull tightening on the same 3-segment quintic trajectory",
        fontsize=15,
        y=0.998,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.987))
    output = image_dir / "bezier_subdivision_depths.png"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    parser.add_argument(
        "--data-dir", type=Path, default=ROOT / "convex_hull_demo_output"
    )
    parser.add_argument("--skip-run", action="store_true")
    args = parser.parse_args()

    executable = args.build_dir / "convex_hull_optimization_demo"
    if not args.skip_run:
        subprocess.run(
            ["cmake", "--build", str(args.build_dir), "--target", executable.name, "-j2"],
            check=True,
        )
        subprocess.run([str(executable), str(args.data_dir)], check=True)

    image_dir = ROOT / "docs" / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 8.8), dpi=180)
    draw_scenario(axes[0, 0], args.data_dir, "esdf", "ESDF obstacle optimization")
    draw_scenario(axes[0, 1], args.data_dir, "corridor", "Convex-corridor optimization")

    for name, color, label in [
        ("esdf", "#0984e3", "ESDF"),
        ("corridor", "#e17055", "corridor"),
    ]:
        history = load_numeric_csv(args.data_dir / f"{name}_history.csv")
        axes[1, 0].semilogy(history["iteration"], history["cost"], color=color, label=label)
    axes[1, 0].set_title("L-BFGS objective convergence")
    axes[1, 0].set_xlabel("iteration")
    axes[1, 0].set_ylabel("objective")
    axes[1, 0].grid(alpha=0.2)
    axes[1, 0].legend()

    for name, color in [("esdf", "#0984e3"), ("corridor", "#e17055")]:
        trajectory = load_numeric_csv(args.data_dir / f"{name}_trajectory.csv")
        normalized_time = (trajectory["t"] - trajectory["t"][0]) / (
            trajectory["t"][-1] - trajectory["t"][0]
        )
        axes[1, 1].plot(
            normalized_time,
            trajectory["speed"] / 3.2,
            color=color,
            label=f"{name} speed / limit",
        )
        axes[1, 1].plot(
            normalized_time,
            trajectory["acceleration"] / 5.0,
            color=color,
            linestyle="--",
            label=f"{name} acceleration / limit",
        )
    axes[1, 1].axhline(1.0, color="#d63031", lw=1.0, linestyle=":")
    axes[1, 1].set_title("Sampled dynamic constraints")
    axes[1, 1].set_xlabel("normalized time")
    axes[1, 1].set_ylabel("value / limit")
    axes[1, 1].grid(alpha=0.2)
    axes[1, 1].legend(fontsize=8)

    fig.tight_layout()
    overview = image_dir / "convex_hull_optimization.png"
    fig.savefig(overview, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.7), dpi=180)
    for ax, name, title in zip(
        axes,
        ("esdf", "corridor"),
        ("Bezier depth 2 vs. MINVO — ESDF", "Bezier depth 2 vs. MINVO — corridor"),
    ):
        trajectory = load_numeric_csv(args.data_dir / f"{name}_trajectory.csv")
        bezier = load_controls(args.data_dir / f"{name}_bezier.csv")
        minvo = load_controls(args.data_dir / f"{name}_minvo.csv")
        draw_control_hulls(ax, minvo, "#e17055", 0.15, "one MINVO hull / segment")
        draw_control_hulls(ax, bezier, "#0984e3", 0.13, "4 Bezier hulls / segment")
        ax.plot(trajectory["x"], trajectory["y"], "k", lw=2.2, label="trajectory")
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.18)
        ax.legend(fontsize=8)
    fig.tight_layout()
    comparison = image_dir / "convex_hull_basis_comparison.png"
    fig.savefig(comparison, bbox_inches="tight")
    plt.close(fig)

    subdivision_stack = draw_bezier_subdivision_stack(
        args.data_dir, image_dir
    )

    print(f"Wrote {overview}")
    print(f"Wrote {comparison}")
    print(f"Wrote {subdivision_stack}")


if __name__ == "__main__":
    main()
