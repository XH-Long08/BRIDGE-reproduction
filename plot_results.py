"""Generate figures from the recorded BRIDGE result tables.

The figures summarize final metrics; they are not training-epoch convergence
curves. Run from this directory with:

    python plot_results.py
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
SUMMARY = ROOT / "results_summary.md"
OUT_DIR = ROOT / "figures"
METRICS = ("ACC", "NMI", "ARI", "PUR")
NUMBER = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)")


def metric_value(value: str) -> float:
    match = NUMBER.match(value)
    if not match:
        raise ValueError(f"Cannot parse metric value: {value!r}")
    # For BDGP mean ± standard deviation entries, plot the reported mean.
    return float(match.group(1))


def read_results() -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for line in SUMMARY.read_text(encoding="utf-8-sig").splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) == 8 and cells[1].isdigit():
            dataset, cp_type, miss_rate, *tail = cells
            acc, nmi, ari, pur, status = tail
            results.append(
                {
                    "experiment": "IID main",
                    "pattern": "IID",
                    "dataset": dataset,
                    "cp_type": int(cp_type),
                    "miss_rate": float(miss_rate),
                    "ACC": metric_value(acc),
                    "NMI": metric_value(nmi),
                    "ARI": metric_value(ari),
                    "PUR": metric_value(pur),
                    "status": status,
                }
            )
        elif len(cells) == 9 and cells[2].isdigit():
            pattern, dataset, cp_type, miss_rate, acc, nmi, ari, pur, status = cells
            results.append(
                {
                    "experiment": "NUSWIDE Non-IID",
                    "pattern": pattern,
                    "dataset": dataset,
                    "cp_type": int(cp_type),
                    "miss_rate": float(miss_rate),
                    "ACC": metric_value(acc),
                    "NMI": metric_value(nmi),
                    "ARI": metric_value(ari),
                    "PUR": metric_value(pur),
                    "status": status,
                }
            )

    iid_count = sum(row["experiment"] == "IID main" for row in results)
    non_iid_count = len(results) - iid_count
    if (iid_count, non_iid_count) != (72, 10):
        raise ValueError(
            f"Expected 72 IID and 10 Non-IID rows, got {iid_count} and {non_iid_count}"
        )
    return results


def write_csv(results: list[dict[str, object]]) -> None:
    fields = ["experiment", "pattern", "dataset", "cp_type", "miss_rate", *METRICS, "status"]
    with (ROOT / "results_data.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)


def plot_iid(results: list[dict[str, object]]) -> None:
    iid = [row for row in results if row["experiment"] == "IID main"]
    datasets = list(dict.fromkeys(str(row["dataset"]) for row in iid))
    miss_rates = (0.3, 0.5, 0.7)
    method_names = {1: "CL", 2: "DD", 3: "MI"}
    columns = [(cp, rate) for cp in (1, 2, 3) for rate in miss_rates]
    labels = [f"{method_names[cp]}\n{rate:.1f}" for cp, rate in columns]

    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    cmap = matplotlib.colormaps["YlGnBu"]
    for ax, metric in zip(axes.flat, METRICS):
        grid = np.full((len(datasets), len(columns)), np.nan)
        for row in iid:
            r = datasets.index(str(row["dataset"]))
            c = columns.index((int(row["cp_type"]), float(row["miss_rate"])))
            grid[r, c] = float(row[metric])
        image = ax.imshow(grid, aspect="auto", cmap=cmap, vmin=0, vmax=100)
        ax.set_title(metric)
        ax.set_xticks(range(len(labels)), labels, fontsize=8)
        ax.set_yticks(range(len(datasets)), datasets)
        ax.set_xlabel("Method / missing rate")
        ax.set_ylabel("Dataset")
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                value = grid[r, c]
                if np.isfinite(value):
                    ax.text(c, r, f"{value:.1f}", ha="center", va="center",
                            fontsize=7, color="white" if value >= 58 else "#17212b")
        colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
        colorbar.set_label("Score (%)")

    fig.suptitle("BRIDGE IID main experiments — final metrics (%)", fontsize=14)
    fig.savefig(OUT_DIR / "main_results_heatmap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_noniid(results: list[dict[str, object]]) -> None:
    non_iid = [row for row in results if row["experiment"] == "NUSWIDE Non-IID"]
    patterns = list(dict.fromkeys(str(row["pattern"]) for row in non_iid))
    pattern_labels = {
        "随机噪声": "Random noise",
        "类别/数量不平衡": "Class/quantity imbalance",
    }
    markers = ("o", "s")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)

    for ax, metric in zip(axes.flat, METRICS):
        for pattern, marker in zip(patterns, markers):
            series = sorted(
                (row for row in non_iid if row["pattern"] == pattern),
                key=lambda row: float(row["miss_rate"]),
            )
            ax.plot(
                [100 * float(row["miss_rate"]) for row in series],
                [float(row[metric]) for row in series],
                marker=marker,
                linewidth=1.8,
                markersize=5,
                label=pattern_labels.get(pattern, pattern),
            )
        ax.set_title(metric)
        ax.set_xlabel("Missing rate (%)")
        ax.set_ylabel("Score (%)")
        ax.set_xticks([10, 30, 50, 70, 90])
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("NUSWIDE Non-IID experiments — final metrics (%)", fontsize=14)
    fig.savefig(OUT_DIR / "noniid_results_by_missing_rate.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    results = read_results()
    write_csv(results)
    plot_iid(results)
    plot_noniid(results)
    print(f"Parsed {len(results)} result rows (72 IID + 10 Non-IID).")
    print(f"Wrote figures and CSV under {ROOT}")


if __name__ == "__main__":
    main()
