import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from pareto_marl.design import CONDITIONS, SEEDS, SIZES

# plan_stage0.md §1, pre-registered: do not change after the run
SPLITS = ("segment", "leg", "joint")
ALPHA = 0.05
GO_MIN_ACTUATORS = 32
CURVE_BINS = 50

# validated categorical order (dataviz palette slots 1-4), light surface
COLORS = dict(
    zip(CONDITIONS, ("#2a78d6", "#eb6834", "#1baf7a", "#eda100"), strict=True)
)
MARKERS = dict(zip(CONDITIONS, ("o", "s", "^", "D"), strict=True))
INK, MUTED = "#0b0b0b", "#52514e"

Results = dict[tuple[int, str], list[dict[str, Any]]]


def load_results(results_dir: Path) -> Results:
    paths = sorted(results_dir.glob("stage0-*.json"))
    if not paths:
        raise FileNotFoundError(
            f"no stage0-*.json in {results_dir}: check the path or copy "
            "the results from PD first"
        )
    by_program: Results = {}
    for path in paths:
        r = json.loads(path.read_text())
        if not math.isfinite(r["eval_return"]):
            raise ValueError(f"{path}: eval_return is {r['eval_return']}")
        by_program.setdefault((r["n_segs"], r["condition"]), []).append(r)
    check_complete(by_program, results_dir)
    return by_program


def check_complete(by_program: Results, results_dir: Path) -> None:
    missing = []
    for n_segs in SIZES:
        for condition in CONDITIONS:
            runs = by_program.get((n_segs, condition), [])
            seeds = sorted(r["seed"] for r in runs)
            if seeds != list(SEEDS):
                missing.append(f"n_segs={n_segs} {condition}: seeds {seeds}")
    if missing:
        raise ValueError(
            f"{results_dir} does not hold the full design (16 programs x "
            f"seeds {list(SEEDS)}); rerun these programs:\n  "
            + "\n  ".join(missing)
        )


def returns(by_program: Results, n_segs: int, condition: str) -> np.ndarray:
    return np.array([r["eval_return"] for r in by_program[n_segs, condition]])


def holm(p_values: list[float]) -> list[float]:
    m = len(p_values)
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(np.argsort(p_values)):
        running = max(running, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


def comparisons(by_program: Results) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for n_segs in SIZES:
        single = returns(by_program, n_segs, "single")
        for condition in SPLITS:
            split = returns(by_program, n_segs, condition)
            test = stats.ttest_ind(
                split, single, equal_var=False, alternative="greater"
            )
            rows.append(
                {
                    "n_segs": n_segs,
                    "actuators": by_program[n_segs, condition][0]["actuators"],
                    "condition": condition,
                    "k": by_program[n_segs, condition][0]["k"],
                    "mean_split": float(split.mean()),
                    "mean_single": float(single.mean()),
                    "ratio": float(split.mean() / single.mean()),
                    "t": float(test.statistic),
                    "p": float(test.pvalue),
                }
            )
    for row, p_holm in zip(rows, holm([r["p"] for r in rows]), strict=True):
        row["p_holm"] = p_holm
        row["passes"] = p_holm < ALPHA
    return rows


def verdict(by_program: Results) -> dict[str, Any]:
    rows = comparisons(by_program)
    go = any(r["passes"] and r["actuators"] >= GO_MIN_ACTUATORS for r in rows)
    runs = [r for runs in by_program.values() for r in runs]
    return {
        "go": go,
        "rule": (
            "one-sided Welch t-test split > single on eval_return per n_segs "
            f"and split, Holm over {len(rows)} comparisons; GO iff some "
            f"comparison at >= {GO_MIN_ACTUATORS} actuators has "
            f"p_holm < {ALPHA}"
        ),
        "run_ids": sorted({r["run_id"] for r in runs}),
        "git_commits": sorted({r["git_commit"] for r in runs}),
        "n_runs": len(runs),
        "comparisons": rows,
    }


def mean_std(values: list[float]) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values, ddof=1))


def program_table(by_program: Results) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for n_segs in SIZES:
        for condition in CONDITIONS:
            runs = by_program[n_segs, condition]
            ret = mean_std([r["eval_return"] for r in runs])
            vel = mean_std([r["eval_x_velocity"] for r in runs])
            rows.append(
                {
                    "n_segs": n_segs,
                    "actuators": runs[0]["actuators"],
                    "condition": condition,
                    "k": runs[0]["k"],
                    "actor_params": runs[0]["actor_params"],
                    "n_seeds": len(runs),
                    "return_mean": ret[0],
                    "return_std": ret[1],
                    "x_velocity_mean": vel[0],
                    "x_velocity_std": vel[1],
                    "steps_per_s": runs[0]["steps_per_s"],
                    "compile_s": runs[0]["compile_s"],
                    "train_s": runs[0]["train_s"],
                }
            )
    return rows


def style(ax: Any) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.grid(axis="y", color="#e6e5e1", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_scaling(rows: list[dict[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for condition in CONDITIONS:
        sub = [r for r in rows if r["condition"] == condition]
        xs = [r["actuators"] for r in sub]
        ys = [r["return_mean"] for r in sub]
        ax.errorbar(
            xs,
            ys,
            yerr=[r["return_std"] for r in sub],
            color=COLORS[condition],
            marker=MARKERS[condition],
            markersize=7,
            linewidth=2,
            capsize=3,
            label=condition,
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks([4 * n for n in SIZES], [str(4 * n) for n in SIZES])
    ax.set_xlabel("actuators", color=INK)
    ax.set_ylabel("eval return, mean ± std over seeds", color=INK)
    style(ax)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def binned_curve(runs: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    total = runs[0]["config"]["total_steps"]
    edges = np.linspace(0, total, CURVE_BINS + 1)
    sums, counts = np.zeros(CURVE_BINS), np.zeros(CURVE_BINS)
    for r in runs:
        steps, rets = np.array(r["learning_curve"]).reshape(-1, 2).T
        bins = np.clip(np.digitize(steps, edges) - 1, 0, CURVE_BINS - 1)
        np.add.at(sums, bins, rets)
        np.add.at(counts, bins, 1)
    centers = (edges[:-1] + edges[1:]) / 2
    keep = counts > 0
    return centers[keep], sums[keep] / counts[keep]


def plot_curves(by_program: Results, path: Path) -> None:
    fig, axes = plt.subplots(
        1, len(SIZES), figsize=(14, 3.6), sharex=True, squeeze=False
    )
    for ax, n_segs in zip(axes[0], SIZES, strict=True):
        for condition in CONDITIONS:
            xs, ys = binned_curve(by_program[n_segs, condition])
            ax.plot(xs, ys, color=COLORS[condition], linewidth=2)
        ax.set_title(f"{4 * n_segs} actuators", color=INK, size=10)
        ax.set_xlabel("env steps", color=INK)
        style(ax)
    axes[0][0].set_ylabel("train episode return, mean over seeds", color=INK)
    handles = [
        plt.Line2D([], [], color=COLORS[c], linewidth=2) for c in CONDITIONS
    ]
    fig.legend(handles, CONDITIONS, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(report: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    print("n_segs act  condition   K  return            x_vel   steps/s")
    for r in rows:
        print(
            f"{r['n_segs']:6d} {r['actuators']:3d}  {r['condition']:8s} "
            f"{r['k']:3d}  {r['return_mean']:7.1f} ± {r['return_std']:6.1f}"
            f"  {r['x_velocity_mean']:6.2f}  {r['steps_per_s']:8.0f}"
        )
    print("\nn_segs act  split     ratio   p        p_holm")
    for c in report["comparisons"]:
        print(
            f"{c['n_segs']:6d} {c['actuators']:3d}  {c['condition']:8s} "
            f"{c['ratio']:5.2f}  {c['p']:.4f}   {c['p_holm']:.4f}"
            + ("  *" if c["passes"] else "")
        )
    print("\nGO" if report["go"] else "\nNO-GO")


def main() -> None:
    parser = argparse.ArgumentParser(description="stage 0 GO / NO-GO")
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    out = args.results
    by_program = load_results(args.results)
    report = verdict(by_program)
    rows = program_table(by_program)
    (out / "verdict.json").write_text(json.dumps(report, indent=2) + "\n")
    write_csv(rows, out / "stage0.csv")
    plot_scaling(rows, out / "scaling.png")
    plot_curves(by_program, out / "curves.png")
    print_summary(report, rows)


if __name__ == "__main__":
    main()
