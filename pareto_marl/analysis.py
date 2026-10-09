import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from pareto_marl.design import DESIGN_PATH, load_design

TESTED_KS = (2, 4)
P_THRESHOLD = 0.05
ICC_THRESHOLD = 0.2
FINAL_FRACTION = 0.1


def load_results(results_dir: Path, prefix: str) -> list[dict[str, Any]]:
    paths = sorted(results_dir.glob(f"{prefix}-*.json"))
    if not paths:
        raise FileNotFoundError(
            f"no {prefix}-*.json in {results_dir}: check the path or copy "
            "the results from PD first"
        )
    return [json.loads(p.read_text()) for p in paths]


def icc1(groups: list[np.ndarray]) -> float:
    n = len(groups[0])
    if any(len(g) != n for g in groups):
        raise ValueError(
            f"ICC(1) needs equal seeds per partition, got "
            f"{sorted({len(g) for g in groups})}: rerun the missing tasks"
        )
    grand = np.mean(np.concatenate(groups))
    a = len(groups)
    msb = n * sum((g.mean() - grand) ** 2 for g in groups) / (a - 1)
    msw = sum(((g - g.mean()) ** 2).sum() for g in groups) / (a * (n - 1))
    return float((msb - msw) / (msb + (n - 1) * msw))


def returns_by_partition(
    results: list[dict[str, Any]], k: int
) -> dict[str, np.ndarray]:
    by_x: dict[str, list[float]] = {}
    for r in results:
        if r["k"] == k:
            by_x.setdefault(r["x"], []).append(r["eval_return"])
    return {x: np.array(v) for x, v in sorted(by_x.items())}


def verdict(results: list[dict[str, Any]]) -> dict[str, Any]:
    per_k = {}
    for k in TESTED_KS:
        groups = list(returns_by_partition(results, k).values())
        anova = stats.f_oneway(*groups)
        icc = icc1(groups)
        per_k[k] = {
            "n_partitions": len(groups),
            "p_value": float(anova.pvalue),
            "f_stat": float(anova.statistic),
            "icc": icc,
            "go": bool(anova.pvalue < P_THRESHOLD and icc >= ICC_THRESHOLD),
        }
    go = any(v["go"] for v in per_k.values())
    return {"go": go, "per_k": per_k}


def final_train_return(curve: list[list[float]], total_steps: int) -> float:
    start = total_steps * (1 - FINAL_FRACTION)
    tail = [r for step, r in curve if step >= start]
    return float(np.mean(tail))


def cleanrl_check(
    mappo: list[dict[str, Any]], cleanrl: list[dict[str, Any]]
) -> dict[str, Any]:
    single = [r for r in mappo if r["k"] == 1]
    steps = single[0]["config"]["total_steps"]
    ours = [final_train_return(r["learning_curve"], steps) for r in single]
    ref = [final_train_return(r["learning_curve"], steps) for r in cleanrl]
    test = stats.ttest_ind(ours, ref, equal_var=False)
    return {
        "mappo_k1": [float(np.mean(ours)), float(np.std(ours))],
        "cleanrl": [float(np.mean(ref)), float(np.std(ref))],
        "welch_p": float(test.pvalue),
    }


def partition_table(
    results: list[dict[str, Any]], design: dict[str, Any]
) -> list[dict[str, Any]]:
    rows = []
    for entry in design["partitions"]:
        runs = [r for r in results if r["x"] == entry["x"]]
        if not runs:
            continue
        ret = np.array([r["eval_return"] for r in runs])
        vel = np.array([r["eval_x_velocity"] for r in runs])
        rows.append(
            {
                **entry,
                "actor_params": runs[0]["actor_params"],
                "n_seeds": len(runs),
                "return_mean": float(ret.mean()),
                "return_std": float(ret.std(ddof=1)) if len(ret) > 1 else 0.0,
                "x_velocity_mean": float(vel.mean()),
            }
        )
    return rows


def plot(rows: list[dict[str, Any]], path: Path) -> None:
    colors = {"reference": "black", "structured": "tab:blue", "random": "0.6"}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for group, color in colors.items():
        sub = [r for r in rows if r["group"] == group]
        ax.errorbar(
            [r["actor_params"] for r in sub],
            [r["return_mean"] for r in sub],
            yerr=[r["return_std"] for r in sub],
            fmt="o",
            color=color,
            label=group,
            capsize=3,
        )
        for r in sub if group != "random" else []:
            ax.annotate(r["x"], (r["actor_params"], r["return_mean"]), size=7)
    ax.set_xlabel("actor parameters (f2)")
    ax.set_ylabel("eval return, mean ± std over seeds")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="GO / NO-GO analysis")
    parser.add_argument("results", type=Path)
    parser.add_argument("--design", type=Path, default=DESIGN_PATH)
    args = parser.parse_args()
    mappo = load_results(args.results, "mappo")
    rows = partition_table(mappo, load_design(args.design))
    report = verdict(mappo)
    report["cleanrl_check"] = cleanrl_check(
        mappo, load_results(args.results, "cleanrl")
    )
    with (args.results / "partitions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.results / "verdict.json").write_text(json.dumps(report, indent=2))
    plot(rows, args.results / "front.png")
    print(json.dumps(report, indent=2))
    print("GO" if report["go"] else "NO-GO")


if __name__ == "__main__":
    main()
