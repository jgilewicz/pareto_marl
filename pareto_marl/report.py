import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from pareto_marl.analysis import (
    ICC_THRESHOLD,
    P_THRESHOLD,
    cleanrl_check,
    load_results,
    partition_table,
    verdict,
)
from pareto_marl.design import DESIGN_PATH, load_design

TEMPLATE = Path(__file__).with_name("report_template.html")
CURVE_BIN = 20_480


def binned_curve(curve: list[list[float]], total_steps: int) -> np.ndarray:
    n_bins = total_steps // CURVE_BIN
    sums, counts = np.zeros(n_bins), np.zeros(n_bins)
    for step, ret in curve:
        b = min(int(step // CURVE_BIN), n_bins - 1)
        sums[b] += ret
        counts[b] += 1
    with np.errstate(invalid="ignore"):
        return sums / counts


def group_curve(
    runs: list[dict[str, Any]], total_steps: int
) -> list[list[float]]:
    stacked = np.vstack(
        [binned_curve(r["learning_curve"], total_steps) for r in runs]
    )
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(stacked, axis=0)
    steps = (np.arange(len(mean)) + 1) * CURVE_BIN
    return [
        [int(s), round(float(m), 1)]
        for s, m in zip(steps, mean, strict=True)
        if np.isfinite(m)
    ]


def curves(
    mappo: list[dict[str, Any]], cleanrl: list[dict[str, Any]], steps: int
) -> dict[str, list[list[float]]]:
    out = {
        f"MAPPO K={k}": group_curve([r for r in mappo if r["k"] == k], steps)
        for k in sorted({r["k"] for r in mappo})
    }
    if cleanrl:
        out["CleanRL PPO"] = group_curve(cleanrl, steps)
    return out


def partition_rows(
    mappo: list[dict[str, Any]], design: dict[str, Any]
) -> list[dict[str, Any]]:
    rows = partition_table(mappo, design)
    for row in rows:
        runs = sorted(
            (r for r in mappo if r["x"] == row["x"]), key=lambda r: r["seed"]
        )
        row["returns"] = [round(r["eval_return"], 1) for r in runs]
        row["velocities"] = [round(r["eval_x_velocity"], 3) for r in runs]
    return rows


def build_data(
    results_dir: Path, design: dict[str, Any], cpu_hours: float | None
) -> dict[str, Any]:
    mappo = load_results(results_dir, "mappo")
    cleanrl = load_results(results_dir, "cleanrl")
    steps = mappo[0]["config"]["total_steps"]
    expected = len(design["partitions"]) * len(design["seeds"])
    return {
        "meta": {
            "run_id": mappo[0].get("run_id", results_dir.name),
            "total_steps": steps,
            "n_mappo": len(mappo),
            "n_mappo_expected": expected,
            "n_cleanrl": len(cleanrl),
            "n_seeds": len(design["seeds"]),
            "cpu_hours": cpu_hours,
            "generated": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        },
        "rule": {"p": P_THRESHOLD, "icc": ICC_THRESHOLD},
        "verdict": verdict(mappo),
        "cleanrl_check": cleanrl_check(mappo, cleanrl),
        "partitions": partition_rows(mappo, design),
        "curves": curves(mappo, cleanrl, steps),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="write the HTML report")
    parser.add_argument("results", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--design", type=Path, default=DESIGN_PATH)
    parser.add_argument("--cpu-hours", type=float, default=None)
    args = parser.parse_args()
    data = build_data(args.results, load_design(args.design), args.cpu_hours)
    html = TEMPLATE.read_text().replace(
        "/*__DATA__*/null", json.dumps(data, separators=(",", ":"))
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html)
    print(f"wrote {args.out} ({'GO' if data['verdict']['go'] else 'NO-GO'})")


if __name__ == "__main__":
    main()
