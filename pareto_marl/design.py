import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from pareto_marl.partition import (
    Partition,
    n_blocks,
    parse_key,
    sample_partitions,
)

DESIGN_PATH = Path("configs/design.json")
N_SEEDS = 10
N_RANDOM_PER_K = 8
RANDOM_KS = (2, 4)

REFERENCE = {"00000000": "single agent", "01234567": "agent per joint"}
STRUCTURED = {
    "00001111": "front / back",
    "00111100": "left / right",
    "00110011": "diagonals",
    "01010101": "hips / ankles",
    "00112233": "agent per leg",
}


def _entry(x: Partition, group: str, label: str) -> dict[str, Any]:
    key = "".join(map(str, x))
    return {"x": key, "k": n_blocks(x), "group": group, "label": label}


def build_design(seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    entries = [
        _entry(parse_key(k), "reference", v) for k, v in REFERENCE.items()
    ]
    entries += [
        _entry(parse_key(k), "structured", v) for k, v in STRUCTURED.items()
    ]
    taken = {parse_key(e["x"]) for e in entries}
    for k in RANDOM_KS:
        for x in sample_partitions(k, N_RANDOM_PER_K, rng, exclude=taken):
            entries.append(_entry(x, "random", ""))
    entries.sort(key=lambda e: (e["k"], e["group"], e["x"]))
    return {
        "sample_seed": seed,
        "seeds": list(range(N_SEEDS)),
        "partitions": entries,
    }


def task(design: dict[str, Any], index: int) -> tuple[Partition, int]:
    seeds = design["seeds"]
    n_tasks = len(design["partitions"]) * len(seeds)
    if not 0 <= index < n_tasks:
        raise IndexError(
            f"task index {index} out of range: design has {n_tasks} tasks "
            f"(0..{n_tasks - 1})"
        )
    entry = design["partitions"][index // len(seeds)]
    return parse_key(entry["x"]), seeds[index % len(seeds)]


def load_design(path: Path = DESIGN_PATH) -> dict[str, Any]:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description="write the partition design")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=DESIGN_PATH)
    args = parser.parse_args()
    design = build_design(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(design, indent=2) + "\n")
    n_parts = len(design["partitions"])
    n_tasks = n_parts * len(design["seeds"])
    print(f"wrote {args.out}: {n_parts} partitions, {n_tasks} tasks")


if __name__ == "__main__":
    main()
