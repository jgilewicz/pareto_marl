import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import wandb

from pareto_marl.design import DESIGN_PATH, load_design, task
from pareto_marl.mappo import PPOConfig, train_mappo
from pareto_marl.partition import PartitionSpec

WANDB_PROJECT = "marl-partition-moo"


def run_task(
    index: int, out_dir: Path, run_id: str, cfg: PPOConfig, design_path: Path
) -> Path:
    x, seed = task(load_design(design_path), index)
    spec = PartitionSpec.build(x)
    name = f"mappo-{''.join(map(str, x))}-s{seed}"
    run = wandb.init(
        project=WANDB_PROJECT,
        group=run_id,
        job_type="mappo",
        name=name,
        config={
            "x": "".join(map(str, x)),
            "k": spec.k,
            "seed": seed,
            "task_index": index,
            "obs_dims": spec.obs_dims,
            "act_dims": spec.act_dims,
            "actor_params": spec.actor_params,
            **asdict(cfg),
        },
        reinit="create_new",
    )

    def log(metrics: dict[str, float], step: int) -> None:
        run.log({**metrics, "global_step": step})

    try:
        result = train_mappo(spec, cfg, seed, log)
        run.summary.update(
            {k: v for k, v in result.items() if k.startswith("eval_")}
        )
    finally:
        run.finish()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps({**result, "run_id": run_id}) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="train one design task")
    parser.add_argument(
        "--index",
        type=int,
        default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "-1")),
        help="task index in the design (default: $SLURM_ARRAY_TASK_ID)",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--design", type=Path, default=DESIGN_PATH)
    args = parser.parse_args()
    cfg = PPOConfig(total_steps=args.total_steps)
    path = run_task(args.index, args.out, args.run_id, cfg, args.design)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
