import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import numpy as np
import wandb

from pareto_marl.design import (
    CONDITIONS,
    INDICES_PATH,
    REPO,
    SEEDS,
    SIZES,
    TOTAL_STEPS,
)
from pareto_marl.envs import many_segment_ant
from pareto_marl.envs.contract import EnvSpec
from pareto_marl.networks.actor_critic import build_layout
from pareto_marl.partition import (
    SEGMENT_ACTUATORS,
    PartitionSpec,
    key_of,
    structured_partitions,
)
from pareto_marl.training.mappo import (
    PPOConfig,
    Setup,
    TrainOutput,
    benchmark,
    train_seeds,
)

WANDB_PROJECT = "marl-partition-moo"
WANDB_INIT_TIMEOUT_S = 300


def build_indices() -> list[dict[str, Any]]:
    programs = []
    for n_segs in SIZES:
        partitions = structured_partitions(n_segs)
        for condition in CONDITIONS:
            x = partitions[condition]
            spec = PartitionSpec.build(x)
            programs.append(
                {
                    "index": len(programs),
                    "n_segs": n_segs,
                    "actuators": SEGMENT_ACTUATORS * n_segs,
                    "condition": condition,
                    "partition_key": key_of(x),
                    "k": spec.k,
                    "actor_params": spec.actor_params,
                    "obs_idx": [idx.tolist() for idx in spec.obs_idx],
                    "act_idx": [idx.tolist() for idx in spec.act_idx],
                }
            )
    return programs


def write_indices(path: Path) -> None:
    text = json.dumps(build_indices(), indent=2)
    # one line per index list instead of one line per index
    text = re.sub(
        r"\[([\d\s,]+)\]",
        lambda m: "[" + ", ".join(v.strip() for v in m[1].split(",")) + "]",
        text,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n")
    print(f"wrote {path}")


def load_program(index: int, path: Path = INDICES_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing: run `just indices` locally and commit it "
            "(never on the cluster, MaMuJoCo writes into site-packages)"
        )
    programs = json.loads(path.read_text())
    if not 0 <= index < len(programs):
        raise IndexError(
            f"program index {index} out of range: {path} has "
            f"{len(programs)} programs (0..{len(programs) - 1}); "
            "pass --index or submit as a SLURM array"
        )
    return programs[index]


def make_setup(
    program: dict[str, Any], total_steps: int
) -> tuple[Setup, EnvSpec]:
    spec = many_segment_ant.make_spec(program["n_segs"])
    layout = build_layout(
        [np.asarray(i) for i in program["obs_idx"]],
        [np.asarray(i) for i in program["act_idx"]],
        spec.obs_dim,
        spec.action_dim,
    )
    setup = Setup(
        cfg=PPOConfig(total_steps=total_steps),
        layout=layout,
        reset=many_segment_ant.reset,
        step=many_segment_ant.step,
    )
    return setup, spec


def git_commit() -> str:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.strip()

    commit = git("rev-parse", "HEAD")
    return commit + ("-dirty" if git("status", "--porcelain") else "")


def learning_curve(
    out: TrainOutput, i: int, batch_size: int
) -> list[list[float]]:
    sums = np.asarray(out.curve_return_sum[i])
    counts = np.asarray(out.curve_episodes[i])
    return [
        [(it + 1) * batch_size, float(s / c)]
        for it, (s, c) in enumerate(zip(sums, counts, strict=True))
        if c > 0
    ]


def seed_results(
    program: dict[str, Any],
    cfg: PPOConfig,
    out: TrainOutput,
    meta: dict[str, Any],
) -> list[dict[str, Any]]:
    head = {
        k: program[k]
        for k in (
            "n_segs",
            "actuators",
            "condition",
            "partition_key",
            "k",
            "actor_params",
        )
    }
    return [
        {
            **head,
            "seed": seed,
            "seeds": list(SEEDS),
            "eval_return": float(out.eval_return[i]),
            "eval_return_std": float(out.eval_return_std[i]),
            "eval_x_velocity": float(out.eval_x_velocity[i]),
            "learning_curve": learning_curve(out, i, cfg.batch_size),
            **meta,
            "config": asdict(cfg),
        }
        for i, seed in enumerate(SEEDS)
    ]


def result_name(result: dict[str, Any]) -> str:
    return f"stage0-n{result['n_segs']}-{result['condition']}-s{result['seed']}"


def log_wandb(result: dict[str, Any]) -> None:
    run = wandb.init(
        project=WANDB_PROJECT,
        group=result["run_id"],
        job_type="stage0",
        name=result_name(result),
        config={
            k: v
            for k, v in result.items()
            if k not in ("learning_curve", "config")
            and not k.startswith("eval_")
        }
        | result["config"],
        reinit="create_new",
        settings=wandb.Settings(init_timeout=WANDB_INIT_TIMEOUT_S),
    )
    try:
        run.define_metric("train/episodic_return", step_metric="global_step")
        for step, ret in result["learning_curve"]:
            run.log({"train/episodic_return": ret, "global_step": step})
        run.summary.update(
            {k: v for k, v in result.items() if k.startswith("eval_")}
        )
    finally:
        run.finish()


def log_all(paths: list[Path]) -> int:
    failed = 0
    for path in paths:
        try:
            log_wandb(json.loads(path.read_text()))
        # any wandb error (init timeout, network, auth): the JSON is already
        # on disk, keep logging the other seeds and report at the end
        except Exception as e:
            failed += 1
            print(
                f"wandb logging failed for {path}: {e!r}; the JSON is "
                "complete, retry with `python -m pareto_marl.stage0 log`",
                file=sys.stderr,
            )
    return failed


def run(index: int, out_dir: Path, run_id: str, total_steps: int) -> int:
    start = time.perf_counter()
    commit = git_commit()
    program = load_program(index)
    setup, spec = make_setup(program, total_steps)
    print(
        f"program {index}: n_segs={program['n_segs']} "
        f"{program['condition']} K={program['k']} seeds={list(SEEDS)} "
        f"T={total_steps} on {jax.devices()[0]}",
        flush=True,
    )
    out, timing = train_seeds(setup, spec, SEEDS)
    meta = {
        "steps_per_s": timing["steps_per_s"],
        "compile_s": timing["compile_s"],
        "train_s": timing["run_s"],
        "wall_s": time.perf_counter() - start,
        "device": jax.devices()[0].device_kind,
        "run_id": run_id,
        "git_commit": commit,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for result in seed_results(program, setup.cfg, out, meta):
        path = out_dir / f"{result_name(result)}.json"
        path.write_text(json.dumps(result) + "\n")
        paths.append(path)
        print(
            f"wrote {path}: eval_return {result['eval_return']:.1f}",
            flush=True,
        )
    return log_all(paths)


def bench(index: int, iterations: int, total_steps: int) -> None:
    program = load_program(index)
    setup, spec = make_setup(program, total_steps)
    timing = benchmark(setup, spec, SEEDS, iterations)
    train_h = len(SEEDS) * total_steps / timing["steps_per_s"] / 3600
    report = {
        "index": index,
        "n_segs": program["n_segs"],
        "condition": program["condition"],
        "k": program["k"],
        "device": jax.devices()[0].device_kind,
        "seeds": len(SEEDS),
        "iterations": iterations,
        **timing,
        "estimated_hours": train_h + timing["compile_s"] / 3600,
    }
    print(json.dumps(report))


def env_index() -> int:
    return int(os.environ.get("SLURM_ARRAY_TASK_ID", "-1"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="stage 0 (plan_stage0.md)")
    sub = parser.add_subparsers(dest="command", required=True)
    indices = sub.add_parser("indices", help="write the agent indices JSON")
    indices.add_argument("--out", type=Path, default=INDICES_PATH)
    for name in ("run", "bench"):
        p = sub.add_parser(name, help=f"{name} one program (5 seeds)")
        p.add_argument(
            "--index",
            type=int,
            default=env_index(),
            help="program 0..15 (default: $SLURM_ARRAY_TASK_ID)",
        )
        p.add_argument("--total-steps", type=int, default=TOTAL_STEPS)
    run_p = sub.choices["run"]
    run_p.add_argument("--out", type=Path, required=True)
    run_p.add_argument("--run-id", required=True)
    sub.choices["bench"].add_argument("--iterations", type=int, default=3)
    log = sub.add_parser("log", help="log result JSONs to wandb")
    log.add_argument("paths", type=Path, nargs="+")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    match args.command:
        case "indices":
            write_indices(args.out)
        case "run":
            failed = run(args.index, args.out, args.run_id, args.total_steps)
            sys.exit(1 if failed else 0)
        case "bench":
            bench(args.index, args.iterations, args.total_steps)
        case "log":
            sys.exit(1 if log_all(args.paths) else 0)


if __name__ == "__main__":
    main()
