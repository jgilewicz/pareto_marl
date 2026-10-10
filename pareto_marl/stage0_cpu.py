import argparse
import contextlib
import io
import json
import os
import platform
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import wandb

from pareto_marl.design import (
    CONDITIONS,
    SEEDS,
    SIZES,
    TOTAL_STEPS,
    git_commit,
    load_program,
)
from pareto_marl.mappo_torch import Agents, MAPPOTrainer, PPOConfig

# package prints an Adroit reward-version notice on import; Adroit is unused
with contextlib.redirect_stderr(io.StringIO()):
    from gymnasium_robotics.envs.multiagent_mujoco.many_segment_ant import (
        gen_asset,
    )

WANDB_PROJECT = "marl-partition-moo"
WANDB_JOB_TYPE = "stage0-cpu"
WANDB_INIT_TIMEOUT_S = 300
N_TASKS = len(SIZES) * len(CONDITIONS) * len(SEEDS)
PROGRESS_LINES = 20
HEAD_KEYS = (
    "n_segs",
    "actuators",
    "condition",
    "partition_key",
    "k",
    "actor_params",
)


def task(index: int) -> tuple[int, int]:
    if not 0 <= index < N_TASKS:
        raise IndexError(
            f"task index {index} out of range 0..{N_TASKS - 1} "
            f"(program = index // {len(SEEDS)}, seed = index % {len(SEEDS)}); "
            "pass --index or submit as a SLURM array"
        )
    return index // len(SEEDS), SEEDS[index % len(SEEDS)]


def env_kwargs(n_segs: int, directory: Path) -> dict[str, Any]:
    # own temp dir: concurrent tasks share the venv, so never write the XML
    # next to MaMuJoCo's assets in site-packages
    xml = directory / f"many_segment_ant_{n_segs}.xml"
    gen_asset(n_segs, str(xml))
    return {"xml_file": str(xml), "include_cfrc_ext_in_observation": False}


def agents_of(program: dict[str, Any]) -> Agents:
    return Agents(
        obs_idx=[np.asarray(i) for i in program["obs_idx"]],
        act_idx=[np.asarray(i) for i in program["act_idx"]],
    )


class CurveLog:
    def __init__(self, cfg: PPOConfig) -> None:
        self.curve: list[list[float]] = []
        self.every = max(1, cfg.num_iterations // PROGRESS_LINES)
        self.batch_size = cfg.batch_size

    def __call__(self, metrics: dict[str, float], step: int) -> None:
        ret = metrics.get("train/episodic_return")
        if ret is not None:
            self.curve.append([step, ret])
        if (step // self.batch_size) % self.every == 0:
            last = f"{self.curve[-1][1]:.1f}" if self.curve else "-"
            print(
                f"step {step} sps {metrics['perf/sps']:.0f} "
                f"train return {last}",
                flush=True,
            )


def train(program: dict[str, Any], seed: int, cfg: PPOConfig) -> dict[str, Any]:
    torch.set_num_threads(1)
    log = CurveLog(cfg)
    with tempfile.TemporaryDirectory(prefix="stage0cpu-") as tmp:
        trainer = MAPPOTrainer(
            agents_of(program),
            env_kwargs(program["n_segs"], Path(tmp)),
            cfg,
            seed,
        )
        try:
            start = time.perf_counter()
            trainer.train(log)
            train_s = time.perf_counter() - start
            evaluation = trainer.evaluate()
        finally:
            trainer.close()
    return {
        **evaluation,
        "learning_curve": log.curve,
        "steps_per_s": trainer.global_step / train_s,
        "compile_s": 0.0,
        "train_s": train_s,
    }


def result_name(result: dict[str, Any], prefix: str) -> str:
    return (
        f"{prefix}-n{result['n_segs']}-{result['condition']}-s{result['seed']}"
    )


def run(index: int, out_dir: Path, run_id: str, total_steps: int) -> Path:
    start = time.perf_counter()
    commit = git_commit()
    program_index, seed = task(index)
    program = load_program(program_index)
    cfg = PPOConfig(total_steps=total_steps)
    print(
        f"task {index}: program {program_index} n_segs={program['n_segs']} "
        f"{program['condition']} K={program['k']} seed={seed} "
        f"T={total_steps} ({cfg.num_iterations} iterations)",
        flush=True,
    )
    trained = train(program, seed, cfg)
    result = {
        **{k: program[k] for k in HEAD_KEYS},
        "seed": seed,
        **trained,
        "wall_s": time.perf_counter() - start,
        "device": f"cpu {platform.machine()} {platform.node()}",
        "run_id": run_id,
        "git_commit": commit,
        "config": asdict(cfg),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    # analysis.py globs stage0-*.json
    path = out_dir / f"{result_name(result, 'stage0')}.json"
    path.write_text(json.dumps(result) + "\n")
    print(
        f"wrote {path}: eval_return {result['eval_return']:.1f}, "
        f"{result['steps_per_s']:.0f} steps/s",
        flush=True,
    )
    return path


def log_wandb(result: dict[str, Any]) -> None:
    run = wandb.init(
        project=WANDB_PROJECT,
        group=result["run_id"],
        job_type=WANDB_JOB_TYPE,
        name=result_name(result, "stage0cpu"),
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
        # on disk, keep going and report at the end
        except Exception as e:
            failed += 1
            print(
                f"wandb logging failed for {path}: {e!r}; the JSON is "
                "complete, retry with `python -m pareto_marl.stage0_cpu log`",
                file=sys.stderr,
            )
    return failed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="stage 0 on CPU: one program x seed per process"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run", help="train one task, write JSON, log")
    run_p.add_argument(
        "--index",
        type=int,
        default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "-1")),
        help=f"task 0..{N_TASKS - 1}: program index // {len(SEEDS)}, "
        f"seed index % {len(SEEDS)} (default: $SLURM_ARRAY_TASK_ID)",
    )
    run_p.add_argument("--total-steps", type=int, default=TOTAL_STEPS)
    run_p.add_argument("--out", type=Path, required=True)
    run_p.add_argument("--run-id", required=True)
    log = sub.add_parser("log", help="log result JSONs to wandb")
    log.add_argument("paths", type=Path, nargs="+")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    match args.command:
        case "run":
            path = run(args.index, args.out, args.run_id, args.total_steps)
            sys.exit(1 if log_all([path]) else 0)
        case "log":
            sys.exit(1 if log_all(args.paths) else 0)


if __name__ == "__main__":
    main()
