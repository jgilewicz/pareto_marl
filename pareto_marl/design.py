import json
import subprocess
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
INDICES_PATH = REPO / "configs" / "stage0_indices.json"
# plan_stage0.md §3: program index = 4 * size index + condition index
SIZES = (2, 4, 8, 16)
CONDITIONS = ("single", "segment", "leg", "joint")
# fixed per program: a seed's result depends on the vmap batch it runs in
SEEDS = (0, 1, 2, 3, 4)
TOTAL_STEPS = 3_000_000


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


def git_commit() -> str:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.strip()

    commit = git("rev-parse", "HEAD")
    return commit + ("-dirty" if git("status", "--porcelain") else "")
