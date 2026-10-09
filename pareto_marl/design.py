from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
INDICES_PATH = REPO / "configs" / "stage0_indices.json"
# plan_stage0.md §3: program index = 4 * size index + condition index
SIZES = (2, 4, 8, 16)
CONDITIONS = ("single", "segment", "leg", "joint")
# fixed per program: a seed's result depends on the vmap batch it runs in
SEEDS = (0, 1, 2, 3, 4)
TOTAL_STEPS = 3_000_000
