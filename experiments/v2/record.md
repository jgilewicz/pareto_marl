# v2: does the actuator partition matter? (Ant-v5 without contact forces)

- Same question, design partitions and GO / NO-GO rule as v1 (`plan.md` §3, §12)
- Changes: obs 27 (no contact forces), T = 3M, seeds 0–9
- Account: hpc-danbor2008-1756464546, `bem2-cpu-short`, 1 core, 4 GB, 4 h per task
- Results: `/lustre/pd03/hpc-danbor2008-1756464546/pareto_marl/results/v2/`
- wandb: `marl-partition-moo`, group `v2`
- Estimate: 240 tasks, ~260 CPU-h

## Jobs

| job | what | tasks | submitted |
| --- | --- | --- | --- |
| 6060207 | MAPPO | 0–229 | 2026-10-09 10:16 |
| 6060208 | CleanRL reference | 0–9 | 2026-10-09 10:16 |
| 6063331 | CleanRL resubmit (wandb init timeout on r30c01b01 in 6060208_6-9) | 6–9 | 2026-10-09 15:31 |
| 6063369 | CleanRL resubmit 2: `--exclude=r30c01b01`, `WANDB_INIT_TIMEOUT=300` (6063331_6-9 failed the same way on r30c01b01) | 6–9 | 2026-10-09 15:36 |
| 6063393 | CleanRL resubmit 3: `WANDB_MODE=offline` (6063369 timed out creating wandb runs on another node too; existing runs kept syncing) | 6–9 | 2026-10-09 15:43 |

## Results (2026-10-09)

- 240 / 240 runs present (230 MAPPO + 10 CleanRL), 10:17 → 17:04, **274.9 CPU-h**
  (incl. 8 failed CleanRL attempts: wandb run creation timed out)
- **NO-GO**: K=2 F=0.85, p=0.59, ICC=−0.02; K=4 F=0.80, p=0.60, ICC=−0.02
- Sanity: MAPPO K=1 vs CleanRL, last 10% train return 2744 ± 887 vs
  2234 ± 935, Welch p=0.25
- Mean eval return by K: K=1 2831, K=2 1939, K=4 1227, K=8 495
  (v1: 593, 496, 427, 404); forward v 2.76 / ~1.8 / ~1.1 / 0.42 m/s
- Report: https://claude.ai/code/artifact/916cb51e-40b3-4e5c-bbd8-3bdd7be1c03d
