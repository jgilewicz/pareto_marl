# v1 — does the actuator partition matter? (MaMuJoCo Ant)

- Question and GO / NO-GO rule: `plan.md` §3 (ANOVA p<0.05 and ICC≥0.2 for K=2 or K=4)
- Design: `configs/design.json`, 23 partitions × seeds 0–4, T = 1M
- Commit: a3d54a7 (branch `wcss-setup`)
- Account: hpc-danbor2008-1756464546, `bem2-cpu-short`, 1 core, 4 GB, 2 h per task
- Results: `/lustre/pd03/hpc-danbor2008-1756464546/pareto_marl/results/v1/`
- wandb: `marl-partition-moo`, group `v1`

## Jobs

| job | what | tasks | submitted |
| --- | --- | --- | --- |
| 6058676 | MAPPO | 0–114 | 2026-10-09 02:24 |
| 6058677 | CleanRL reference | 0–4 | 2026-10-09 02:24 |

## Results (2026-10-09)

- All 120 tasks COMPLETED, 02:25 → 03:40, 17–33 min per task, **43.8 CPU-h**
- **NO-GO**: K=2 F=1.70, p=0.10, ICC=0.12; K=4 F=1.07, p=0.41, ICC=0.01
- Sanity: MAPPO K=1 vs CleanRL, last 10% train return 447 ± 99 vs 452 ± 36,
  Welch p=0.92
- Mean eval return by K: K=1 593, K=2 496, K=4 427, K=8 404
- Caveat: low-return regime (eval ~300–770, forward v ~0.3–0.9 m/s), CleanRL
  itself reaches ~450 train return at 1M on Ant-v5 with 8×256
- Report: https://claude.ai/code/artifact/a65fed4c-9f42-4eb8-8a2d-8b0b6bfe7a35
