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
