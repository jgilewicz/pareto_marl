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
