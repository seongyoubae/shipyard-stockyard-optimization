# Source provenance and public adaptation

This repository is based on the user's uploaded master's research code. It is a public demonstration with privacy cleanup and execution repairs, not a byte-for-byte thesis replication.

| Uploaded source | Public location / disposition |
|---|---|
| `data(2).py` | `stockyard/data/generator.py`: Plate and synthetic generation retained; empirical dwell-time narrative and distribution replaced by generic uniform sampling |
| `cfg(4).py` | `stockyard/config.py`: compatibility settings retained; public checkpoint path, CPU default and generic weight range |
| `env(4).py` | `stockyard/environment/yard.py`: original state, masks, movement, dynamic arrivals and rewards |
| `network(3).py` | `stockyard/models/network.py`: original encoders and actor/critic |
| `train(5).py` | `stockyard/training/research_train.py`: original training flow, package imports and non-personal seed; shared evaluation adapter repaired |
| `eval.network(1).py` | Single-scenario evaluation logic adapted to `stockyard/evaluation/policy.py`; hard-coded batch experiment runner excluded |
| `heuristic(2).py` | `stockyard/baselines/heuristic.py`: rules retained; hard-coded main runner excluded |
| `gurobi(2).py` | `stockyard/baselines/gurobi.py`: sequential MIP retained; dataset runner and import-time output writes excluded; solver warm-search time capped to requested budget |
| `SA(3).py` | `stockyard/baselines/sa.py`: action-sequence search retained; hard-coded runner excluded |
| `ACO(2).py` | `stockyard/baselines/aco.py`: heuristic-biased static ACO retained; hard-coded batch runner excluded |
| `SA_move(1).py` | `stockyard/baselines/legacy/sa_move.py`: earlier implementation retained for provenance |
| `SSY_SA(1).py` | `stockyard/baselines/legacy/annealer.py`: earlier annealer retained; standalone runner excluded |
| `training_log.csv` | Excluded: the upload contains only its header, with no observed performance to publish |

New public-facing additions: package structure, sample schema/generator, compact PPO trainer, common benchmark CLI, plotting CLI, synthetic visuals, documentation, dependency lists, tests and CI workflow. Do not present these additions as features independently verified in the original thesis.

No industrial raw dataset, industrial distribution quantiles, researcher contact information, hard-coded personal date seed, local machine path, license file or pretrained weight is included. Model and runtime outputs are ignored. The original uploads remain separate from this sanitized deliverable.

No open-source license grant is added automatically; research code ownership and institutional rights have not been independently established. A license can be chosen by the rights holder later.
