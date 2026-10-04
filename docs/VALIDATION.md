# Local validation

Validated on Python 3.12 in a CPU execution environment. Commands below were run locally; hosted GitHub Actions status is not claimed.

## Environment

| Package | Version |
|---|---|
| numpy | 2.3.5 |
| pandas | 2.2.3 |
| torch | 2.14.1 |
| matplotlib | 3.10.8 |
| openpyxl | 3.1.5 |
| tensorboard | 2.21.0 |
| pytest | 9.1.1 |
| gurobipy | 13.0.3 |

## Checks

- Six tests passed: blocking pair order/ties; top-only moves/capacity/padding masks; dynamic arrivals; invalid actions/step cap; terminal GAE; model and evaluation adapter.
- Package compilation passed.
- Public sample generation passed (24 plates).
- CPU PPO completed two updates with finite losses and saved a loadable checkpoint.
- Random, EDD–MOD, SOP–MFB, SA, ACO and PPO completed the common static scenario with 24 moves each.
- Dynamic sample generation and Random/EDD–MOD/PPO evaluation completed; static-only baselines were explicitly skipped.
- Gurobi solved a four-plate static sample to objective zero, status Optimal, gap zero. This validates a tiny instance only.
- Benchmark SVG rendering passed.
- Original research trainer help/import check passed. Full large-scale training was not rerun.

## Results interpretation

The original uploaded training log contains only a header, so no thesis performance claim is derived from it. The two-update PPO smoke run was weaker than some baselines; the repository makes no performance superiority claim. Demo objective values are intentionally not presented as validated research results.

## Limitations

The retained TorchScript helper emits a deprecation warning on this runtime. It did not fail the checks. Large Gurobi models require a suitable license. GPU training and the full experiment grid have not been validated. Dependency ranges support installation; this table records the environment actually checked.
