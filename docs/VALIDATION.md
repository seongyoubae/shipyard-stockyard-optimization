# Validation record

Final review: 2026-10-05. Validated locally on Python 3.12 in a CPU environment. Hosted CI results should be checked separately in the repository's Actions tab.

## Environment

| Package | Version |
|---|---|
| numpy | 2.3.5 |
| pandas | 2.2.3 |
| torch | 2.14.1+cpu |
| matplotlib | 3.10.8 |
| openpyxl | 3.1.5 |
| tensorboard | 2.21.0 |
| pytest | 9.1.1 |
| gurobipy | 13.0.3 |
| ruff | 0.16.10 |

## Automated checks

**48 tests passed**, including:

- Blocking pair order and equal-date ties; top-only moves, capacity and padding masks.
- Dynamic arrivals, invalid actions, episode step limits and reset overflow handling.
- Terminal GAE and model/evaluation integration.
- Invalid CSV fields, duplicate IDs/sequences, nonfinite values, fractional dates, empty identifiers and unsupported scheduled confirmations.
- String ID preservation, row ordering, and exclusion of experimental capacity/obstacle overrides.
- Deterministic SA repair and plan replay; incomplete plans receive no finite objective.
- Forward execution, feasible actions and matching rollout/evaluation likelihoods for 20 encoder names and three compatibility aliases.
- Unknown encoder rejection and research checkpoint boolean CLI flags.
- Original heuristic evaluation helper completion/failure reporting and legacy annealer time tracking.
- Single-source barge buffer behavior.

Ruff's configured Python error checks and formatting checks passed. Package compilation passed. A wheel was built, installed into a separate temporary directory and imported successfully outside the source checkout.

## Execution checks

The README quick-start sequence was run end to end:

```bash
python -m stockyard.data.sample --seed 42
python -m stockyard.training.ppo --updates 2 --horizon 32
python -m stockyard.evaluation.benchmark --methods random edd-mod sop-mfb sa aco ppo
python -m stockyard.analysis.plot
```

- The static sample contains 24 plates. All six methods completed exactly 24 transfers and reported finite final objectives.
- PPO completed two updates with finite losses and saved a checkpoint that the benchmark loaded.
- Dynamic sample evaluation completed for Random, EDD–MOD, SOP–MFB and PPO. SA, ACO and Gurobi were marked as skipped; dynamic result plotting passed.
- Gurobi solved the four-plate static sample with objective zero, status Optimal and gap zero. This checks a tiny instance only.
- Both static and dynamic SVG output files were generated.
- Original research training completed one CPU epoch with one environment, four rollout steps, one PPO optimization pass and embedding dimension 32. Its final evaluation used a small 2-source / 2-destination synthetic scenario in the review harness instead of the full anchor grid. A checkpoint was saved. This validates the training/evaluation wiring, not the full research experiment.

## Review changes

The final review repaired stochastic SA candidate rescoring, partial-transfer reporting, undefined names in retained helpers, ambiguous checkpoint booleans, silent encoder-name fallbacks and reset overflow handling. Public input validation now runs before numeric conversion and sorting. Evaluation records completion and scenario settings, and plots exclude failed/skipped rows. Python files share a consistent format; dependencies and generated-file exclusions were checked.

The README and methodology now describe the retained environment's actual date behavior: dynamic arrivals are implemented, scheduled outbound-date confirmations and physical outbound dispatch are not. Public source files and the included synthetic CSV were scanned for industrial records, personal details, credentials and local absolute paths.

## Interpretation and limits

A two-update PPO checkpoint is an execution example. These checks do not establish thesis performance, statistical superiority, GPU behavior or large-scale training reproducibility. No thesis performance is inferred from the uploaded header-only training log. The retained TorchScript helper emits a deprecation warning on this runtime without failing the checks. Larger Gurobi models require a suitable license. Dependency ranges describe supported installation bounds; the table above records the environment actually exercised.
