# Public scenario and result formats

The included `data/sample/scenario.csv` is independently generated synthetic data. Run `python -m stockyard.data.sample --seed 42` to recreate it.

## Scenario CSV

| Column | Type | Meaning |
|---|---|---|
| `pileno` | Nonempty string | Source pile identifier |
| `pileseq` | Positive integer | Bottom-to-top order within a source; unique within that source |
| `markno` | Unique nonempty string | Plate identifier |
| `unitw` | Finite nonnegative number | Plate weight field; no weight constraint is enforced |
| `inbound` | Nonnegative integer | Arrival day |
| `outbound` | Integer, at least `inbound` | Retrieval priority expressed as outbound day |
| `topile` | Nonempty string | A member of the available destination pool |

`topile` values define the destination pool, not compulsory per-plate assignments. The model supports at most 30 distinct source and 30 distinct destination identifiers. The loader preserves leading zeroes in identifiers, validates fields before sorting, and sorts by source and pile sequence.

For a static scenario, all arrival days are equal. Multiple source piles must each fit `--max-stack`; the retained single-source barge mode allows a larger source buffer. All plates must fit the total destination capacity. Dynamic arrivals follow arrival time, preserving pile sequence among plates arriving at the same time; new arrivals are stacked above plates already present.

Extra experimental metadata such as `max_stack` and `obs_max` is excluded from the public input passed to solvers. Set capacity with the benchmark CLI. A nonempty `confirm_time` is rejected because the retained environment does not apply scheduled outbound-date confirmations. The public sample uses fixed outbound priority per plate.

## Benchmark CSV

| Column | Meaning |
|---|---|
| `method` | Solver or policy name |
| `blocking_pairs` | Final destination interference; failed runs have an infinite or missing value |
| `moves` | Actual completed plate transfers; missing for skipped/no-solution runs |
| `runtime_seconds` | Method runtime; timing boundaries are described in `METHODOLOGY.md` |
| `status` | Yard termination or solver status; Gurobi includes its optimality gap |
| `completed` | Whether all input plates were transferred successfully |
| `seed` | Evaluation seed |
| `max_stack` | Shared stack capacity |
| `plate_count` | Input plate count |

The plotting command displays only completed, finite results. Static-only SA, ACO and Gurobi are marked as skipped when given a dynamic arrival scenario. A short PPO demonstration checkpoint is an execution example, not a trained thesis model.
