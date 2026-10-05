**English** · [한국어](README.ko.md)

<div align="center">

# Shipyard Stockyard Optimization

### Reinforcement learning to reduce retrieval interference during steel plate rehandling

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![PPO](https://img.shields.io/badge/Reinforcement%20Learning-PPO-2563EB?style=flat-square)

Research code and runnable examples from a master's project on steel plate stockyard optimization

</div>

## Overview

Shipyards store steel plates in stacks and retrieve them according to production schedules. When an earlier-due plate lies below a later-due plate, the upper plate must be moved before the target can be retrieved. This **retrieval interference** depends on how the plates are stacked.

The project optimizes **which top plate to move and where to place it** while transferring plates from source piles to destination piles. A stockyard environment represents the handling constraints, and a priority-aware GRU with PPO learns a relocation policy. Evaluation code supports comparisons with rule-based methods, SA, ACO and a Gurobi MIP.

![Example of steel plate retrieval interference](docs/stockyard.svg)

*An earlier-due plate buried beneath other plates, compared with a target that can be retrieved directly from the top.*

## Problem Definition

Each pile stores plates in bottom-to-top order. A crane selects the top plate of a source pile and places it on a destination pile. Repeating this process transfers the source inventory.

| Element | Setting |
| :--- | :--- |
| Decisions | Which source pile's top plate to move, and which destination pile to use |
| Handling constraint | Only the top plate can be moved |
| Capacity constraint | Respect the destination pile's maximum stack height |
| Objective | Reduce retrieval interference in destination piles after relocation |
| Main metric | Blocking pairs: a lower plate is due earlier than a plate above it |

Sorting plates by outbound date alone is insufficient. Source piles must be unloaded from the top, and every placement changes the destination stack seen by the next decision. The policy therefore needs to consider the remaining inventory and destination state as well as the moves currently available.

## Method

### Stockyard environment

The environment tracks source and destination piles together with plate arrival and outbound information. The state includes top-plate outbound dates, summaries of deeper layers, scheduled arrivals, time and interference information.

Action masks exclude empty source piles and destination piles at their height limit. In dynamic-arrival scenarios, new plates become available over time; while waiting for inventory, the environment advances to the next arrival.

### Priority-aware GRU

Outbound dates provide the plates' handling priorities. The model combines pile-state embeddings with representations of top-plate outbound priority, then uses a GRU to process information across piles.

The actor selects a source and a destination, while the critic estimates state value. The code also includes GRU, LSTM, MLP and attention encoder variants, along with alternative priority-fusion configurations for encoder comparisons.

### PPO training

Rewards account for changes in interference, newly created interference and differences in outbound dates. Training uses PPO's clipped objective and GAE, together with value-function loss and an entropy term.

Two training entry points are provided:

| Code | Purpose |
| :--- | :--- |
| [Research training](stockyard/training/research_train.py) | Original parallel rollout, training and scenario-evaluation workflow |
| [CPU example](stockyard/training/ppo.py) | Small-scale execution using the same environment and network |

State construction, reward equations and model behavior are described in the [methodology document](docs/METHODOLOGY.md).

## Baselines & Evaluation

The following methods share public inputs and evaluation metrics:

| Method | Implementation |
| :--- | :--- |
| Random | Random selection of feasible source and destination piles |
| EDD–MOD / SOP–MFB | Rule-based selection using outbound priority and interference |
| PPO + Priority-aware GRU | Source and destination selection using a trained policy |
| Simulated Annealing | Search over modified sequences of relocation actions |
| Ant Colony Optimization | Search using a heuristic initial solution and pheromone updates |
| Gurobi MIP | Mathematical optimization of source handling order and destination stacking order |

Evaluation records **final blocking pairs, move count, runtime and termination status**, together with completion, seed, stack height and plate count. Plots include completed runs only. Gurobi results also expose solver status and optimality gap. The public SA, ACO and Gurobi benchmarks operate on static scenarios.

Blocking pairs measure interference in the retrieval order. They are not interpreted as the exact number of additional crane moves.

## Quick Start

Use Python 3.10 or later and run the commands from the repository root.

```bash
git clone https://github.com/seongyoubae/shipyard-stockyard-optimization.git
cd shipyard-stockyard-optimization

python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt

# 1. Generate a synthetic scenario
python -m stockyard.data.sample --seed 42

# 2. Run a small PPO training example
python -m stockyard.training.ppo --updates 2 --horizon 32

# 3. Evaluate the policy and baselines
python -m stockyard.evaluation.benchmark --methods random edd-mod sop-mfb sa aco ppo

# 4. Visualize the results
python -m stockyard.analysis.plot
```

The default example uses 24 plates, four source piles and four destination piles. Two training updates check that the example runs; they do not establish the performance of a sufficiently trained policy.

| Generated file | Contents |
| :--- | :--- |
| `outputs/policy.pt` | Trained model checkpoint |
| `outputs/training.csv` | Training log |
| `outputs/benchmark.csv` | Evaluation results by method |
| `outputs/benchmark.svg` | Evaluation plot |

<details>
<summary><strong>Dynamic-arrival example</strong></summary>

```bash
python -m stockyard.data.sample --dynamic --output outputs/dynamic.csv
python -m stockyard.evaluation.benchmark --data outputs/dynamic.csv --methods random edd-mod ppo
```

</details>

<details>
<summary><strong>Optional Gurobi baseline</strong></summary>

```bash
python -m pip install -r requirements-gurobi.txt
python -m stockyard.data.sample --sources 2 --destinations 2 --plates-per-source 2 --output outputs/tiny.csv
python -m stockyard.evaluation.benchmark --data outputs/tiny.csv --methods gurobi --budget-seconds 2
```

A valid Gurobi license is required. The permitted model size depends on the license.

</details>

## Code Structure

| Path | Role |
| :--- | :--- |
| [stockyard/environment/](stockyard/environment/) | Stockyard state, handling constraints, action masks, rewards and termination |
| [stockyard/models/](stockyard/models/) | GRU and encoder variants, actor–critic |
| [stockyard/training/](stockyard/training/) | Research training and the CPU PPO example |
| [stockyard/baselines/](stockyard/baselines/) | Heuristics, SA, ACO and Gurobi |
| [stockyard/evaluation/](stockyard/evaluation/) | Policy rollout and shared benchmarks |
| [stockyard/data/](stockyard/data/) | Plate model and synthetic-data generation/loading |
| [stockyard/analysis/](stockyard/analysis/) | Result visualization |
| [tests/](tests/) | Handling constraints, interference, GAE, model and evaluation checks |
| [docs/](docs/) | Methodology, source mapping and validation records |

To follow the execution flow, start with the [environment](stockyard/environment/yard.py), then the [network](stockyard/models/network.py), [training](stockyard/training/ppo.py) and [evaluation](stockyard/evaluation/benchmark.py).

## Validation

The repository provides **48 tests** covering environment constraints, GAE, CSV validation, SA plan replay, encoder variants and model/evaluation integration. Local checks covered synthetic-data generation, a short PPO training run, baselines and result visualization. The original research trainer was also run with a small configuration, and Gurobi was checked on a four-plate example.

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m ruff check stockyard tests
python -m ruff format --check stockyard tests
```

GitHub Actions runs the tests and example workflow. See the [validation record](docs/VALIDATION.md) for the execution environment and scope.

## Data & Documentation

Industrial shipyard data and research checkpoints are not published. The repository provides independently generated synthetic data. The [provenance document](docs/PROVENANCE.md) distinguishes original research implementations from additions for public execution.

- [Problem definition and methodology](docs/METHODOLOGY.md)
- [Sample input and evaluation schema](docs/DATA_SCHEMA.md)
- [Original code and public adaptations](docs/PROVENANCE.md)
- [Execution validation](docs/VALIDATION.md)
