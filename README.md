<div align="center">

# Shipyard Stockyard Optimization

### Deep Reinforcement Learning for Minimizing Retrieval Interference in Steel Plate Stockyards

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![PPO](https://img.shields.io/badge/Reinforcement%20Learning-PPO-2563EB?style=flat-square)
[![Public sample validation](https://github.com/seongyoubae/shipyard-stockyard-optimization/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/seongyoubae/shipyard-stockyard-optimization/actions/workflows/ci.yml)

**M.S. research project · Shipbuilding production optimization · Python / PyTorch / PPO**

</div>

## At a Glance

This project develops a reinforcement learning framework for steel plate reshuffling in shipyard stockyards.

- **Problem** — Minimize retrieval-order interference caused by unfavorable stacking
- **Decision** — Select a source pile and destination pile for each top-plate transfer
- **Model** — Priority-aware GRU encoder with an actor–critic network
- **Learning** — Proximal Policy Optimization (PPO) with Generalized Advantage Estimation (GAE)
- **Comparisons** — Random and rule-based policies, Simulated Annealing, Ant Colony Optimization, and Gurobi MIP
- **Evaluation** — Blocking pairs, plate transfers, runtime, and completion status
- **Data** — Reproducible synthetic scenarios; industrial raw data and trained research checkpoints are excluded

## Overview

Steel plates in shipyards are temporarily stored in multi-layer piles before retrieval according to production schedules. When an earlier-due plate is buried beneath later-due plates, additional handling may be required. Stacking decisions therefore affect future retrieval interference.

This project formulates reshuffling as a sequential decision problem: select a top plate from a source pile and assign it to a feasible destination pile while considering retrieval priorities and the current stockyard state. A priority-aware GRU-based actor–critic model is trained with PPO, and alternative algorithms are evaluated through a common benchmark interface.

> **Objective:** minimize final retrieval-order interference while satisfying top-only movement and destination stack-height constraints.

![Example of steel plate retrieval interference](docs/stockyard.svg)

*An earlier-due target plate requires access through upper plates when buried; placing it on top allows direct retrieval. This illustration explains interference rather than simulating outbound dispatch.*

## What I Implemented

The research implementation connects stockyard simulation, state representation, and policy learning:

- Stockyard environment with top-only movement, stack-height constraints, and dynamic arrivals
- State features for retrieval priorities, pile conditions, upcoming arrivals, time, and blocking information
- Priority-aware GRU encoder and actor–critic network, with alternative encoder and priority-fusion variants
- Action masks for empty sources, padded piles, and full destinations
- PPO research training flow and heuristic, SA, ACO, and sequential Gurobi MIP implementations

The public repository adds a compact CPU PPO runner, a synthetic scenario generator, a common benchmark and plotting interface, input validation, automated tests, and a CI workflow. These additions support reproducible execution of the public adaptation. The [provenance document](docs/PROVENANCE.md) distinguishes retained research code from public additions and maintenance changes.

## Problem Definition

Each pile stores plates from bottom to top. A crane transfers one source top plate to a destination pile, repeating until all source plates and pending arrivals are cleared.

| Item | Definition |
| :--- | :--- |
| Decision | Which source top plate to move and which destination pile to use |
| Movement constraint | Only the top plate of a source pile can be moved |
| Capacity constraint | Destination piles must respect the maximum stack height |
| Objective | Minimize destination blocking pairs after reshuffling |
| Blocking pair | A lower plate has an earlier retrieval date than an upper plate in the same pile; equal dates do not count |

A simple sort by retrieval date is insufficient: source access is constrained by the existing stack, and each destination choice changes the options for subsequent placements.

**Blocking pairs measure retrieval-order interference, not the actual number of additional crane moves.** The public environment supports dynamic arrivals, but does not physically dispatch destination plates on their outbound dates. Retrieval dates express priority; geometry, crane travel distance, collision avoidance, and weight capacity are outside the public model.

## Method

### 1. Stockyard Environment

The environment represents source and destination piles and exposes pile-level retrieval, arrival, and blocking features. An action is a pair `(source_pile, destination_pile)`; infeasible choices are excluded through action masks. Idle periods in dynamic scenarios advance to the next arrival.

Rewards reflect changes in interference, newly created blocking pairs, and retrieval-date gaps, together with a terminal blocking-ratio term.

### 2. Priority-aware GRU

The encoder combines pile embeddings with an explicit representation of the top plate's retrieval priority. A bidirectional GRU processes the sequence of pile representations to capture context across piles.

The actor has source and destination selection heads, and the critic estimates state value. The GRU processes piles within a state, rather than an episode's temporal history.

### 3. PPO

PPO training uses GAE, a clipped policy objective, value-function loss, and entropy regularization.

| Training path | Purpose |
| :--- | :--- |
| [`research_train.py`](stockyard/training/research_train.py) | Retained research training flow with parallel rollouts and evaluation adapter repairs |
| [`ppo.py`](stockyard/training/ppo.py) | Compact CPU demonstration using the same environment and network |

See [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) for the exact state representation, reward, action distributions, and public implementation scope.

## Results

The public example demonstrates execution and integration. It does not reproduce the full industrial research experiment or establish algorithm superiority.

The [validation record](docs/VALIDATION.md) documents the following local execution checks:

| Check | Recorded outcome |
| :--- | :--- |
| Static synthetic example | All six methods—Random, EDD–MOD, SOP–MFB, SA, ACO, and PPO—completed 24 transfers and returned finite objectives |
| Compact PPO training | Two updates produced finite losses and a checkpoint loaded by the benchmark |
| Dynamic arrivals | Random, EDD–MOD, SOP–MFB, and PPO completed; static-only methods were skipped |
| Tiny Gurobi example | Four-plate instance solved with objective zero, Optimal status, and zero gap |

A two-update policy is an execution example. No thesis improvement percentage, research ranking, or statistical significance claim is inferred from these checks. Full thesis results are not reproduced because the industrial dataset and trained research checkpoints are not publicly distributed.

## Benchmark Algorithms

All methods consume the same validated scenario input and destination capacity.

| Method | Implementation | Public scenario scope |
| :--- | :--- | :--- |
| Random | Random feasible source and destination selection | Static and dynamic arrivals |
| EDD–MOD / SOP–MFB | Retrieval-priority and interference-based rules | Static and dynamic arrivals |
| PPO + Priority-aware GRU | Greedy evaluation of a locally trained checkpoint | Static and dynamic arrivals |
| Simulated Annealing | Action-sequence search with feasibility repair and best-plan replay | Static |
| Ant Colony Optimization | Heuristic-seeded, heuristic-biased pheromone search | Static |
| Gurobi MIP | Sequential binary model with source precedence and destination ordering | Static |

### Evaluation Metrics

- **Blocking pairs** — Final retrieval-order interference
- **Crane moves** — Number of source-to-destination plate transfers
- **Runtime** — Method-specific execution time reported by the benchmark
- **Completion** — Whether all required source plates were successfully transferred
- **Solver status and MIP gap** — Reported for Gurobi where available

Output also records the seed, stack height, and plate count. Plots include completed runs only. Runtime boundaries and search schedules differ between methods, so demo timings are not an equal-compute research comparison. See the [comparison notes](docs/METHODOLOGY.md#comparisons-and-fairness) before interpreting benchmark results.

## Quick Start

The following commands run a small CPU-friendly demonstration using synthetic data. Use Python 3.10 or later and run them from the repository root.

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

# 2. Run a short PPO training example
python -m stockyard.training.ppo --updates 2 --horizon 32

# 3. Evaluate the policy and baselines
python -m stockyard.evaluation.benchmark --methods random edd-mod sop-mfb sa aco ppo

# 4. Plot benchmark results
python -m stockyard.analysis.plot
```

The default scenario contains 24 plates, four source piles, and four destination piles. Two PPO updates verify execution; they do not produce a policy suitable for research performance claims.

| Generated file | Contents |
| :--- | :--- |
| `outputs/policy.pt` | Locally trained model checkpoint |
| `outputs/training.csv` | Training log |
| `outputs/benchmark.csv` | Per-method evaluation results |
| `outputs/benchmark.svg` | Benchmark plot |

<details>
<summary><strong>Dynamic arrival example</strong></summary>

Run after the quick start has created the PPO checkpoint.

```bash
python -m stockyard.data.sample --dynamic --output outputs/dynamic.csv
python -m stockyard.evaluation.benchmark --data outputs/dynamic.csv --methods random edd-mod ppo
```

</details>

<details>
<summary><strong>Gurobi example</strong></summary>

```bash
python -m pip install -r requirements-gurobi.txt
python -m stockyard.data.sample --sources 2 --destinations 2 --plates-per-source 2 --output outputs/tiny.csv
python -m stockyard.evaluation.benchmark --data outputs/tiny.csv --methods gurobi --budget-seconds 2
```

A valid Gurobi license is required. Supported model size depends on the license.

</details>

## Repository Structure

| Path | Purpose |
| :--- | :--- |
| [`stockyard/environment/`](stockyard/environment/) | Yard state, movement constraints, masks, rewards, and termination |
| [`stockyard/models/`](stockyard/models/) | GRU and alternative encoders, actor–critic network |
| [`stockyard/training/`](stockyard/training/) | Research training flow and compact CPU PPO runner |
| [`stockyard/baselines/`](stockyard/baselines/) | Heuristics, SA, ACO, and Gurobi |
| [`stockyard/evaluation/`](stockyard/evaluation/) | Policy rollouts and common benchmark |
| [`stockyard/data/`](stockyard/data/) | Plate model, synthetic generation, and validated input loading |
| [`stockyard/analysis/`](stockyard/analysis/) | Result visualization |
| [`tests/`](tests/) | Environment, input, GAE, baseline, and model integration checks |
| [`docs/`](docs/) | Methodology, schema, provenance, and validation record |

Suggested code reading order: [environment](stockyard/environment/yard.py), [network](stockyard/models/network.py), [training](stockyard/training/ppo.py), and [benchmark](stockyard/evaluation/benchmark.py).

## Engineering Validation

The [validation record](docs/VALIDATION.md) reports **48 passing automated tests** covering movement and capacity constraints, blocking calculations, dynamic arrivals, GAE, CSV validation, baseline replay, encoder variants, and model–evaluation integration.

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m ruff check stockyard tests
python -m ruff format --check stockyard tests
```

The [GitHub Actions workflow](.github/workflows/ci.yml) runs tests, Ruff checks, synthetic data generation, short PPO training, benchmark execution, and plotting on Python 3.11. Its current status is available in the [Actions tab](https://github.com/seongyoubae/shipyard-stockyard-optimization/actions).

## Data & Documentation

Public examples use independently generated synthetic data. See these documents for implementation details and reproducibility boundaries:

- [Problem definition and methodology](docs/METHODOLOGY.md)
- [Sample data and evaluation output schema](docs/DATA_SCHEMA.md)
- [Research code provenance and public changes](docs/PROVENANCE.md)
- [Execution validation record](docs/VALIDATION.md)

## Research Context

This repository is a public adaptation of a master's research project in shipyard production optimization. The original research used industrial shipyard data; raw industrial records, trained research checkpoints, and proprietary information are excluded.

The public adaptation retains the research environment and network with documented repairs, and provides synthetic scenarios and a compact execution workflow. Public additions and modifications are listed in [`docs/PROVENANCE.md`](docs/PROVENANCE.md).

## Author

**Seongyou Bae**

M.S. in Naval Architecture and Ocean Engineering  
Seoul National University

Former Marine Engineer  
Interests: Maritime AI · Industrial Optimization · Reinforcement Learning
