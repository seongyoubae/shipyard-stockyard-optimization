# Problem and implementation notes

## Scope

The public task transfers plates from source piles to destination piles. Each action moves one source top plate. Destination choice is flexible: `topile` identifies the destination pool, not a binding per-plate destination assignment. Pile geometry, crane collision, travel distance, plate dimensions and real operational safety constraints are outside the public model. `unitw` is carried as a data field but is not an enforced weight capacity constraint.

Dynamic arrivals and outbound-date confirmation updates are supported by the uploaded environment. The current step implementation does not physically dispatch destination plates on their outbound dates. Dates express retrieval priority, and the episode ends when all source work and pending arrivals are cleared. Do not describe this as a complete outbound operations simulator.

## Blocking objective

For each destination pile listed bottom-to-top, count pairs `(i,j)` satisfying `i < j` and `outbound[i] < outbound[j]`. Sum over destination piles. Equal dates contribute zero. Blocking pairs measure order interference; actual relocation counts depend on the dispatch procedure.

## State and action

The retained model has a fixed 30-source / 30-destination padded representation. The default per-pile state has 21 features: 10 top outbound features, 4 deeper summaries, 4 upcoming inbound/outbound features, time, type and blocking features. Source/destination masks exclude empty sources, padded piles and full destinations. Idle dynamic periods are advanced by the environment when no source is active.

`PiGRUEncoder` fuses an embedding of the top outbound feature with the pile embedding and processes pile representations using a bidirectional GRU. This is a sequence over piles, not a GRU over an episode history. The public name “priority-aware GRU” describes the implementation; no separate unverified encoder name is asserted.

The retained network provides source/destination heads. Its `act_batch` sampling and `evaluate` likelihood use separate masked categorical distributions; the joint log probability is their sum. It should not be described as an exact autoregressive destination distribution conditioned on the sampled source action. `act_batch` switches to evaluation mode; the public PPO runner calls `train()` for optimization.

## Reward in the uploaded environment

Let `B_before`, `B_after` be destination blocking counts. Define `new_pairs` as earlier-due plates below the just-placed plate and `severity = sum(1 + 0.1 * due_gap)` over those plates.

- Step reward: `B_before - B_after - 0.5 * severity - 2 * new_pairs`
- Successful terminal reward: `10 - 100 * blocking_ratio`
- `blocking_ratio`: final pairs divided by `max(1, sum(n_pile*(n_pile-1)/2))`
- Overflow receives the environment's terminal overflow penalty

The current potential term is an undiscounted difference; configuration fields named `gamma` or reward weights do not imply that the step implementation applies discounted potential shaping. Several legacy configuration fields are retained for compatibility and are not all active in the reward.

## Public PPO runner

The small demo uses the uploaded environment and GRU network, an Adam optimizer at `1e-4`, discount `0.995`, GAE lambda `0.95`, ratio clip `0.2`, four optimization passes, value coefficient `0.5`, entropy coefficient `0.01`, and gradient norm clip `1.0`. Advantages respect episode termination and bootstrap at an unfinished rollout boundary. Normalized advantages are separated from unnormalized value returns.

This trainer is a new compact reproduction harness, not a claim that the original thesis training setup was identical. The much larger original trainer remains in `research_train.py` with import, seed and evaluation adapter repairs.

## Comparisons and fairness

All public benchmark methods consume the same CSV sorted by source and pile sequence, use the same destination capacity and initialize without obstacle plates. Seeds and checkpoints are explicit. The demo uses a small search budget; SA and ACO have different iteration schedules and may stop on their iteration cap before consuming the time budget. ACO's budget checks and SA's whole-sequence evaluations can slightly exceed the nominal deadline. Gurobi's solver time omits model construction, whereas rule/PPO rollout times omit reset and model loading. Therefore runtime numbers are not end-to-end, equal-compute research comparisons.

| Method | Actual implementation | Public scope |
|---|---|---|
| Random | Random available source / random feasible destination | Static and dynamic arrivals |
| EDD–MOD | Uploaded `source_P1_EDD` / `dest_PH1_MOD` | Static and dynamic arrivals |
| SOP–MFB | Uploaded shortest-outbound source / minimum final blocking destination | Static and dynamic arrivals |
| SA | Mutated action sequences replayed through yard with feasibility handling | Public runner restricted to static |
| ACO | Heuristic-seeded, heuristic-biased pheromone search | Static only; adapted implementation, not an independent paper reproduction |
| Gurobi | Sequential binary MIP with source precedence and pairwise destination ordering | Static, no obstacles or dynamic confirmation |
| PPO | Greedy evaluation of locally trained checkpoint | Static and dynamic arrivals, same capacity as checkpoint |

SA and ACO contain strong initialization or feasibility heuristics. Report this if using them in a paper. Gurobi termination status and optimality gap must accompany results. No research ranking or significance claim is derived from the smoke run. A proper experiment should use disjoint training/test seeds, multiple scenarios and model seeds, documented time budgets, completion rates, and dispersion of objective values.
