[English](README.md) · **한국어**

<div align="center">

# Shipyard Stockyard Optimization

### 조선소 강재 재배치의 반출 간섭 최소화를 위한 강화학습

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![PPO](https://img.shields.io/badge/Reinforcement%20Learning-PPO-2563EB?style=flat-square)

석사과정에서 수행한 강재 적치장 최적화 연구의 코드와 실행 예제

</div>

## Overview

조선소에서는 강재를 여러 층으로 쌓아 보관하고, 생산 일정에 따라 필요한 강재를 반출합니다. 이때 먼저 반출할 강재가 아래에 있고 나중에 반출할 강재가 위에 있으면, 위의 강재를 다른 곳으로 옮기는 작업이 필요합니다. 이러한 **반출 간섭**은 적치 순서에 따라 달라집니다.

이 연구는 출발지에 쌓인 강재를 도착지로 재배치할 때, **이동 순서와 도착지 선택을 함께 결정해 이후 반출 간섭을 줄이는 문제**를 다룹니다. 적치장 환경을 구현하고, 반출 우선순위를 반영한 GRU와 PPO를 사용해 이동 정책을 학습합니다. 규칙 기반 방법, SA, ACO, Gurobi MIP와 비교할 수 있는 평가 코드도 포함합니다.

![Example of steel plate retrieval interference](docs/stockyard.svg)

*반출일이 빠른 강재가 아래에 묻힌 경우와, 최상단에서 바로 반출할 수 있는 경우의 비교 예시*

## Problem Definition

강재는 각 pile에 아래에서 위 순서로 적치되어 있습니다. 크레인은 출발지 pile의 최상단 강재를 선택해 도착지 pile에 놓습니다. 이 과정을 반복해 출발지의 강재를 이전합니다.

| 항목 | 문제 설정 |
| :--- | :--- |
| 결정할 내용 | 어느 출발지의 최상단 강재를 어느 도착지로 옮길지 |
| 이동 제약 | 최상단 강재만 이동 가능 |
| 적재 제약 | 도착지 pile의 최대 적재 높이 준수 |
| 목적 | 재배치 완료 후 도착지의 반출 간섭 최소화 |
| 주 지표 | 아래 강재의 반출일이 위 강재보다 빠른 조합의 수, 즉 blocking pairs |

단순히 반출일 순서대로 강재를 나열하는 것만으로는 해결할 수 없습니다. 출발지에서는 최상단 강재부터 꺼내야 하고, 도착지에 한 번 놓은 강재는 다음 강재의 적치 상태에 영향을 주기 때문입니다. 따라서 현재 가능한 이동뿐 아니라 남은 강재와 도착지 상태까지 고려해야 합니다.

## Method

### Stockyard environment

출발지·도착지 pile과 강재의 입고·반출 정보를 관리하는 환경을 구현했습니다. 상태에는 상단 강재의 반출일, 깊은 층의 요약 통계, 예정 입고 정보, 시간과 간섭 정보를 포함합니다.

출발지가 비어 있거나 도착지의 적재 높이가 한계에 도달한 경우 해당 선택을 행동 마스크로 제외합니다. 동적 입고 시나리오에서는 시간에 따라 새 강재가 출발지에 도착하며, 입고 대기 중에는 다음 입고 시점으로 진행합니다.

### Priority-aware GRU

반출일은 강재의 작업 우선순위를 결정하는 주요 정보입니다. 모델은 pile 상태의 embedding에 상단 강재의 반출 우선순위 표현을 결합하고, GRU로 pile 간 정보를 처리합니다.

이 표현을 사용해 actor는 출발지와 도착지를 선택하고, critic은 상태 가치를 추정합니다. 코드에는 기본 GRU·LSTM·MLP·attention과 우선순위 결합 방식의 변형도 포함되어 있어 encoder 구성을 비교할 수 있습니다.

### PPO training

이동 전후의 간섭 변화와 새로 생성된 간섭, 반출일 차이를 보상에 반영합니다. 학습에는 PPO의 clipped objective와 GAE를 사용하며, 가치함수 손실과 entropy 항을 함께 계산합니다.

학습 코드는 두 가지 경로로 제공합니다.

| 코드 | 용도 |
| :--- | :--- |
| [`training/research_train.py`](stockyard/training/research_train.py) | 원본 연구의 병렬 rollout, 학습 및 시나리오 평가 흐름 |
| [`training/ppo.py`](stockyard/training/ppo.py) | 동일 환경·네트워크를 사용하는 소규모 CPU 실행 예제 |

상태 구성, 보상 식과 모델의 세부 동작은 [방법론 문서](docs/METHODOLOGY.md)에 설명했습니다.

## Baselines & Evaluation

공통 입력과 평가 지표를 사용해 다음 방법들을 실행할 수 있습니다.

| 방법 | 구현 내용 |
| :--- | :--- |
| Random | 이동 가능한 출발지·도착지의 무작위 선택 |
| EDD–MOD / SOP–MFB | 반출 우선순위와 간섭을 고려하는 규칙 기반 선택 |
| PPO + Priority-aware GRU | 학습된 정책으로 출발지·도착지 선택 |
| Simulated Annealing | 이동 행동 시퀀스를 변형하고 평가하며 해 탐색 |
| Ant Colony Optimization | 휴리스틱 초기해와 페로몬을 활용한 탐색 |
| Gurobi MIP | 출발지 이동 순서와 도착지 적치 순서를 반영한 수리최적화 |

평가 결과는 **최종 blocking pairs, 이동 수, 실행시간, 종료 상태**로 기록합니다. 완료 여부와 seed, 적재 높이, 강재 수도 함께 저장하며, 그래프에는 완료된 결과만 표시합니다. Gurobi는 종료 상태와 optimality gap도 확인할 수 있습니다. 공개 benchmark에서 SA·ACO·Gurobi는 정적 시나리오를 대상으로 합니다.

Blocking pairs는 반출 순서의 간섭 정도를 나타내는 지표입니다. 실제 추가 크레인 이동 횟수와 동일한 값으로 해석하지 않습니다.

## Quick Start

Python 3.10 이상에서 저장소를 내려받아 루트 디렉터리에서 실행합니다.

```bash
git clone https://github.com/seongyoubae/shipyard-stockyard-optimization.git
cd shipyard-stockyard-optimization

python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt

# 1. 합성 시나리오 생성
python -m stockyard.data.sample --seed 42

# 2. PPO 소규모 학습
python -m stockyard.training.ppo --updates 2 --horizon 32

# 3. 정책 및 baseline 평가
python -m stockyard.evaluation.benchmark --methods random edd-mod sop-mfb sa aco ppo

# 4. 평가 결과 시각화
python -m stockyard.analysis.plot
```

기본 예제는 강재 24장, 출발지 4개, 도착지 4개입니다. 위 명령의 2회 업데이트는 실행 확인용이며, 충분히 학습된 정책의 성능 평가와는 구분합니다.

| 생성 파일 | 내용 |
| :--- | :--- |
| `outputs/policy.pt` | 학습한 모델의 체크포인트 |
| `outputs/training.csv` | 학습 로그 |
| `outputs/benchmark.csv` | 방법별 평가 결과 |
| `outputs/benchmark.svg` | 평가 결과 그래프 |

<details>
<summary><strong>동적 입고 예제</strong></summary>

```bash
python -m stockyard.data.sample --dynamic --output outputs/dynamic.csv
python -m stockyard.evaluation.benchmark --data outputs/dynamic.csv --methods random edd-mod ppo
```

</details>

<details>
<summary><strong>Gurobi 실행</strong></summary>

```bash
python -m pip install -r requirements-gurobi.txt
python -m stockyard.data.sample --sources 2 --destinations 2 --plates-per-source 2 --output outputs/tiny.csv
python -m stockyard.evaluation.benchmark --data outputs/tiny.csv --methods gurobi --budget-seconds 2
```

유효한 Gurobi 라이선스가 필요합니다. 라이선스에 따라 실행할 수 있는 모델 크기가 달라집니다.

</details>

## Code Structure

| 경로 | 역할 |
| :--- | :--- |
| [`stockyard/environment/`](stockyard/environment/) | 적치장 상태, 이동 제약, 행동 마스크, 보상·종료 |
| [`stockyard/models/`](stockyard/models/) | GRU 및 encoder 변형, actor–critic |
| [`stockyard/training/`](stockyard/training/) | 연구 학습 코드와 CPU용 PPO 예제 |
| [`stockyard/baselines/`](stockyard/baselines/) | Heuristic, SA, ACO, Gurobi |
| [`stockyard/evaluation/`](stockyard/evaluation/) | 정책 rollout과 공통 benchmark |
| [`stockyard/data/`](stockyard/data/) | 강재 모델, 합성 데이터 생성·로딩 |
| [`stockyard/analysis/`](stockyard/analysis/) | 결과 시각화 |
| [`tests/`](tests/) | 환경 제약, 간섭 계산, GAE, 모델·평가 연결 검증 |
| [`docs/`](docs/) | 방법론, 원본 코드와의 대응 관계, 검증 기록 |

코드를 살펴볼 때는 [환경](stockyard/environment/yard.py), [네트워크](stockyard/models/network.py), [학습](stockyard/training/ppo.py), [평가](stockyard/evaluation/benchmark.py) 순서로 보면 전체 흐름을 확인할 수 있습니다.

## Validation

환경 제약과 GAE, CSV 입력 검증, SA 계획 재실행, encoder 변형 및 모델·평가 연결을 확인하는 **48개 테스트**를 제공합니다. 합성 데이터 생성부터 PPO 짧은 학습, baseline 실행, 결과 시각화까지 로컬에서 확인했습니다. 원본 연구 학습 코드도 작은 설정으로 실행했고, Gurobi는 강재 4장의 소규모 예제를 확인했습니다.

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m ruff check stockyard tests
python -m ruff format --check stockyard tests
```

GitHub Actions에는 테스트와 예제 실행 workflow를 구성했습니다. 실행 환경과 검증 범위는 [검증 기록](docs/VALIDATION.md)에서 확인할 수 있습니다.

## Data & Documentation

실제 조선소 데이터와 연구 체크포인트는 공개하지 않고, 독립적으로 생성한 합성 데이터를 제공합니다. 원본 연구 구현과 공개 실행을 위해 추가한 코드는 [구현 이력](docs/PROVENANCE.md)에 구분해 정리했습니다.

- [문제 정의와 방법론](docs/METHODOLOGY.md)
- [샘플 데이터와 평가 결과 형식](docs/DATA_SCHEMA.md)
- [원본 코드 및 공개용 수정 사항](docs/PROVENANCE.md)
- [실행 검증 기록](docs/VALIDATION.md)
