# Shipyard Stockyard Optimization

### 조선소 강재 적치·반출 간섭 최소화를 위한 강화학습 기반 최적화

**현장 운영 문제를 환경·제약·보상으로 모델링하고, priority-aware GRU와 PPO로 재배치 의사결정을 학습한 석사 연구의 공개용 포트폴리오입니다**

`Python` · `PyTorch` · `PPO` · `GRU` · `Operations Research` · `Simulation`

![Problem illustration](docs/stockyard.svg)

## 프로젝트를 1분 안에 이해하기

강재가 여러 층으로 쌓인 적치장에서는 먼저 반출할 강재 위에 나중에 반출할 강재가 놓이면 추가 이동이 필요합니다. 이 프로젝트는 **어느 출발지의 최상단 강재를 선택해 어느 도착지에 적치할지** 결정하여, 재배치 후 반출 순서 간섭을 줄이는 문제를 다룹니다.

| 항목 | 구현 내용 |
|---|---|
| 의사결정 | 출발지 pile 선택 + 도착지 pile 선택 |
| 제약조건 | 최상단 강재만 이동, 도착지 적재 높이 제한, 유효 행동 마스킹 |
| 상태 | 상단 강재의 반출일, 깊은 층의 요약 통계, 입고 예정 정보, 시간, pile 유형, 간섭 정보 |
| 모델 | 반출 우선순위를 결합하는 GRU encoder + actor/critic |
| 학습 | PPO clipped objective, GAE, value loss, entropy regularization |
| 평가 | 최종 blocking pair 수, 이동 수, 실행시간, 종료 상태 |
| 비교 방법 | Random, EDD–MOD, SOP–MFB, SA, ACO, Gurobi MIP |

## 이 코드에서 확인할 수 있는 역량

- **산업 문제의 수학적 모델링**: 적치 순서와 반출일을 비교해 간섭 지표를 정의하고, 이동·용량 제약을 환경에 반영
- **강화학습 구현**: 상태 설계, 행동 마스킹, reward shaping, PPO 학습·평가 연결
- **시계열·우선순위 표현 학습**: 반출 정보와 pile 표현을 결합하는 GRU 및 여러 encoder 변형 구현
- **최적화 비교 실험**: 규칙 기반 방법, 메타휴리스틱, 수리최적화를 공통 문제와 지표로 비교
- **재현 가능한 코드 정리**: 역할별 패키지, 합성 데이터, 실행 명령, 테스트, 결과 시각화

원본 연구 코드에서 가져온 알고리즘과 공개용 실행 편의를 위해 추가한 코드를 구분한 [구현 이력](docs/PROVENANCE.md)을 제공합니다.

## 빠른 실행

Python 3.10 이상에서 저장소 루트 기준으로 실행합니다. CPU만으로 데모를 실행할 수 있습니다. 기본 예제는 강재 24장, 출발지 4개, 도착지 4개입니다.

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# 1. 공개용 합성 시나리오 생성
python -m stockyard.data.sample --seed 42

# 2. PPO 짧은 실행 검증 — 성능 학습에는 더 많은 update가 필요
python -m stockyard.training.ppo --updates 2 --horizon 32

# 3. 동일 시나리오에서 정책 및 baseline 평가
python -m stockyard.evaluation.benchmark --methods random edd-mod sop-mfb sa aco ppo

# 4. 결과 시각화
python -m stockyard.analysis.plot
```

생성 결과는 `outputs/training.csv`, `outputs/policy.pt`, `outputs/benchmark.csv`, `outputs/benchmark.svg`에 저장됩니다. 체크포인트는 실행자가 직접 생성하며 저장소에 포함하지 않습니다.

### Gurobi 선택 실행

```bash
python -m pip install -r requirements-gurobi.txt
python -m stockyard.data.sample --sources 2 --destinations 2 --plates-per-source 2 --output outputs/tiny.csv
python -m stockyard.evaluation.benchmark --data outputs/tiny.csv --methods gurobi --budget-seconds 2
```

Gurobi 실행에는 유효한 라이선스가 필요합니다. 라이선스 파일은 커밋하지 않습니다. 큰 문제는 제한 라이선스의 모델 크기를 초과할 수 있습니다. Gurobi는 정적 시나리오의 순차 이동 MIP이며, 시간 제한으로 종료된 경우 feasible solution과 optimal solution을 구분합니다.

### 동적 입고 예제

```bash
python -m stockyard.data.sample --dynamic --output outputs/dynamic.csv
python -m stockyard.evaluation.benchmark --data outputs/dynamic.csv --methods random edd-mod ppo
```

공개 benchmark의 SA·ACO·Gurobi 비교는 정적 입고에 한정됩니다. 동적 입력을 주면 이 세 방법은 명시적으로 건너뜁니다.

## 코드 구조

| 경로 | 역할 |
|---|---|
| `stockyard/data/` | Plate 모델, 연구용 합성 생성기, 공개 sample 생성·로딩 |
| `stockyard/environment/yard.py` | 적치장 상태, top-only 이동, 마스크, 동적 입고, 보상·종료 |
| `stockyard/models/network.py` | priority GRU, actor/critic, encoder 변형 |
| `stockyard/training/ppo.py` | 작은 CPU용 공개 PPO 데모 |
| `stockyard/training/research_train.py` | 원본의 대규모 병렬 rollout·연구 학습 흐름 |
| `stockyard/evaluation/` | 정책 rollout, 학습 평가 adapter, 공통 benchmark |
| `stockyard/baselines/` | heuristic, Gurobi, SA, ACO |
| `stockyard/baselines/legacy/` | 이전 SA move/annealer 구현; 기본 benchmark에서 미사용 |
| `stockyard/analysis/` | benchmark 결과 시각화 |
| `data/sample/` | 독립적으로 생성한 공개 CSV |
| `docs/` | 문제 정의, 구현 이력, 검증 결과, 시각 자료 |
| `tests/` | 제약·간섭·동적 입고·네트워크·GAE 검증 |

연구 학습 흐름은 `python -m stockyard.training.research_train --help`로 설정을 확인할 수 있습니다. 이 경로는 실험 규모가 크며 전체 학습을 재수행하여 검증하지 않았습니다. 검증된 시작점은 위의 공개 데모입니다.

## 지표와 실험 해석

pile의 배열은 아래에서 위 순서입니다. 아래 강재의 반출일이 위 강재보다 빠른 조합을 **blocking pair**로 셉니다. 같은 반출일은 간섭으로 세지 않습니다. 이 수치는 잠재적인 반출 순서 간섭이며 실제 추가 크레인 이동 횟수와 일대일로 같지 않습니다.

보상은 간섭 변화, 새로 생성된 간섭 수, 반출일 차이의 심각도, 종료 시 간섭 비율을 반영합니다. 모든 식과 방법별 범위는 [방법론](docs/METHODOLOGY.md)에 설명합니다.

이 저장소는 **공개용 합성 데이터의 실행 가능성을 보여주는 연구 코드 포트폴리오**입니다. 실제 조선소 데이터, 원본 실적 분포 통계, 원본 학습 체크포인트, 논문 실험 결과는 포함하지 않습니다. 짧은 PPO 실행은 학습 성공이나 heuristic 대비 성능 우위를 입증하지 않습니다. 합성 데모 수치를 학위논문의 성능 수치로 사용하지 않습니다.

## 검증

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

GitHub Actions는 테스트와 합성 데이터 생성 → PPO 짧은 학습 → baseline/정책 평가 → 시각화 흐름을 실행하도록 구성되어 있습니다. 로컬 확인 결과와 제한사항은 [검증 기록](docs/VALIDATION.md)을 참고하세요.

## 공개 데이터 원칙

공개 CSV는 `SYN_` 식별자와 상대적인 day 값을 사용하며 실제 업체·선박·작업번호·담당자 정보를 담지 않습니다. 기본 생성기의 기간·중량 설정은 일반적인 예제 값입니다. 비공개 데이터, 환경 변수 파일, 라이선스, 학습 모델, 로컬 실행 결과는 `.gitignore`로 제외합니다.

실제 운영 적용에는 추가 현장 제약, 시간 단위, 장비·이동 모델, 반출 처리, 안전 규칙 및 분포 변화 검증이 필요합니다. 코드 공개 자체가 상용 시스템 적용 완료를 의미하지 않습니다.
