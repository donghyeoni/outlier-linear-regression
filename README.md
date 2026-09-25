# 이상치가 선형 회귀와 경사 기반 최적화에 미치는 영향

## 개요

본 프로젝트는 선형 회귀에서 이상치(outlier)가 경사 기반 최적화의 해에 미치는
영향을 분석하고, residual 기반의 반복적인 inlier 재선정으로 정상 집단의 weight를
복원하는 방법(`ours`)을 제안하고 평가한다.

데이터는 두 집단의 혼합으로 구성된다. 각 샘플은 확률 0.9로 weight `w1`을 따르는 정상
집단에, 확률 0.1로 부호가 반전된 `w2 = -w1`을 따르는 이상치 집단에 속한다. 그림 1은
입력이 1차원인 경우의 예시이다(점 200개 중 이상치 20개, 실제 기울기 1). 정상 집단만으로
최소제곱 적합을 하면 기울기는 1.01이지만, 이상치를 포함한 전체 데이터로 적합하면
0.84가 된다.

<p align="center">
  <img src="docs/images/concept.png" alt="그림 1" width="480">
</p>

*그림 1. 이상치 혼입에 따른 회귀선 편향(1차원 예시, 설명용 합성 데이터).*

모든 optimizer(GD, AdaGrad, RMSProp, Adam)와 closed-form solution은 scikit-learn이나
autograd 없이 NumPy로 직접 구현하였다. 방법의 개발 과정과 세부 수치는
[`docs/experiment-log.md`](docs/experiment-log.md)에 기록한다.

## 실험 내용

### 실험 설정

- **데이터:** `X ~ U[0,1)^{1000×4}`, `y = Xw + ε`, `ε ~ 0.1·N(0, 1)`. 실제
  weight(`w_true`, `w1`)는 데이터셋마다 `U[0,1)^4`에서 뽑는다.
- **혼합 데이터(실험 2):** 각 샘플은 확률 0.9로 `w1`을, 0.1로 `w2 = -w1`을 따른다.
- **데이터셋과 seed:** 데이터셋 하나는 seed 하나로 정해진다. seed는 `X`, 실제 weight,
  노이즈, 각 샘플의 이상치 여부를 결정한다. 학습의 초기 weight는 모든 실행에서
  seed 0으로 생성한다.
- **데이터셋 구분:** 방법의 개발과 선택에는 `seed=0–140`의 데이터셋을 사용하였다.
  최종 평가는 이와 겹치지 않는 `seed=141–240`의 데이터셋 100개(이상치 81–123개)에서
  한 번 수행하였으며, 이 데이터셋은 다른 용도로 사용하지 않았다.
- **epoch:** 파라미터 갱신 1회를 뜻한다. full batch는 전체 데이터, mini-batch는 샘플
  32개, SGD는 샘플 1개로 한 번 갱신한다.
- **평가 지표:**
  - **weight error** `‖ŵ − w_ref‖₂`: 주 지표. `w_ref`는 실험 1에서 `w_true`, 실험 2에서
    정상 집단의 `w1`이다.
  - **MSE**: 실험 1에서만 사용한다. optimizer의 MSE는 마지막 epoch에서 갱신 직전의
    예측으로 구한 해당 batch의 평균 제곱 오차이고(full batch에서는 전체 데이터의 학습
    MSE), closed-form solution의 MSE는 전체 데이터의 평균 제곱 오차이다.
- **표기:** residual은 `r = Xŵ − y`이다. 표준편차는 모집단 표준편차이다.

실험 구성은 그림 2와 같다.

```mermaid
flowchart TD
    A["합성 데이터 생성<br/>N = 1000, D = 4"] --> B["실험 1<br/>정상 데이터"]
    A --> C["실험 2<br/>혼합 데이터 (이상치 10%)"]

    B --> B1["optimizer 4종 학습"]
    B1 --> B2["closed-form solution과<br/>비교"]

    C --> C1["Oracle<br/>정상 집단만의<br/>closed-form solution"]
    C --> C2["Naive<br/>전체 데이터의<br/>closed-form solution"]
    C --> C3["optimizer 4종 학습"]
    C --> C4["ours<br/>residual 기반<br/>inlier 재선정"]
    C1 & C2 & C3 & C4 --> C5["정상 집단 weight w1<br/>대비 평가"]
```

*그림 2. 실험 구성.*

### 실험 1: 정상 데이터에서의 optimizer 비교

이상치가 없는 데이터에서 네 optimizer가 closed-form solution에 수렴하는지
확인한다. 이후 실험에서 사용할 optimizer 구현을 검증하는 기준 실험이다.

- optimizer 4종 × init 3종 × batch 3종(`full`, `mini-batch` 32, `SGD`)의 전체 조합.
  init은 `random`(`N(0, 1)`), `zero`(전부 0), `sparse`(`N(0, 1)`에서 약 80%를 0으로)이다.
- learning rate 0.1, 1000 epoch
- closed-form solution은 정규방정식 `(XᵀX)⁻¹Xᵀy`로 구한다.
- 이상치가 없는 정상 데이터셋 1개(`seed=0`)에서 수행한다.

### 실험 2: 혼합 데이터에서의 이상치 영향

다음 네 조건을 비교한다.

1. **Oracle:** 집단 label `z`를 이용해 정상 집단만으로 구한 closed-form
   solution(pseudo-inverse). 실제로는 `z`를 알 수 없으므로 비교 기준으로 사용한다.
2. **Naive:** 전체 데이터에 대한 closed-form solution(pseudo-inverse). 이상치를
   처리하지 않은 기준선이다.
3. **optimizer grid:** 실험 1과 같은 36개 조합(learning rate 0.01, 1000 epoch).
   optimizer 선택만으로 이상치의 영향이 줄어드는지 확인한다.
4. **`ours`:** 제안 방법.

### 제안 방법: `ours`

이상치 집단은 정상 집단과 반대 규칙을 따르므로, 적합된 모델에 대해 큰 residual을
가질 것으로 가정한다. `ours`는 full-batch Adam 학습과 inlier 집합의 재선정을 cycle
단위로 반복한다(그림 3).

1. weight와 optimizer 상태를 초기화하고, 현재 inlier 집합으로 Adam을 gradient
   norm이 `1e-5` 미만이 될 때까지 학습한다(cycle당 최대 20000 epoch). 첫 cycle의
   inlier 집합은 전체 데이터이다.
2. 현재 inlier의 residual로 robust scale
   `σ_MAD = 1.4826 · median(|r − median(r)|)`를 구한다.
3. **전체 샘플**의 residual을 계산하여, `|r| ≤ 3·σ_MAD`인 샘플을 새 inlier 집합으로
   정한다. 이전 cycle에서 제외된 샘플도 이 조건을 만족하면 다시 포함된다.
4. inlier 집합이 이전과 같으면 종료하고, 다르면 1로 돌아간다. 학습은 최대 5 cycle이며,
   5번째 cycle 뒤에는 재선정하지 않고 종료한다.

```mermaid
flowchart LR
    S["inlier 집합<br/>= 전체 데이터"] --> T["초기화 후 Adam<br/>수렴까지 학습"]
    T --> R["전체 샘플 중<br/>|r| ≤ 3·σ_MAD인 샘플로<br/>inlier 집합 재선정"]
    R --> D{"inlier 집합<br/>변화?<br/>(최대 5 cycle)"}
    D -- "예" --> T
    D -- "아니오" --> E["최종 weight"]
```

*그림 3. `ours`의 학습 절차.*

첫 cycle의 적합은 이상치의 영향을 받으므로, 이때 제외되는 정상 샘플은 residual
분포의 한쪽으로 치우칠 수 있다. `ours`는 매 cycle 전체 샘플에서 inlier를 다시 선정하여,
앞선 cycle에서 제외된 샘플이 다시 포함될 수 있게 한다. 개발 과정에서의 비교와 세부
진단은 로그(E4, E5, E8)에 기록한다.

*표 1. `ours`의 설정값.*

| 항목 | 값 |
| --- | --- |
| 재선정 기준 `k`(`k·σ_MAD`) | 3 |
| 수렴 기준(gradient norm) | `1e-5` |
| 최대 cycle 수 | 5 |
| cycle당 최대 epoch | 20000 |
| learning rate | 0.1 |
| 초기 weight | `N(0, 1)`. seed 0으로 한 번 설정한 난수열에서 매 cycle 새로 추출 |
| Adam | `β1 = 0.9`, `β2 = 0.999`, `ε = 1e-8` |

설정값은 개발 단계에서 정하였으며(로그 E3–E6), 최종 평가 전에 고정하였다. 구현은
`src/outlier_regression/outlier_removal.py`의 `ours(X, y, w_ref)`이며, 기본 설정값은
표 1과 같다.

## 결과

### 요약

1. 정상 데이터의 `init=zero`, `batch=full` 설정에서 GD, AdaGrad, Adam의 weight error는
   closed-form solution과 소수점 다섯째 자리까지 같다(0.02362).
2. 10%의 이상치만으로 Naive 해의 weight error는 Oracle의 약 13.2배(평균 0.0188 →
   0.2478)로 증가한다. optimizer grid의 36개 조합 중 평균 weight error가 가장 작은
   조합도 0.1882이다.
3. `ours`는 평가 데이터셋 100개 모두에서 Naive보다 작은 weight error를 보이며, 평균
   weight error는 0.0188로 Naive 대비 92.4% 작다. Oracle과의 평균 차이(+0.00003)는
   통계적으로 유의하지 않다.
4. learning rate 0.01, 0.1, 0.5에서 `ours`의 평균 weight error는 모두 0.0188이다.

### 실험 1: 정상 데이터

closed-form solution의 weight error는 **0.0236**, MSE는 **0.0103**이다.

표 2는 `init=zero`, `batch=full` 설정의 결과이며, 그림 4는 같은 설정에서의 학습
곡선이다(전체 조합은
[`results/baseline/optimizer_grid.csv`](results/baseline/optimizer_grid.csv)).

*표 2. 정상 데이터에서의 optimizer별 결과(`init=zero`, `batch=full`).*

| optimizer | MSE | weight error |
| --- | --- | --- |
| GD | 0.01028 | 0.02362 |
| AdaGrad | 0.01028 | 0.02362 |
| RMSProp | 0.02108 | 0.10608 |
| Adam | 0.01028 | 0.02362 |

RMSProp은 learning rate 0.1에서 마지막 100 epoch 동안 weight error가 진동하며,
closed-form solution에 도달하지 않는다
([`results/baseline/convergence_tail.json`](results/baseline/convergence_tail.json)).

| MSE | weight error |
| --- | --- |
| ![estimation](results/baseline/estimation_error.png) | ![weight](results/baseline/weight_error.png) |

*그림 4. `init=zero`, `batch=full` 설정에서 정상 데이터의 epoch별 MSE(좌)와 weight
error(우).*

### 실험 2: 혼합 데이터

표 3과 그림 5는 평가 데이터셋 100개(`seed=141–240`)에서의 결과이다
([`results/evaluation/`](results/evaluation/)).

*표 3. 평가 데이터셋 100개에서의 weight error. optimizer grid는 learning rate 0.01,
`ours`는 learning rate 0.1의 결과이다.*

| 방법 | weight error(평균 ± 표준편차) | 최소–최대 |
| --- | --- | --- |
| Oracle | 0.0188 ± 0.0082 | 0.0046–0.0415 |
| Naive | 0.2478 ± 0.0674 | 0.1218–0.4573 |
| optimizer grid, Adam(`full`, `zero`) | 0.2478 ± 0.0674 | 0.1218–0.4573 |
| `ours` | 0.0188 ± 0.0078 | 0.0053–0.0414 |

<p align="center">
  <img src="results/evaluation/weight_error_by_seed.png" alt="그림 5" width="520">
</p>

*그림 5. 평가 데이터셋 100개에서의 방법별 weight error(log scale). 점 하나가
데이터셋 하나이며, 검은 막대와 숫자는 중앙값이다.*

- **optimizer grid:** 모든 optimizer는 전체 데이터의 MSE라는 같은 손실을 최소화하며,
  이 손실을 최소화하는 해는 Naive 해이다. full-batch Adam(`zero` 초기화)의 weight error는
  데이터셋별로 Naive와 0.00055 이내로 일치한다. 36개 조합 중 평균 weight error가 가장
  작은 조합도 0.1882이다(전체 조합은
  [`results/evaluation/summary.json`](results/evaluation/summary.json)).
- **Oracle과의 비교:** `ours`와 Oracle의 평균 weight error 차이는 +0.00003이며,
  통계적으로 유의하지 않다(95% bootstrap 신뢰구간 [−0.00044, +0.00049], 양측
  sign-flip 검정 p = 0.895). 100개 중 49개에서는 `ours`의 weight error가 Oracle보다
  작다. Oracle은 정상 샘플을 모두 사용하고, `ours`는 정상 샘플 일부(평균 2.55개)를
  제외하는 대신 이상치 일부(100개 데이터셋에서 총 35개)를 포함하므로, 두 방법은 서로
  다른 샘플 집합에 적합된다.

### 한계

- residual이 `3·σ_MAD` 이내인 이상치는 제거되지 않는다. 평가 데이터셋 100개 중 24개에서
  이상치가 남았다(총 35개).
- 평가 데이터셋 100개 중 11개는 5 cycle 안에 inlier 집합이 더 이상 바뀌지 않음을
  확인하지 못하였다. 이 중 2개는 최대 cycle을 늘리면 결과가 달라진다(로그 E8).
- 재선정 기준 `k`는 3만 사용하였다.

### 후속 연구

본 실험은 부호가 반전된 단일 이상치 집단(`w2 = -w1`)과 이상치 비율 10%에서만
수행하였다. 후속 연구로 다음 조건에서 `ours`의 유효성을 확인할 필요가 있다.

- **이상치 비율:** 10%보다 높은 비율에서의 inlier 선정 정확도와 `σ_MAD` 추정의
  안정성.
- **이상치 형태:** 정상 규칙에 가까운 이상치(`w2 ≈ w1`), 입력 공간 밖의 leverage
  point, heavy-tailed 노이즈 등, 이상치의 residual이 크다는 가정이 약해지는 경우.
- **최대 cycle 수:** 5 cycle 안에 수렴하지 않는 경우를 없애는 종료 조건.
