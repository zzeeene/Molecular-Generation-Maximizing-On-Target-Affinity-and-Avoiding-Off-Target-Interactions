# Molecular Generation for Selective LRRK2 Drug Discovery

> **파킨슨병 관련 표적 LRRK2에 대한 결합 점수는 높이고, 잠재적 off-target 결합과 독성은 낮추는 다목적 분자 생성 프로젝트**

이 프로젝트는 사전학습된 SMILES 생성 모델 **NovoMolGen**을 강화학습으로 미세조정하여, 다음 조건을 동시에 만족하는 신규 저분자 후보를 생성하는 proof-of-concept 연구입니다.

- LRRK2에 대한 높은 예측 결합 점수
- LRRK2를 제외한 KIBA kinase pool에 대한 낮은 평균 결합 점수
- 낮은 Tox21 독성 확률
- 높은 QED drug-likeness
- 낮은 SA(Synthetic Accessibility) score

> 이 저장소의 결과는 계산 모델이 산출한 예측값이며, 실제 생물학적 활성이나 임상적 유효성을 의미하지 않습니다.

## Motivation

파킨슨병에는 질병의 진행 자체를 근본적으로 억제하는 치료제가 충분하지 않으며, 새로운 disease-modifying therapeutic 후보 발굴이 필요합니다. 이 프로젝트는 파킨슨병과 연관된 kinase인 **LRRK2**를 on-target으로 설정했습니다.

저분자 약물은 의도하지 않은 단백질에도 결합할 수 있기 때문에, 단순히 on-target affinity만 높이는 최적화는 off-target interaction과 독성 위험을 함께 증가시킬 수 있습니다. 따라서 본 프로젝트는 결합 예측, 독성 예측, 약물 유사성, 합성 용이성을 하나의 reward로 통합하여 선택적인 분자 생성을 시도합니다.

## End-to-End Pipeline

```
KIBA + Tox21
     │
     ├─ DTA predictor
     │   Morgan count fingerprint + ESM-2 protein embedding
     │   → LRRK2 / 228 off-target kinase binding score
     │
     ├─ Tox21 multitask predictor
     │   Morgan count fingerprint → 12 toxicity probabilities
     │
     └─ Molecular descriptors
         QED + SA score
                 │
                 ▼
        Multi-objective reward
                 │
                 ▼
    NovoMolGen policy optimization
      REINFORCE + KL prior anchor
                 │
                 ▼
     Ranked candidate molecules
```

## Data and Molecular Representation

### 1. KIBA: Drug-Target Affinity

KIBA 데이터는 PyTDC를 통해 불러오며, 새로운 화합물에 대한 일반화 성능을 평가하기 위해 **cold-drug split**을 사용합니다.

| Split | Samples |
|---|---:|
| Train | 82,539 |
| Validation | 12,023 |
| Test | 23,095 |

- Train/validation/test 간 drug overlap: **0**
- KIBA target proteins: **229 kinases**
- On-target: **LRRK2** (`Target_ID`에 `Q5S007` 포함)
- Off-target pool: LRRK2를 제외한 **228 kinases**

**Drug representation**

- Morgan count fingerprint
- Radius: `2`
- Dimension: `1,024`

**Protein representation**

- `esm2_t6_8M_UR50D`
- Representation layer: `6`
- Mean-pooled embedding dimension: `320`
- Maximum sequence length: `2,527`

### 2. Tox21: Multitask Toxicity Prediction

Tox21 데이터의 12개 binary endpoint를 동시에 예측합니다.

- Valid molecules: **7,831**
- Split: random `80/10/10`
- Missing label은 mask 처리
- Class imbalance는 endpoint별 `pos_weight`로 보정

사용된 endpoints:

`NR-AR`, `NR-AR-LBD`, `NR-AhR`, `NR-Aromatase`, `NR-ER`, `NR-ER-LBD`, `NR-PPAR-gamma`, `SR-ARE`, `SR-ATAD5`, `SR-HSE`, `SR-MMP`, `SR-p53`

## Models

### Drug-Target Affinity Predictor

KIBA 결합 점수를 예측하는 PyTorch 모델입니다.

1. 1,024차원 drug fingerprint와 320차원 protein embedding을 각각 256차원으로 projection
2. 4-head cross-attention으로 drug/protein feature interaction 계산
3. 두 representation을 concatenate
4. MLP regressor로 KIBA score 예측

주요 학습 설정:

| Parameter | Value |
|---|---:|
| Optimizer | AdamW |
| Learning rate | `1e-4` |
| Batch size | `256` |
| Max epochs | `50` |
| Early-stopping patience | `10` |
| Hidden dimension | `512` |
| Dropout | `0.2` |

### Tox21 Predictor

Morgan count fingerprint를 입력으로 받는 12-task MLP입니다.

### Molecular Generator

- Base model: `chandar-lab/NovoMolGen_32M_SMILES_AtomWise`
- Revision: `hf-checkpoint`
- Sampling: temperature `1.0`, top-p `0.95`
- Maximum generated tokens: `96`

## Reward Function

각 유효 SMILES에 대해 다음 reward를 계산합니다.

```
Reward = 0.4 × LRRK2_binding
       - 0.4 × mean_binding_to_228_non_LRRK2_kinases
       - 0.5 × mean_Tox21_toxicity_probability
       + 0.4 × QED
       - 0.1 × SA_score
```

| Term | Optimization direction |
|---|---|
| LRRK2 predicted binding | Maximize |
| Mean non-LRRK2 kinase binding | Minimize |
| Mean Tox21 toxicity probability | Minimize |
| QED | Maximize |
| SA score | Minimize |

Invalid SMILES에는 `-10.0`의 reward를 부여합니다.

## Reinforcement Learning

NovoMolGen을 policy로 사용하고, frozen pretrained model을 prior로 유지합니다.

- Policy-gradient method: **REINFORCE**
- Exponential moving reward baseline
- Batch 내 standardized advantage
- KL-style prior anchor로 pretrained distribution에서 과도하게 벗어나는 현상 완화
- Gradient clipping: `1.0`
- Best checkpoint criterion: batch `reward_mean`

| Parameter | Value |
|---|---:|
| RL steps | `400` |
| Batch size | `32` |
| Learning rate | `1e-5` |
| KL anchor beta | `0.001` |
| Checkpoint interval | `25` steps |
| Random seed | `42` |

## Results

### Before vs. After Reinforcement Learning

각 phase에서 32개 분자를 생성하고 동일한 scoring pipeline으로 평가했습니다.

| Metric | Before RL | After RL | Delta |
|---|---:|---:|---:|
| Valid fraction | 1.0000 | 1.0000 | 0.0000 |
| Mean reward | -0.0571 | 0.0349 | **+0.0921** |
| Mean LRRK2 binding score | 11.5099 | 11.6593 | **+0.1494** |
| Mean non-LRRK2 kinase binding | 11.1813 | 11.2653 | +0.0840 |
| Mean Tox21 probability | 0.2387 | 0.2375 | **-0.0012** |
| Mean QED | 0.7116 | 0.7693 | **+0.0577** |
| Mean SA score | 3.5393 | 3.1168 | **-0.4225** |

RL 이후 reward, LRRK2 score, Tox prob, QED, SA score가 모두 개선되었습니다. 반면 non-LRRK2 kinase에 대한 평균 결합 점수도 함께 증가하여, **on-target affinity와 kinase selectivity 사이의 trade-off가 완전히 해결되지는 않았습니다.**

### Top-Ranked Post-RL Candidate

```
SMILES: CN(CC1CCN(C(=O)c2cccnc2)CC1)C(=O)c1cccc(O)c1
Reward: 0.2351
LRRK2 binding: 11.7404
Mean non-LRRK2 kinase binding: 11.3105
Mean Tox21 probability: 0.1770
QED: 0.9160
SA score: 2.1474
```

이 분자는 계산 모델에 의해 상위 후보로 선정된 것이며, docking, molecular dynamics, ADMET evaluation, in vitro assay 등의 후속 검증이 필요합니다.

## Getting Started

### Recommended Environment

- Python 3.10+
- CUDA-enabled GPU 권장
- Google Colab 또는 Jupyter Notebook

### Installation

Notebook 첫 번째 cell을 실행하거나, 로컬 환경에서 다음 패키지를 설치합니다.

```bash
pip install torch pytdc transformers rdkit fair-esm pandas numpy scipy scikit-learn tqdm jupyter
```

### Tox21 Data

KIBA 데이터는 PyTDC가 자동으로 다운로드하지만, Tox21 CSV는 준비가 필요합니다.

- Colab에서는 `tox21.csv`를 `/content/`에 업로드합니다.
- 로컬 실행 시 실제 파일 위치에 맞게 `TOX21_CSV_PATH`를 수정합니다.

### Run

```bash
jupyter notebook Parkinson_Drug_Discovery_RL.ipynb
```

Notebook에서 위에서 아래 순서로 모든 cell을 실행합니다. 전체 실행 과정에는 다음 작업이 포함됩니다.

1. KIBA download 및 cold-drug split
2. Morgan fingerprint 생성
3. ESM-2 protein embedding 생성 및 cache 저장
4. DTA predictor 학습
5. Tox21 predictor 학습
6. NovoMolGen loading
7. Pre-RL sampling 및 scoring
8. 400-step reinforcement learning
9. Post-RL sampling, ranking, model/checkpoint 저장

## Reproducibility

- Global random seed: `42`
- KIBA split seed: `42`
- Tox21 split seed: `42`
- Protein embeddings are cached under `cache/`
- Cold-drug split의 drug overlap을 명시적으로 검증합니다.

GPU 종류, CUDA/PyTorch 버전, dependency 버전에 따라 생성 분자와 학습 결과는 달라질 수 있습니다.

## Limitations and Future Work

1. **Predictor-dependent reward**  
   생성 분자의 품질은 DTA 및 toxicity predictor의 정확도에 의존합니다.

2. **Limited toxicity coverage**  
   Tox21의 12개 endpoint만으로 전체 독성 및 약동학 특성을 평가할 수 없습니다. 향후 hERG, CYP450, BBB permeability, solubility, metabolic stability 등 더 넓은 ADMET endpoint가 필요합니다.

3. **On-target/off-target coupling**  
   LRRK2 결합 점수를 높일수록 다른 kinase에 대한 결합 점수도 함께 높아지는 경향이 관찰되었습니다. Pareto optimization, constrained RL, adaptive reward weighting 또는 multi-objective RL을 적용할 수 있습니다.

4. **No experimental validation**  
   생성 후보는 구조적 유효성만 확인되었으며, docking, selectivity panel, ADMET assay, 합성 및 생물학적 실험 검증이 수행되지 않았습니다.

## Acknowledgements

This project was built with PyTorch, Hugging Face Transformers, NovoMolGen, PyTDC, ESM-2, RDKit, scikit-learn, and the KIBA/Tox21 datasets.

## Disclaimer

본 프로젝트는 교육 및 연구 목적의 계산 실험입니다. 생성된 분자와 예측 결과를 의학적 조언, 임상적 의사결정 또는 실제 약물 후보로 직접 사용해서는 안 됩니다.
