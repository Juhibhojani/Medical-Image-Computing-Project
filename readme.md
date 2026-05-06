# Knowledge Distillation for Medical Imaging

This project is done as part of the course `CS:736 Medical Image Computing` at `IIT Bombay`. It includes recreation of results by the paper [HDKD: Hybrid Data-Efficient Knowledge Distillation Network for Medical Image Classification](https://arxiv.org/abs/2407.07516)

It includes eight different configurations tried and tested including architecture changes, changes in loss and empricial analysis.

The experiments primarily focus on understanding the behavior of hybrid CNN + Transformer distillation under limited data settings.

For more details regarding details please refer to the following presentation : [Presentation link](https://docs.google.com/presentation/d/1NQvN1-e4NLwciAYnFjDiXn2L7lDtD8n-D00jRp0x2Qs/edit?slide=id.p#slide=id.p)

## Multiple ablation study details are as follows:

- Distillation weighting
- Feature distillation
- Weighted loss function
- Reverse + Forward KL divergence
- Transformer teacher
- Multi-distill tokens
- Swin Transformer replacement
- Cross-dataset distillation


## Paper Summary

The original HDKD framework proposes:

- A CNN teacher
- A Hybrid CNN + Transformer student
- Shared CNN blocks between teacher and student
- Combination of:
    1. Logit Distillation
    2. Feature Distillation

The student learns:

- Local representations through CNN blocks
- Global representations through the Transformer block

## Datasets

The following datasets were used:

- [HAM10000](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) (ISIC 2018)
- [BCN20000](https://api.isic-archive.com/collections/249/) (ISIC 2019)

Number of classes: 7

## Ablation Studies

### 1. CLS vs Distill Token Weighting

Study of weighting combinations between CLS and Distill token logits.

#### Observation
- Best performance obtained with:
  - Distill token weight = **0.7**

---

### 2. Multiple Data Sizes + Feature Visualization

Experiments performed with:
- 350 samples
- 700 samples
- 2833 samples

Feature maps were visualized every 25 epochs to observe:
- Teacher–student alignment
- Reduction in feature MSE over training

#### Observation
- Distillation consistently improves student performance
- Student features progressively align with teacher features

---

### 3. Weighted Loss vs SMOTE

Instead of using SMOTE for imbalance handling:
- Weighted Cross Entropy Loss was used

#### Observation
- Weighted loss outperformed SMOTE
- Teacher accuracy improved by ~3.5%

---

### 4. Forward + Reverse KL Divergence

Weighted combination of:
- Forward KL (mean-seeking)
- Reverse KL (mode-seeking)

#### Observation
- Slight variations in performance
- No significant improvement over standard KD

---

### 5. Transformer-based Teacher

Teacher modified to include Transformer blocks.

#### Observation
- Transformer teacher performed worse than CNN teacher
- Likely due to:
  - Increased parameter count
  - Overfitting under limited data

---

### 6. Multi-Distill Tokens

Additional distill tokens added to learn:
- Stage-2 features
- Stage-3 features
- Teacher logits

#### Observation
- Combining:
  - Feature Distillation
  - Logit Distillation
  - Token Distillation
  produced the best results

---

### 7. Replacing ViT with Swin Transformer

DFLT block replaced with a Swin-style hierarchical transformer.

#### Observation
- Swin performs comparably or slightly better in some settings
- Better handling of locality and hierarchical features

---

### 8. Cross-Dataset Distillation

Teacher:
- Pre-trained / fine-tuned on BCN20000

Student:
- Evaluated under limited-data setup

#### Observation
- Distillation from fine-tuned teacher performs better
- Student often outperforms teacher
- Distillation acts as a regularizer and improves generalization

---