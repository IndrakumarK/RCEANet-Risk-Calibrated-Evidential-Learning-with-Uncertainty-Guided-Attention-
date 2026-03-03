# RCEANet (Risk-Calibrated Evidential Attention Network)

Official research-grade implementation of **RCEANet**, a reliability-centered deep learning framework for calibrated and uncertainty-aware brain tumor MRI classification.

RCEANet integrates:

- Dirichlet-based evidential learning
- Calibration-aware multi-objective optimization
- Uncertainty-guided spatial attention
- Risk-derived reliability scoring
- Selective prediction (abstention)
- Distribution shift robustness evaluation

The framework is designed to support deployment-aware medical imaging informatics and radiologist-in-the-loop clinical decision-support systems.

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt

# Train the model
python scripts/train.py

# Evaluate a trained model
python scripts/evaluate.py --model checkpoints/best_model.pth

# Run distribution shift experiments
python scripts/run_shift_experiment.py

# Run ablation study
python scripts/run_ablation.py
```

---

## 📁 Project Structure

```
RCEANet/
├── configs/
│   ├── dataset1.yaml
│   ├── dataset2.yaml
│   └── dataset3.yaml
│
├── data/
│   ├── dataset_loader.py
│   └── transforms.py
│
├── models/
│   ├── backbone.py
│   ├── attention.py
│   ├── evidential_head.py
│   └── rceanet.py
│
├── losses/
│   ├── evidential_loss.py
│   ├── kl_regularization.py
│   ├── calibration_loss.py
│   └── attention_alignment_loss.py
│
├── evaluation/
│   ├── metrics.py
│   ├── calibration.py
│   ├── reliability.py
│   ├── risk_coverage.py
│   └── uncertainty_analysis.py
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── run_shift_experiment.py
│   └── run_ablation.py
│
├── checkpoints/
│
├── requirements.txt
└── README.md
```

---

## 🧠 Training

Training includes:

- Multi-objective evidential optimization
- KL regularization for uncertainty control
- Calibration-aware loss integration
- Attention–uncertainty alignment
- Validation-based checkpointing
- Deterministic seed handling

The best-performing model (based on validation Expected Calibration Error) is saved to:

```
checkpoints/best_model.pth
```

---

## 📊 Evaluation

Evaluation reports:

- Accuracy
- Precision / Recall / F1-score
- AUC
- Expected Calibration Error (ECE)
- Brier Score
- Negative Log-Likelihood (NLL)
- Uncertainty–error correlation
- Risk–coverage analysis

---

## 🔁 Distribution Shift Robustness

The framework evaluates zero-shot robustness under:

- Additive Gaussian noise
- Intensity scaling

Metrics analyzed:

- Accuracy degradation
- Calibration stability (ECE under shift)
- Controlled uncertainty escalation

No retraining is performed during shift evaluation.

---

## 🧪 Ablation Study

The following configurations are supported:

- Full RCEANet
- Without KL regularization
- Without calibration loss
- Without attention–uncertainty alignment

---

## 🔁 Reproducibility

To reproduce experimental results:

1. Install dependencies
2. Prepare datasets according to manuscript protocol
3. Train model
4. Evaluate model
5. Run shift experiments
6. Run ablation study

Random seeds are fixed to ensure deterministic behavior.

---

## 📂 Datasets

RCEANet was evaluated on three publicly available brain MRI datasets from Kaggle.
Datasets are not included in this repository and must be downloaded separately.

Dataset 1 — Brain Tumor MRI Dataset

Source:
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Description:
Multi-class brain tumor MRI dataset containing glioma, meningioma, pituitary tumor, and no-tumor classes. Images are organized into training and testing folders.

Dataset 2 — Brain Tumor Classification (MRI)

Source:
https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri

Description:
Curated brain MRI dataset for tumor classification with labeled categories. Widely used for benchmarking deep learning models in medical image analysis.

Dataset 3 — Brain MRI Scans for Tumor Classification

Source:
https://www.kaggle.com/datasets/shreyag1103/brain-mri-scans-for-brain-tumor-classification

Description:
Multi-class MRI dataset designed for supervised tumor classification experiments and generalization analysis.

📁 Expected Directory Structure

After downloading and extracting the datasets, organize them as follows:

data/
├── dataset1/
│   ├── train/
│   └── test/
├── dataset2/
│   ├── train/
│   └── test/
└── dataset3/
    ├── train/
    └── test/

Each dataset should follow a class-wise folder structure:

train/
├── glioma/
├── meningioma/
├── pituitary/
└── no_tumor/

---

## 🧠 Imaging Informatics Perspective

RCEANet shifts brain tumor MRI classification from purely accuracy-driven modeling toward a reliability-centered paradigm by:

- Embedding epistemic uncertainty into representation learning
- Aligning predictive confidence with empirical accuracy
- Supporting selective prediction for clinician handoff
- Enabling calibration-aware deployment in clinical workflows

---

## 📄 Citation

If you use this code in your research, please cite:

```
@article{RCEANet2026,
  title={Risk-Calibrated Evidential Attention Network for Reliable Brain Tumor MRI Classification},
  author={Indrakumar K, Ravikumar M.},
  journal={Journal of Imaging Informatics in Medicine},
  year={2026}
}
```

---

## 📌 Notes

- This repository is intended for academic and research use.
- The framework is modular and extendable to other medical image classification tasks.
