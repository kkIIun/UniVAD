# UniVAD: Unified Video Anomaly Detection (WACV2026) - Official Repository

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)
[![Conference: WACV 2026](https://img.shields.io/badge/WACV-2026-blue.svg)](http://wacv2026.thecvf.com/)

**UniVAD** is a unified framework for video anomaly detection that categorizes anomalies into three types and employs a tri-branch autoencoder architecture to detect them simultaneously. This repository contains the official implementation of the paper **"UniVAD: Unified Video Anomaly Detection"**, accepted at **WACV 2026**.

---

## 📖 Overview

UniVAD addresses the challenge of detecting diverse types of anomalies in video surveillance by:

1. **Categorizing anomalies** into three distinct types:
   - **Human-related Anomaly:** Abnormal human behaviors (fighting, falling, etc.)
   - **Object-related Anomaly:** Object-based anomalies (throwing objects, prohibited activities)
   - **Object-independent Anomaly:** Contextual/environmental deviations

2. **Tri-Branch Architecture:**
   - **Skeleton Branch:** Processes human pose data for motion pattern analysis
   - **Local Visual Branch:** Analyzes object-level appearance features
   - **Global Visual Branch:** Captures frame-level contextual information

---

## 🏗️ Repository Structure

```
UniVAD/
├── skeleton/          # Skeleton-based detection (STGCN, STAE)
├── local-visual/      # Local visual features (AutoEncoder)
├── global-visual/     # Global visual features (AutoEncoder)
├── final_score/       # Dataset results (nwpu, shanghai, ubnormal)
├── calc_total_score.py # Score aggregation
└── requirements.txt    # Dependencies
```

---

## 🚀 Key Features

- **Multi-modal Processing:** Combines skeleton, local visual, and global visual features
- **Predictive Learning:** Uses past and future frame prediction for anomaly detection
- **Flexible Architecture:** Each branch can be trained independently
- **Comprehensive Evaluation:** Supports multiple benchmark datasets (ShanghaiTech, NWPUCampus, UBnormal)

---

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- PyTorch 2.1.0+
- CUDA (for GPU acceleration)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/kkIIun/UniVAD.git
cd UniVAD

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install PyTorch (if not already installed)
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0
```

---

## 📊 Datasets

UniVAD supports the following benchmark datasets:

- **ShanghaiTech Campus Dataset**
- **NWPUCampus Dataset** 
- **UBnormal Dataset**

Each dataset should be preprocessed to extract:
- Skeleton features (pose keypoints)
- Local visual features (object-level embeddings)
- Global visual features (frame-level embeddings)

---

## 🏃‍♂️ Usage

### Training Individual Branches

#### 1. Skeleton Branch
```bash
cd skeleton
bash train_skeleton.sh
```

#### 2. Local Visual Branch
```bash
cd local-visual
bash train_local_visual.sh
```

#### 3. Global Visual Branch
```bash
cd global-visual
bash train_global_visual.sh
```

### Training All Branches
```bash
bash train_all.sh
```

### Evaluation and Score Aggregation

After training all three branches, aggregate the scores:

```bash
python calc_total_score.py
```

This will compute the final anomaly scores using weighted combination:
- Skeleton score weight (α): 1.0
- Local visual score weight (β): 0.1  
- Global visual score weight (γ): 0.1

---

## 📈 Performance

UniVAD achieves state-of-the-art performance on multiple anomaly detection benchmarks:

| Dataset      | Metric | Ours  | Ours† |
|--------------|--------|-------|-------|
| ShanghaiTech | Micro  | 89.3  | 89.5  |
|              | Macro  | 91.5  | 91.6  |
| UBnormal     | Micro  | 79.3  | 82.7  |
|              | Macro  | 90.1  | 91.4  |
| NWPUCampus   | Micro  | 72.2  | 73.4  |
|              | Macro  | 87.6  | 87.5  |

*Ours† denotes results with additional data augmentation.

---

## 🤝 Citation

If you use this code for your research, please cite:

```bibtex
@inproceedings{univad2026,
  title={UniVAD: Unified Video Anomaly Detection},
  author={[Author Names]},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year={2026}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Built with PyTorch and PyTorch Lightning
- Uses OpenCV for video processing
- Skeleton processing inspired by STGCN and STAE models