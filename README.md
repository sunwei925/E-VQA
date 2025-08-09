# E-VQA: Efficient Video Quality Assessment

<div align="center">

![visitors](https://visitor-badge.laobi.icu/badge?page_id=sunwei925/E-VQA)
[![GitHub stars](https://img.shields.io/github/stars/sunwei925/E-VQA)](https://github.com/sunwei925/E-VQA)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13%2B-brightgreen?logo=PyTorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/sunwei925/E-VQA)
[![arXiv](https://img.shields.io/badge/arXiv-CVPR%202025-brightgreen)](https://openaccess.thecvf.com/content/CVPR2025W/NTIRE/papers/Sun_An_Empirical_Study_for_Efficient_Video_Quality_Assessment_CVPRW_2025_paper.pdf)

</div>

## Overview

This repository contains the official implementation of **"An Empirical Study for Efficient Video Quality Assessment"**, which achieved the **third place** in the [CVPR NTIRE 2025 UGC Video Quality Assessment Challenge](https://codalab.lisn.upsaclay.fr/competitions/17638).

## 📚 Publications

- **Paper**: [An Empirical Study for Efficient Video Quality Assessment](https://openaccess.thecvf.com/content/CVPR2025W/NTIRE/papers/Sun_An_Empirical_Study_for_Efficient_Video_Quality_Assessment_CVPRW_2025_paper.pdf) (CVPRW 2025)
- **Challenge Report**: [NTIRE 2025 Challenge on Short-form UGC Video Quality Assessment](https://openaccess.thecvf.com/content/CVPR2025W/NTIRE/papers/A_Li_NTIRE_2025_Challenge_on_Short-form_UGC_Video_Quality_Assessment_and_CVPRW_2025_paper.pdf)

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 1.13+
- CUDA-compatible GPU (recommended)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/sunwei925/E-VQA.git
   cd E-VQA
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   > **Note**: This implementation is compatible with the FastVQA environment. We also provide our exported environment configuration.

## 📦 Data Preparation

### Knowledge Distillation Database

Download the Knowledge Distillation database from [Baidu Drive](https://pan.baidu.com/share/init?surl=5gsNZXTDi9kVJn_vcF7LxA) (extraction code: `buhz`).

### Pre-trained Models

Download our trained model (final submission version) from [Google Drive](https://drive.google.com/file/d/129SIkg03Ov4YDsBLjVAbL9whhPHH-vT1/view?usp=share_link) and place it in the `ckpt_save` directory.

## 🏋️ Training

### Training on Knowledge Distillation Database

1. Update the data paths in the configuration files
2. Execute the training script:
   ```bash
   sh run-dvq.sh
   ```

### Training on KVQ Dataset

1. Configure the dataset paths in the shell script
2. Run the training:
   ```bash
   sh run.sh
   ```

## 🧪 Evaluation

### Testing on KVQ Test Set

Because we adopted the fragment of FastVQA, which is affected by random numbers, we provided two test codes for frame extraction (test_NTIRE_video_f.py) and video testing (test_videos.py):

#### Method 1: Frame Extraction Testing

1. Download pre-extracted frames from [Google Drive](https://drive.google.com/file/d/1uid5WL9DWlEH4eEeLyCiNbafSzIPZqMm/view?usp=sharing)
2. Modify the file path on line 52 of `test_NTIRE_video_f.py`
3. Execute the evaluation:
   ```bash
   sh test-f.sh
   ```

#### Method 2: Direct Video Testing

Configure the video directory path and run:
```bash
python test_videos.py --model_path ./ckpts_save/UVQA_0.8967.pth \
                      --video_path ./test_data.csv \
                      --video_dir <your_video_directory>
```

### Computational Complexity Analysis

To evaluate the model's computational efficiency:
```bash
python fvcore_test.py
```

## 📊 Results

Our method achieved competitive performance on the NTIRE 2025 challenge benchmark. For detailed results and comparisons, please refer to our paper and the challenge report.

## 🤝 Citation

If you find this work useful for your research, please cite our paper:

```bibtex
@inproceedings{sun2025empirical,
  title={An Empirical Study for Efficient Video Quality Assessment},
  author={Sun, Wei and Fu, Kang and Cao, Linhan and Zhu, Dandan and Zhang, Kaiwei and Zhu, Yucheng and Zhang, Zicheng and Hu, Menghan and Min, Xiongkuo and Zhai, Guangtao},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
  pages={1403--1413},
  year={2025}
}
```

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

We thank the organizers of the NTIRE 2025 UGC Video Quality Assessment Challenge for providing the benchmark and evaluation platform.

---


**⭐ Star this repository if you find it helpful!**


