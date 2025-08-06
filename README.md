# 🧠 YOLO-Guided Face Super-Resolution

This project proposes a lightweight U-Net architecture for **face super-resolution** from 16×16 to 128×128, enhanced through **YOLO-based attention maps**, **perceptual loss**, and **adversarial training**.

Donwload the `.h5` file with the heatmaps from [HuggingFace](https://huggingface.co/datasets/RiccardoCarraro/heatmaps) with the following commands:
- `!curl -L -o heatmaps.h5 https://huggingface.co/datasets/RiccardoCarraro/heatmaps/resolve/main/heatmaps.h5` (`.h5` file with full set of 50k heatmaps)
- or `!curl -L -o heatmaps.h5 https://huggingface.co/datasets/RiccardoCarraro/heatmaps/resolve/main/heatmaps_10k.h5` (with just 10k heatmaps)

---

## ✨ Key Features

- **🔍 YOLO-Based Attention (No FAN Needed)**  
  We generate heatmaps using [YOLO](https://github.com/WongKinYiu/yolov7) detections (eyes, nose, mouth), avoiding the need for a pre-trained Facial Alignment Network (FAN).  
  - No extra pretraining required  
  - Works out of the box on other domains by changing YOLO classes  
  - Decouples attention generation from training pipeline

- **⚡ Lightweight Architecture**  
  Our U-Net replaces standard convolutions with **depthwise-separable convolutions**, resulting in significantly fewer parameters and faster training.

- **🎯 Loss Composition**  
  Configurable multi-loss setup:  
  - `pixel` loss (MSE)  
  - `perceptual` loss (VGG-based)  
  - `attention` loss (YOLO-guided)  
  - `adversarial` loss (PatchGAN)

- **🖼️ Training Visualization**  
  During training, the models are compared side-by-side on a fixed validation image across epochs.

---

## 🗂️ Dataset

We use the **CelebA** dataset downsampled to match the blur level in *Kim et al., "FAN: Feature-Aware Normalization for Super-Resolution of Face Images"*.  
Ground truth images are 128×128, and inputs are bicubically downscaled to 16×16. YOLO-derived heatmaps are precomputed and reused during training.

---

## 📊 Models Compared

We train and compare the following:

| Model Name            | Pixel | Perceptual | Attention | Adversarial |
|-----------------------|:-----:|:----------:|:---------:|:-----------:|
| Pixel + Perceptual    | ✅    | ✅         | ❌        | ❌          |
| + Attention           | ✅    | ✅         | ✅        | ❌          |
| + GAN                 | ✅    | ✅         | ❌        | ✅          |
| Full (ours)           | ✅    | ✅         | ✅        | ✅          |

All four models are trained from scratch using the same settings and evaluated on identical validation batches.

---

📎 Notes
Training is fast even on a mid-range GPU (<2GB per model).
Models generalize well across faces, even under severe blur.

Easily extensible to other domains (e.g., hands, animals) by changing YOLO classes.
