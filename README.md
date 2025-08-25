# 🧠 YOLO-Guided Face Super-Resolution

This repository presents a lightweight U-Net architecture for **face super-resolution**, reconstructing high-quality $128 \times 128$ images from extremely low-resolution inputs of size $16 \times 16$. The method leverages **YOLO-based attention heatmaps** and a composite **perceptual loss** to enhance reconstruction fidelity.

This work builds upon the degradation model and training protocol introduced by Kim et al. $[arXiv:2103.07039](https://arxiv.org/abs/2103.07039)$.

***

## 🚀 Key Features

- **YOLO-Based Attention Maps**
Attention maps are generated via [YOLO](https://github.com/WongKinYiu/yolov7) detections of facial landmarks (eyes, nose, mouth), circumventing the need for a pretrained Facial Alignment Network (FAN).
    - Eliminates dependency on external alignment networks
    - Easily adaptable to other domains by customizing YOLO classes
    - Decouples attention map generation from the training pipeline, simplifying implementation and reducing overhead
- **Efficient Lightweight Architecture**
The U-Net backbone employs depthwise-separable convolutions in place of standard convolutional layers, substantially reducing model parameters and accelerating training without sacrificing performance.
- **Composite Multi-Loss Objective**
Training utilizes a configurable combination of loss components:
    - Pixel-wise Mean Squared Error (MSE) loss
    - Perceptual loss based on pretrained VGG features
    - YOLO-guided attention loss to focus model learning on key facial regions
- **Training Visualization Tools**
Performance is monitored through qualitative visual comparisons on a fixed validation image, showing model outputs side-by-side across epochs, facilitating interpretability and debugging.

***

## 📂 Dataset Description

Experiments employ the **CelebA** dataset, downsampled to replicate the degradation model described in Kim et al.’s “FAN: Feature-Aware Normalization for Super-Resolution of Face Images” $[arXiv:2103.07039](https://arxiv.org/abs/2103.07039)$.

- Ground truth images: $128 \times 128$ pixels
- Inputs: $16 \times 16$ pixels (downsampled as Kim et al for direct comparison)
- YOLO-generated facial heatmaps are precomputed and stored in an `.h5` file to accelerate training.

Download the precomputed heatmaps from [HuggingFace](https://huggingface.co/datasets/RiccardoCarraro/heatmaps) using:

```bash
curl -L -o heatmaps.h5 https://huggingface.co/datasets/RiccardoCarraro/heatmaps/resolve/main/heatmaps_30k.h5
```

or a subset with 10,000 heatmaps:

```bash
curl -L -o heatmaps_10k.h5 https://huggingface.co/datasets/RiccardoCarraro/heatmaps/resolve/main/heatmaps_10k.h5
```


***

## ⚖️ Models Compared

The following variants are implemented and evaluated under identical training conditions:


| Model Variant | Pixel Loss (MSE) | Perceptual Loss (VGG) | Attention Loss (YOLO-guided) |
| :-- | :--: | :--: | :--: |
| Baseline (MSE + Perceptual) | ✓ | ✓ | ✗ |
| Heatmap-Guided (MSE + Perceptual + Heatmap Loss) | ✓ | ✓ | ✓ |
| Multiscale Loss (MSE + Perceptual + Heatmap Loss) | ✓ | ✓ | ✓ |
| Deep Supervision (MSE + Perceptual + Heatmap Loss) | ✓ | ✓ | ✓ |

All models are trained from scratch and evaluated quantitatively and qualitatively to assess the impact of YOLO-guided attention supervision on reconstruction quality.
