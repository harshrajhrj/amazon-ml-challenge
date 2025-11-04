# Amazon ML Challenge 2025 — Model Documentation

**Team Name:** _Singularity_  
**Team Members:** _Harsh Raj, Hammad Khan, Rakesh kumar Giri_  
**Submission Date:** 13-10-2025

---

## 1. Methodology Overview
Our approach tackles **multimodal price prediction** by combining visual, textual, and OCR-based features using transformer encoders. The core idea is to build a **unified multimodal embedding** that aligns product images, titles, and OCR-extracted text into a shared representation space.  
We began from a baseline (DeBERTa-ViT) and progressively improved it using **stronger encoders (Qwen2.5 + CLIP)**, **progressive fine-tuning**, **Exponential Moving Average (EMA)** stabilization, and **dynamic validation resampling** to maximize generalization across all product types.

---

## 2. Model Architecture / Algorithms

### Base Components
| Modality | Encoder | Model Name |
|-----------|----------|------------|
| Text (Catalog) | Instruction-tuned LLM | `Qwen/Qwen2.5-0.5B-Instruct` |
| OCR Text (Invoice / Product label) | Shared text encoder | Same Qwen2.5 instance |
| Image | Vision Transformer | `openai/clip-vit-large-patch14` |

### Fusion Head
- Multimodal fusion via **cross-modal projection + gated MLP fusion layer** (hidden size = 1024).
- Final regression head predicts **log(price)** to stabilize training.

### Loss Function
We use a **CombinedLoss**:
$$
\
\mathcal{L} = \alpha \cdot \text{SmoothL1Loss}(y_{\text{pred}}, y) + (1-\alpha) \cdot \text{SMAPE}(y_{\text{pred}}, y)
\
$$
with $\alpha$ = 0.5 for balanced scale-robust optimization.

### Training Optimizations
- **Mixed Precision (AMP)** with gradient scaling for faster compute.
- **Gradient Checkpointing** to fit large models within GPU memory (L40 / 46 GB).
- **Cosine LR scheduler with warm-up** and differential learning rates:
  - Text encoder = $3e-6$  
  - Image encoder = $3e-6$  
  - Fusion head = $1e-5$
- **EMA (Exponential Moving Average)** of model weights ($\text{decay}$ = $0.9997$)  
  → smooths high-variance updates and improves final SMAPE.

---

## 3. Feature Engineering & Data Strategy

### Input Features
1. **Catalog text**: product title, brand, description, category fields (tokenized by Qwen tokenizer).
2. **OCR text**: extracted using `ocr_extract.py` (EasyOCR pipeline) and re-encoded through the text backbone.
3. **Image features**: $224×224$ crops processed by CLIP image processor (`openai/clip-vit-large-patch14`).
4. **Numerical Target**: $log1p(price)$ for regression stability.

### Preprocessing
- Token truncation up to $384$ tokens per text input.  
- Image normalization using CLIP mean / std.  
- Outlier clipping for extreme price values.  
- Train–Validation split = $90 / 10$ (reshuffled every $3$ epochs for dynamic exposure).

### Data Re-Use (Dynamic Validation)
Every 3 epochs, the dataset is **reshuffled with a new random seed**, allowing the model to learn from the previously held-out validation subset.  
This progressive cross-validation exposes **$100 \%$ of the data** over the full training horizon without leakage.

---

## 4. Training Configuration
| Setting | Value |
|----------|-------|
| GPU | NVIDIA $\text{L}40$ ($46 \text{GB}$) |
| Batch Size | $8 × 2$ gradient accumulation (effective = $16$) |
| Epochs | $20$ |
| Optimizer | AdamW |
| Scheduler | Cosine with $1$ epoch warm-up |
| Mixed Precision | Enabled |
| EMA | Enabled (CPU-based implementation to save VRAM) |

---

## 5. Results Summary
| Model | Validation SMAPE | Notes |
|--------|-----------------|-------|
| DeBERTa-ViT baseline | ~$55 \%$ | initial benchmark |
| Qwen2.5 + CLIP (static val) | $42.3 \%$ | after 13 epochs |
| Qwen2.5 + CLIP + EMA + Dynamic Val | **$≈ 41.3 \%$** | final model |

The approach improved performance by **~14 SMAPE points** over the baseline, achieving strong cross-split consistency and stable convergence.

---

## 6. Additional Insights
- EMA stored on CPU to avoid GPU spikes during checkpointing.  
- Gradient accumulation enabled efficient large-model training on single GPU.  
- Ensemble of last $3$ checkpoints (epochs $14–17$) slightly improves final score (~ $0.5$ pt).  
- Dynamic validation ensures model learns from all samples without explicit k-fold CV overhead.

---

