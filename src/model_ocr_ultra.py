# src/model_ocr_ultra.py
import torch
import torch.nn as nn
from typing import Dict, Any

from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    CLIPVisionModel,
)


# ---------- small utilities ----------

def masked_mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Computes mean pooling over valid tokens.
    last_hidden_state: [B, T, D]
    attention_mask:    [B, T] (1 for valid tokens)
    returns: [B, D]
    """
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # [B,T,1]
    summed = (last_hidden_state * mask).sum(dim=1)                  # [B,D]
    denom = mask.sum(dim=1).clamp(min=1e-6)                         # [B,1]
    return summed / denom


class SafeGradCheckpointMixin:
    @staticmethod
    def enable_checkpointing_if_supported(hf_model):
        try:
            if hasattr(hf_model, "gradient_checkpointing_enable"):
                hf_model.gradient_checkpointing_enable()
        except Exception:
            pass


# ---------- text backbone (Qwen2.5) ----------

class QwenTextBackbone(nn.Module, SafeGradCheckpointMixin):
    """
    A robust wrapper that works with Qwen2.5-* models whether they expose AutoModel
    or only AutoModelForCausalLM. It returns a pooled text embedding via masked mean.
    """
    def __init__(self, model_name: str, checkpointing: bool = True):
        super().__init__()
        self.model_name = model_name

        base = None
        self.is_causal_lm = False

        # Try AutoModel first (preferred, returns last_hidden_state directly)
        try:
            base = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        except Exception:
            # Fall back to CausalLM variant
            base = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
            self.is_causal_lm = True

        if checkpointing:
            self.enable_checkpointing_if_supported(base)

        self.backbone = base
        # hidden size discovery
        if hasattr(base.config, "hidden_size"):
            self.hidden_size = base.config.hidden_size
        elif hasattr(base.config, "n_embd"):
            self.hidden_size = base.config.n_embd
        else:
            # Safe default
            self.hidden_size = 1024

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.is_causal_lm:
            # For CausalLM, request hidden states explicitly
            out = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False
            )
            last_hidden = out.hidden_states[-1]  # [B,T,D]
        else:
            out = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False
            )
            # Some models return .last_hidden_state, some wrap it under .base_model_output
            last_hidden = getattr(out, "last_hidden_state", None)
            if last_hidden is None and hasattr(out, "hidden_states") and out.hidden_states:
                last_hidden = out.hidden_states[-1]

        if last_hidden is None:
            raise RuntimeError("Failed to obtain last_hidden_state from Qwen backbone outputs.")

        pooled = masked_mean_pool(last_hidden, attention_mask)  # [B,D]
        return pooled  # [B, hidden_size]


# ---------- image backbone (CLIP ViT-L/14) ----------

class ClipVisionBackbone(nn.Module, SafeGradCheckpointMixin):
    """
    CLIP vision tower wrapper that returns a pooled image embedding.
    """
    def __init__(self, model_name: str, checkpointing: bool = True):
        super().__init__()
        self.model = CLIPVisionModel.from_pretrained(model_name)
        if checkpointing:
            self.enable_checkpointing_if_supported(self.model)

        # Hidden size discovery
        if hasattr(self.model.config, "hidden_size"):
            self.hidden_size = self.model.config.hidden_size
        else:
            self.hidden_size = 1024  # ViT-L/14 default

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.model(pixel_values=pixel_values)
        # Prefer pooled output (projection of CLS)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output  # [B, D]
        # fallback: mean pool tokens
        last_hidden = out.last_hidden_state  # [B, N, D]
        return last_hidden.mean(dim=1)


# ---------- gated cross-attention fusion ----------

class GatedCrossAttentionFusion(nn.Module):
    """
    Projects text/image/ocr to a shared space, performs cross-attention, and
    learns a gate over modality contributions. Output is regressed to 1D (log-price).
    """
    def __init__(self, text_dim: int, img_dim: int, hidden_dim: int = 1024):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.img_proj  = nn.Linear(img_dim,  hidden_dim)
        self.ocr_proj  = nn.Linear(text_dim, hidden_dim)

        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True)

        self.fuse_ln = nn.LayerNorm(hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.Sigmoid()
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, t_vec: torch.Tensor, v_vec: torch.Tensor, o_vec: torch.Tensor) -> torch.Tensor:
        # project
        t = self.text_proj(t_vec).unsqueeze(1)  # [B,1,H]
        v = self.img_proj(v_vec).unsqueeze(1)   # [B,1,H]
        o = self.ocr_proj(o_vec).unsqueeze(1)   # [B,1,H]

        # cross-attention over concatenated sequence
        seq = torch.cat([t, v, o], dim=1)       # [B,3,H]
        fused, _ = self.cross_attn(seq, seq, seq)  # [B,3,H]
        fused = self.fuse_ln(fused.mean(dim=1))    # [B,H]

        # gating over modalities (residual style)
        g_in = torch.cat([t.squeeze(1), v.squeeze(1), o.squeeze(1)], dim=-1)  # [B,3H]
        g = self.gate(g_in)                                                    # [B,H]
        fused = fused * g + fused

        return self.head(fused).squeeze(-1)  # [B]


# ---------- main multimodal model ----------

class MultiModalOCRUltra(nn.Module):
    """
    Tri-modal regressor:
      - Text branch (catalog): Qwen2.5-0.5B-Instruct
      - OCR  branch (text):    Qwen2.5-0.5B-Instruct
      - Image branch:          CLIP ViT-L/14
      - Fusion:                Gated cross-attn + MLP head
    Predicts log(price) directly.
    """
    def __init__(
        self,
        text_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
        img_model: str  = "openai/clip-vit-large-patch14",
        hidden_dim: int = 1024,
        checkpointing: bool = True
    ):
        super().__init__()
        # backbones
        self.text_encoder = QwenTextBackbone(text_model, checkpointing=checkpointing)
        self.ocr_encoder  = QwenTextBackbone(text_model,  checkpointing=checkpointing)
        self.img_encoder  = ClipVisionBackbone(img_model, checkpointing=checkpointing)

        # fusion
        self.fusion = GatedCrossAttentionFusion(
            text_dim=self.text_encoder.hidden_size,
            img_dim=self.img_encoder.hidden_size,
            hidden_dim=hidden_dim
        )

        # cache config sizes (useful for schedulers/optim groups outside)
        self.text_hidden = self.text_encoder.hidden_size
        self.img_hidden  = self.img_encoder.hidden_size
        self.hidden_dim  = hidden_dim

    def _pick(self, d: Dict[str, torch.Tensor], keys) -> Dict[str, torch.Tensor]:
        return {k: d[k] for k in keys if k in d}

    def forward(self, cat_inputs: Dict[str, Any], img_inputs: Dict[str, Any], ocr_inputs: Dict[str, Any]) -> torch.Tensor:
        """
        Inputs follow your dataset prefixes:
          - cat_*  → catalog text tokenizer outputs
          - ocr_*  → OCR text tokenizer outputs
          - pixel_values → CLIP image tensor
        """
        # strip prefixes for huggingface models
        cat_inputs = {k.replace('cat_', ''): v for k, v in cat_inputs.items()}
        ocr_inputs = {k.replace('ocr_', ''): v for k, v in ocr_inputs.items()}

        # ensure required keys exist
        cat_ids  = cat_inputs.get("input_ids")
        cat_mask = cat_inputs.get("attention_mask")
        ocr_ids  = ocr_inputs.get("input_ids")
        ocr_mask = ocr_inputs.get("attention_mask")
        pix      = img_inputs.get("pixel_values")

        if cat_ids is None or cat_mask is None:
            raise ValueError("Catalog inputs must include 'cat_input_ids' and 'cat_attention_mask'.")
        if ocr_ids is None or ocr_mask is None:
            raise ValueError("OCR inputs must include 'ocr_input_ids' and 'ocr_attention_mask'.")
        if pix is None:
            raise ValueError("Image inputs must include 'pixel_values' (CLIP preprocessed).")

        # forward each branch
        t_vec = self.text_encoder(cat_ids, cat_mask)    # [B, Dt]
        o_vec = self.ocr_encoder(ocr_ids, ocr_mask)     # [B, Dt]
        v_vec = self.img_encoder(pix)                   # [B, Dv]

        # fuse & regress log-price
        price_log = self.fusion(t_vec, v_vec, o_vec)    # [B]
        return price_log
