import torch
import torch.nn as nn
from transformers import AutoModel

class GatedFusion(nn.Module):
    def __init__(self, text_dim, img_dim, hidden_dim=768):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.img_proj  = nn.Linear(img_dim, hidden_dim)
        self.ocr_proj  = nn.Linear(text_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.Sigmoid()
        )
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, t_emb, v_emb, o_emb):
        t = self.text_proj(t_emb).unsqueeze(1)
        v = self.img_proj(v_emb).unsqueeze(1)
        o = self.ocr_proj(o_emb).unsqueeze(1)
        all_inputs = torch.cat([t, v, o], dim=1)
        fused, _ = self.cross_attn(all_inputs, all_inputs, all_inputs)
        fused_mean = fused.mean(dim=1)
        gate_in = torch.cat([t.squeeze(1), v.squeeze(1), o.squeeze(1)], dim=-1)
        g = self.gate(gate_in)
        fused_out = fused_mean * g + fused_mean
        return self.mlp(fused_out).squeeze(-1)

class MultiModalOCRLarge(nn.Module):
    def __init__(self, text_model, img_model, hidden_dim=768, checkpointing=True):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(text_model)
        self.ocr_encoder  = AutoModel.from_pretrained(text_model)
        self.img_encoder  = AutoModel.from_pretrained(img_model)
        if checkpointing:
            self.text_encoder.gradient_checkpointing_enable()
            self.ocr_encoder.gradient_checkpointing_enable()
            self.img_encoder.gradient_checkpointing_enable()
        self.fusion = GatedFusion(
            text_dim=self.text_encoder.config.hidden_size,
            img_dim=self.img_encoder.config.hidden_size,
            hidden_dim=hidden_dim,
        )

    def forward(self, cat_inputs, img_inputs, ocr_inputs):
        cat_inputs = {k.replace('cat_', ''): v for k, v in cat_inputs.items()}
        ocr_inputs = {k.replace('ocr_', ''): v for k, v in ocr_inputs.items()}

        t_emb = self.text_encoder(
            **{k: v for k, v in cat_inputs.items() if k in ['input_ids', 'attention_mask']}
        ).last_hidden_state[:, 0, :]

        v_out = self.img_encoder(
            **{k: v for k, v in img_inputs.items() if k == 'pixel_values'}
        )
        if hasattr(v_out, "pooler_output") and v_out.pooler_output is not None:
            v_emb = v_out.pooler_output
        else:
            v_emb = v_out.last_hidden_state[:, 0, :]

        o_emb = self.ocr_encoder(
            **{k: v for k, v in ocr_inputs.items() if k in ['input_ids', 'attention_mask']}
        ).last_hidden_state[:, 0, :]

        return self.fusion(t_emb, v_emb, o_emb)

