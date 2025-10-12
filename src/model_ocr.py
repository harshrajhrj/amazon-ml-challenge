import torch
import torch.nn as nn
from transformers import AutoModel

class TriModalFusion(nn.Module):
    def __init__(self, text_dim=768, img_dim=768, hidden_dim=512):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.img_proj  = nn.Linear(img_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 1)
        )
    def forward(self, text_emb, img_emb, ocr_emb):
        t = self.text_proj(text_emb).unsqueeze(1)
        i = self.img_proj(img_emb).unsqueeze(1)
        o = self.text_proj(ocr_emb).unsqueeze(1)
        all_inputs = torch.cat([t, i, o], dim=1)
        fused, _ = self.cross_attn(all_inputs, all_inputs, all_inputs)
        pooled = fused.mean(dim=1)
        out = self.regressor(pooled)
        return out.squeeze(-1)

class MultiModalOCR(nn.Module):
    def __init__(self, text_model, img_model, hidden_dim=512, checkpointing=False):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(text_model)
        self.img_encoder  = AutoModel.from_pretrained(img_model)
        self.ocr_encoder  = AutoModel.from_pretrained(text_model)
        if checkpointing:
            self.text_encoder.gradient_checkpointing_enable()
            self.img_encoder.gradient_checkpointing_enable()
            self.ocr_encoder.gradient_checkpointing_enable()
        self.fusion = TriModalFusion(self.text_encoder.config.hidden_size, self.img_encoder.config.hidden_size, hidden_dim)
    def forward(self, cat_inputs, img_inputs, ocr_inputs):
        # Filter supported keys per encoder (token_type_ids may not exist for some tokenizers)
        cat_inputs = {k.replace('cat_', ''): v for k, v in cat_inputs.items()}
        ocr_inputs = {k.replace('ocr_', ''): v for k, v in ocr_inputs.items()}
        t_emb = self.text_encoder(**{k: v for k, v in cat_inputs.items() if k in ['input_ids','attention_mask','token_type_ids'] and k in cat_inputs}).last_hidden_state[:,0,:]
        v_emb = self.img_encoder(**{k: v for k, v in img_inputs.items() if k in ['pixel_values']}).pooler_output
        o_emb = self.ocr_encoder(**{k: v for k, v in ocr_inputs.items() if k in ['input_ids','attention_mask','token_type_ids'] and k in ocr_inputs}).last_hidden_state[:,0,:]
        price_log = self.fusion(t_emb, v_emb, o_emb)
        return price_log