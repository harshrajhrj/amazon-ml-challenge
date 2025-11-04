import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoImageProcessor
from src.data_ocr import PricingDatasetOCR
from src.model_ocr_ultra import MultiModalOCRUltra
from torch import serialization as torch_serial

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description="Memory-efficient ensemble predictions from multiple checkpoints")
    ap.add_argument("--test_csv", type=str, required=True)
    ap.add_argument("--image_root", type=str, required=True)
    ap.add_argument("--ckpts", nargs="+", required=True, help="List of .pt checkpoint paths")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--max_len", type=int, default=384)
    ap.add_argument("--out_csv", type=str, default="dataset/test_out_ensemble.csv")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Fix for PyTorch 2.6 ---
    torch_serial.add_safe_globals([np.core.multiarray.scalar, np.dtype])

    # --- Load config from first ckpt ---
    first_ckpt = torch.load(args.ckpts[0], map_location="cpu", weights_only=False)
    text_model = first_ckpt.get("text_model", "Qwen/Qwen2.5-0.5B-Instruct")
    img_model  = first_ckpt.get("img_model",  "openai/clip-vit-large-patch14")
    print(f"[INFO] Using text_model={text_model}")
    print(f"[INFO] Using img_model={img_model}")

    tok = AutoTokenizer.from_pretrained(text_model, trust_remote_code=True)
    img_proc = AutoImageProcessor.from_pretrained(img_model)

    # --- Dataset ---
    df = pd.read_csv(args.test_csv)
    ds = PricingDatasetOCR(df, args.image_root, tok, img_proc,
                           max_len=args.max_len, train=False)
    loader = DataLoader(ds, batch_size=args.bs, shuffle=False,
                        num_workers=8, pin_memory=True)

    # --- Preload checkpoints ON CPU ---
    ckpt_states = []
    for path in args.ckpts:
        print(f"[INFO] Loading checkpoint to CPU: {path}")
        ckpt_states.append(torch.load(path, map_location="cpu", weights_only=False))

    preds_all, ids_all = [], []

    for batch_idx, batch in enumerate(loader):
        cat_inputs = {k: v.to(device) for k, v in batch.items() if k.startswith("cat_")}
        ocr_inputs = {k: v.to(device) for k, v in batch.items() if k.startswith("ocr_")}
        img_inputs = {k: v.to(device) for k, v in batch.items() if k == "pixel_values"}

        batch_preds = []

        # --- Evaluate one model at a time (free GPU memory after each) ---
        for ckpt in ckpt_states:
            m = MultiModalOCRUltra(text_model, img_model,
                                   hidden_dim=1024, checkpointing=False).to(device)
            m.load_state_dict(ckpt["model"], strict=False)
            m.eval()

            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                log_price = m(cat_inputs, img_inputs, ocr_inputs)
                pred = torch.expm1(log_price).clamp(min=1.0)
            batch_preds.append(pred.detach().cpu().numpy())

            # --- Free GPU memory ---
            del m, log_price, pred
            torch.cuda.empty_cache()

        # --- Ensemble average on CPU ---
        batch_preds = np.stack(batch_preds, axis=0).mean(axis=0)
        preds_all.append(batch_preds)
        ids_all.append(batch["sample_id"].cpu().numpy())

        if (batch_idx + 1) % 50 == 0:
            print(f"[INFO] Processed {batch_idx+1}/{len(loader)} batches")

    # --- Save predictions ---
    preds_all = np.concatenate(preds_all)
    ids_all = np.concatenate(ids_all)
    out = pd.DataFrame({"sample_id": ids_all, "price": np.round(preds_all, 2)})
    out = out.sort_values("sample_id").reset_index(drop=True)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"✅ Saved ensemble predictions → {args.out_csv} | total={len(out)}")

# --------------------------------------------------------------
if __name__ == "__main__":
    main()
