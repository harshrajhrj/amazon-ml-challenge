import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoImageProcessor
<<<<<<< HEAD
from tqdm import tqdm

from src.data_ocr import PricingDatasetOCR
from src.model_ocr_ultra import MultiModalOCRUltra
from torch import serialization as torch_serial

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description="Predict prices using a single multimodal OCR model (with progress bar)")
    ap.add_argument("--test_csv", type=str, required=True)
    ap.add_argument("--image_root", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True, help="Path to the trained model checkpoint (.pt)")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--max_len", type=int, default=384)
    ap.add_argument("--out_csv", type=str, default="dataset/test_out_single.csv")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Compatibility fix for PyTorch 2.6 ---
    torch_serial.add_safe_globals([np.core.multiarray.scalar, np.dtype])

    # --- Load checkpoint ---
    print(f"[INFO] Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)

    text_model = ckpt.get("text_model", "Qwen/Qwen2.5-0.5B-Instruct")
    img_model  = ckpt.get("img_model",  "openai/clip-vit-large-patch14")
    print(f"[INFO] Using text_model={text_model}")
    print(f"[INFO] Using img_model={img_model}")

    # --- Load processors ---
    tok = AutoTokenizer.from_pretrained(text_model, trust_remote_code=True)
    img_proc = AutoImageProcessor.from_pretrained(img_model)

    # --- Dataset ---
    df = pd.read_csv(args.test_csv)
    ds = PricingDatasetOCR(df, args.image_root, tok, img_proc,
                           max_len=args.max_len, train=False)
    loader = DataLoader(ds, batch_size=args.bs, shuffle=False,
                        num_workers=8, pin_memory=True)

    # --- Model ---
    model = MultiModalOCRUltra(text_model, img_model,
                               hidden_dim=1024, checkpointing=False).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    # --- Prediction loop with tqdm progress bar ---
    preds, ids = [], []
    pbar = tqdm(loader, total=len(loader), desc="Predicting", ncols=120)

    for batch in pbar:
        cat_inputs = {k: v.to(device) for k, v in batch.items() if k.startswith("cat_")}
        ocr_inputs = {k: v.to(device) for k, v in batch.items() if k.startswith("ocr_")}
        img_inputs = {k: v.to(device) for k, v in batch.items() if k == "pixel_values"}

        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            log_price = model(cat_inputs, img_inputs, ocr_inputs)
            pred = torch.expm1(log_price).clamp(min=1.0)

        preds.append(pred.cpu().numpy())
        ids.append(batch["sample_id"].cpu().numpy())

        # update progress bar postfix with running total
        pbar.set_postfix_str(f"done={len(preds)*args.bs}")

    preds = np.concatenate(preds)
    ids = np.concatenate(ids)

    # --- Save output ---
    out = pd.DataFrame({"sample_id": ids, "price": np.round(preds, 2)})
    out = out.sort_values("sample_id").reset_index(drop=True)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    print(f"✅ Saved predictions → {args.out_csv} | total={len(out)}")

# --------------------------------------------------------------
if __name__ == "__main__":
=======
from src.data_ocr import PricingDatasetOCR
from src.model_ocr import MultiModalOCR
import torch.serialization as torch_serial


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test_csv', type=str, default='dataset/test_ocr.csv')
    ap.add_argument('--image_root', type=str, default='data/images')
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--bs', type=int, default=128)
    ap.add_argument('--out_csv', type=str, default='dataset/test_out.csv')
    args = ap.parse_args()

    # --- PyTorch 2.6+ compatibility: allow NumPy dtype objects ---
    import numpy as np
    torch_serial.add_safe_globals([np.core.multiarray.scalar, np.dtype])

    try:
        ckpt = torch.load(args.ckpt, map_location='cpu')
    except Exception as e:
        # retry with unrestricted pickle deserialization for trusted checkpoint
        if "weights_only" in str(e) or "Float32DType" in str(e):
            print("[WARN] Retrying checkpoint load with weights_only=False (safe for trusted checkpoints).")
            ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
        else:
            raise

    # ---------------------------------------------------------------------
    # build tokenizer / processor / dataset
    text_model = ckpt['text_model']
    img_model = ckpt['img_model']

    tok = AutoTokenizer.from_pretrained(text_model)
    img_proc = AutoImageProcessor.from_pretrained(img_model)

    df = pd.read_csv(args.test_csv)
    ds = PricingDatasetOCR(df, args.image_root, tok, img_proc, max_len=256, train=False)
    loader = DataLoader(ds, batch_size=args.bs, shuffle=False, num_workers=8, pin_memory=True)

    # ---------------------------------------------------------------------
    # model + predict
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiModalOCR(text_model, img_model).to(device)
    model.load_state_dict(ckpt['model'], strict=True)
    model.eval()

    preds, sample_ids = [], []
    for batch in loader:
        cat_inputs = {k: v.to(device) for k, v in batch.items() if k.startswith('cat_')}
        ocr_inputs = {k: v.to(device) for k, v in batch.items() if k.startswith('ocr_')}
        img_inputs  = {k: v.to(device) for k, v in batch.items() if k == 'pixel_values'}

        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            pred_log = model(cat_inputs, img_inputs, ocr_inputs)
            pred = torch.clamp(torch.expm1(pred_log), min=1.0)

        preds.append(pred.cpu().numpy())
        sample_ids.append(batch['sample_id'].numpy())

    preds = np.concatenate(preds)
    sample_ids = np.concatenate(sample_ids)

    out = pd.DataFrame({'sample_id': sample_ids, 'price': np.round(preds, 2)})
    out = out.sort_values('sample_id').reset_index(drop=True)

    os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"✅ Saved predictions → {args.out_csv} | total={len(out)}")


if __name__ == '__main__':
>>>>>>> f9f3a42997332c90f174d78f3071d7bbc3012ad7
    main()
