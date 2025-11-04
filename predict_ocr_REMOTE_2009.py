import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoImageProcessor
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
    main()
