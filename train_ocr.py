import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoImageProcessor, get_cosine_schedule_with_warmup
from src.data_ocr import PricingDatasetOCR
from src.model_ocr import MultiModalOCR
from src.losses import CombinedLoss
from src.ema import EMA
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import serialization as torch_serial



SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

def smape_np(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred) / ((np.abs(y_true) + np.abs(y_pred)) / 2 + 1e-8))

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_csv', type=str, default='dataset/train_ocr.csv')
    ap.add_argument('--val_frac', type=float, default=0.1)
    ap.add_argument('--image_root', type=str, default='data/images')
    ap.add_argument('--text_model', type=str, default='microsoft/deberta-v3-base')
    ap.add_argument('--img_model', type=str, default='google/vit-base-patch16-224-in21k')
    ap.add_argument('--max_len', type=int, default=256)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--bs', type=int, default=48)
    ap.add_argument('--lr_text', type=float, default=2e-5)
    ap.add_argument('--lr_img', type=float, default=2e-5)
    ap.add_argument('--lr_head', type=float, default=1e-4)
    ap.add_argument('--warmup_epochs', type=int, default=1)
    ap.add_argument('--num_workers', type=int, default=8)
    ap.add_argument('--ema_decay', type=float, default=0.999)
    ap.add_argument('--checkpointing', action='store_true')
    ap.add_argument('--out_dir', type=str, default='checkpoints_ocr')
    ap.add_argument('--resume_ckpt', type=str, default=None,
                    help='Path to checkpoint to resume from')
    ap.add_argument('--resume_epoch', type=int, default=None,
                    help='Epoch number to resume from')
    return ap.parse_args()

def make_loaders(args):
    df = pd.read_csv(args.train_csv)
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    n_val = int(len(df) * args.val_frac)
    val_df = df.iloc[:n_val].copy(); tr_df = df.iloc[n_val:].copy()

    tok = AutoTokenizer.from_pretrained(args.text_model)
    img_proc = AutoImageProcessor.from_pretrained(args.img_model)

    tr_ds = PricingDatasetOCR(tr_df, args.image_root, tok, img_proc, max_len=args.max_len, train=True)
    va_ds = PricingDatasetOCR(val_df, args.image_root, tok, img_proc, max_len=args.max_len, train=False)

    tr_loader = DataLoader(tr_ds, batch_size=args.bs, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    va_loader = DataLoader(va_ds, batch_size=args.bs, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    return tr_loader, va_loader

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tr_loader, va_loader = make_loaders(args)
    model = MultiModalOCR(args.text_model, args.img_model, checkpointing=False).to(device)

    no_decay = ['bias', 'LayerNorm.weight']
    text_params = list(model.text_encoder.named_parameters()) + list(model.ocr_encoder.named_parameters())
    img_params  = list(model.img_encoder.named_parameters())
    head_params = list(model.fusion.named_parameters())

    def group(params, lr):
        return [
            {'params': [p for n, p in params if not any(nd in n for nd in no_decay)], 'lr': lr, 'weight_decay': 0.01},
            {'params': [p for n, p in params if any(nd in n for nd in no_decay)],  'lr': lr, 'weight_decay': 0.0},
        ]

    optimizer = torch.optim.AdamW(
        [*group(text_params, args.lr_text),
         *group(img_params, args.lr_img),
         {'params': [p for _, p in head_params], 'lr': args.lr_head, 'weight_decay': 0.01}]
    )

    total_steps = len(tr_loader) * args.epochs
    warmup_steps = len(tr_loader) * args.warmup_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    loss_fn = CombinedLoss(alpha=0.7)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    ema = EMA(model, decay=args.ema_decay)

    start_epoch = 1
    best_smape = 1e9

    # Resume logic
    if args.resume_ckpt and os.path.exists(args.resume_ckpt):
        print(f"[INFO] Resuming from checkpoint: {args.resume_ckpt}")
        torch_serial.add_safe_globals([
                        np.core.multiarray.scalar,
                        np.dtype
                    ])
        ckpt = torch.load(args.resume_ckpt, map_location=device,weights_only=False)
        model.load_state_dict(ckpt['model'])
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler'])
        if 'scaler' in ckpt:
            scaler.load_state_dict(ckpt['scaler'])
        if 'ema_shadow' in ckpt:
            ema.shadow = ckpt['ema_shadow']
        start_epoch = args.resume_epoch + 1 if args.resume_epoch else ckpt.get('epoch', 1) + 1
        best_smape = ckpt.get('best_smape', best_smape)
        print(f"[INFO] Resumed from epoch {start_epoch-1}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_losses = []
        pbar = tqdm(tr_loader, total=len(tr_loader), desc=f"Epoch [{epoch}/{args.epochs}]", ncols=120)

        for batch in pbar:
            price_log = torch.log1p(batch['price']).to(device)
            cat_inputs = {k: v.to(device) for k, v in batch.items() if k.startswith('cat_')}
            ocr_inputs = {k: v.to(device) for k, v in batch.items() if k.startswith('ocr_')}
            img_inputs = {k: v.to(device) for k, v in batch.items() if k == 'pixel_values'}

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                pred_log = model(cat_inputs, img_inputs, ocr_inputs)
                loss = loss_fn(pred_log, price_log)

            try:
                scaler.scale(loss).backward()
            except RuntimeError as e:
                if "second time" in str(e):
                    print("[WARN] Gradient checkpoint reuse issue — skipping batch.")
                    optimizer.zero_grad(set_to_none=True)
                    continue
                else:
                    raise

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # free old graph memory to avoid AMP checkpoint reuse
            loss_val = loss.detach().item()
            del loss
            torch.cuda.empty_cache()

            ema.update(model)
            train_losses.append(loss_val)
            pbar.set_postfix(loss=f"{np.mean(train_losses[-50:]):.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        avg_tr_loss = float(np.mean(train_losses))

        # === Validation ===
        model.eval()
        ema.apply_to(model)
        preds, gts = [], []
        with torch.no_grad():
            for batch in tqdm(va_loader, desc="Validating", ncols=100):
                price = batch['price'].numpy()
                cat_inputs = {k: v.to(device) for k, v in batch.items() if k.startswith('cat_')}
                ocr_inputs = {k: v.to(device) for k, v in batch.items() if k.startswith('ocr_')}
                img_inputs = {k: v.to(device) for k, v in batch.items() if k == 'pixel_values'}
                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    pred_log = model(cat_inputs, img_inputs, ocr_inputs)
                    pred = torch.clamp(torch.expm1(pred_log), min=1.0)
                preds.append(pred.detach().cpu().numpy())
                gts.append(price)

        ema.restore(model)
        preds = np.concatenate(preds); gts = np.concatenate(gts)
        val_smape = smape_np(gts, preds)
        print(f"Epoch {epoch:02d} | TrainLoss {avg_tr_loss:.4f} | Val SMAPE {val_smape*100:.2f}%")

        # Save checkpoint every epoch
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'ema_shadow': ema.shadow,
            'epoch': epoch,
            'best_smape': best_smape,
            'text_model': args.text_model,
            'img_model': args.img_model
        }

        ckpt_path = os.path.join(args.out_dir, f"epoch{epoch:02d}.pt")
        torch.save(state, ckpt_path)

        # Save best model
        if val_smape < best_smape:
            best_smape = val_smape
            best_path = os.path.join(args.out_dir, f"best_epoch{epoch:02d}_smape{best_smape:.5f}.pt")
            torch.save(state, best_path)
            print(f"✅ Saved best model → {best_path}")

if __name__ == '__main__':
    main()
