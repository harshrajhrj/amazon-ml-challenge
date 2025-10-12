import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class PricingDatasetOCR(Dataset):
    def __init__(self, df, image_root, tok, img_proc, max_len=256, train=True):
        self.df = df.reset_index(drop=True)
        self.image_root = image_root
        self.tok = tok
        self.img_proc = img_proc
        self.max_len = max_len
        self.train = train
        self.has_price = 'price' in df.columns

    def _clean_text(self, x):
        x = '' if x is None else str(x)
        return ' '.join(x.lower().split())

    def _get_image(self, link):
        path = os.path.join(self.image_root, os.path.basename(str(link)))
        if os.path.exists(path):
            try:
                return Image.open(path).convert('RGB')
            except Exception:
                pass
        return Image.new('RGB', (224, 224), color=(255, 255, 255))

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        cat_text = self._clean_text(row.get('catalog_content', ''))
        ocr_text = self._clean_text(row.get('ocr_text', ''))

        tok_cat = self.tok(cat_text, padding='max_length', truncation=True, max_length=self.max_len, return_tensors='pt')
        tok_cat = {f'cat_{k}': v.squeeze(0) for k, v in tok_cat.items()}

        tok_ocr = self.tok(ocr_text, padding='max_length', truncation=True, max_length=self.max_len, return_tensors='pt')
        tok_ocr = {f'ocr_{k}': v.squeeze(0) for k, v in tok_ocr.items()}

        img = self._get_image(row.get('image_link', ''))
        img_inputs = self.img_proc(images=img, return_tensors='pt')
        img_inputs = {k: v.squeeze(0) for k, v in img_inputs.items()}

        out = {**tok_cat, **tok_ocr, **img_inputs, 'sample_id': torch.tensor(int(row['sample_id']))}
        if self.has_price:
            try:
                price = float(row['price'])
            except Exception:
                price = 1.0
            out['price'] = torch.tensor(price, dtype=torch.float32)
            out['log_price'] = torch.tensor(np.log1p(price), dtype=torch.float32)
        return out

    def __len__(self):
        return len(self.df)