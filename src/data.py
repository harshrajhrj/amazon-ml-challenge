import os
import re
import io
from PIL import Image
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

NUM_RE = re.compile(r"(\d+[.,]?\d*)\s*(ml|g|kg|l|litre|pack|pcs|unit|cm|mm)", re.I)

class PricingDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_root: str, tok, img_proc, max_len: int = 256, train: bool = True):
        self.df = df.reset_index(drop=True)
        self.image_root = image_root
        self.tok = tok
        self.img_proc = img_proc
        self.max_len = max_len
        self.train = train
        self.has_price = 'price' in df.columns

    def _clean_text(self, x: str):
        x = str(x)
        x = re.sub(r"[^A-Za-z0-9\s]", " ", x)
        x = re.sub(r"\s+", " ", x).strip().lower()
        return x

    def _image_path_from_link(self, link: str):
        fname = os.path.basename(str(link))
        return os.path.join(self.image_root, fname)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        text = self._clean_text(row['catalog_content'])
        enc = self.tok(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        enc = {k: v.squeeze(0) for k, v in enc.items()}

        img_path = self._image_path_from_link(row['image_link'])
        if os.path.exists(img_path):
            image = Image.open(img_path).convert('RGB')
        else:
            # fallback: blank image
            image = Image.new('RGB', (224, 224), color=(255, 255, 255))
        img_inputs = self.img_proc(images=image, return_tensors='pt')
        img_inputs = {k: v.squeeze(0) for k, v in img_inputs.items()}

        out = {**enc, **img_inputs, 'sample_id': torch.tensor(int(row['sample_id']))}

        if self.has_price:
            price = float(row['price'])
            out['price'] = torch.tensor(price, dtype=torch.float32)
            out['log_price'] = torch.tensor(np.log1p(price), dtype=torch.float32)
        return out
