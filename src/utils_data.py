import os
from pathlib import Path
import urllib.request
from multiprocessing.pool import ThreadPool
from tqdm import tqdm


def _download_one(link_save):
    link, save_dir = link_save
    try:
        fname = Path(link).name
        out = os.path.join(save_dir, fname)
        if not os.path.exists(out):
            urllib.request.urlretrieve(link, out)
        return True
    except Exception:
        return False


def download_images(csv_path, image_col='image_link', save_dir='data/images', workers=64):
    import pandas as pd
    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    pairs = [(l, save_dir) for l in df[image_col].astype(str).tolist()]
    ok = 0
    with ThreadPool(processes=workers) as pool:
        for done in tqdm(pool.imap_unordered(_download_one, pairs), total=len(pairs)):
            ok += int(done)
    print(f"Downloaded {ok}/{len(pairs)} images to {save_dir}")