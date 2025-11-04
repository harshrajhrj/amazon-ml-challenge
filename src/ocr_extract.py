import os
import pandas as pd
from tqdm import tqdm
from PIL import Image

# Prefer PaddleOCR (accuracy), fall back to EasyOCR. If neither is available, return empty strings.

def _init_paddle():
    try:
        from paddleocr import PaddleOCR  # type: ignore
        return PaddleOCR(lang='en')
    except Exception:
        return None


def _init_easyocr():
    try:
        import easyocr  # type: ignore
        return easyocr.Reader(['en'], gpu=False)
    except Exception:
        return None


def extract_with_paddle(engine, img_path):
    try:
        res = engine.ocr(img_path, cls=True)
        if res and res[0]:
            return " ".join([line[1][0] for line in res[0]]).strip()
    except Exception:
        pass
    return ""


def extract_with_easyocr(engine, img_path):
    try:
        res = engine.readtext(img_path, detail=0, paragraph=True)
        if res:
            return " ".join(res).strip()
    except Exception:
        pass
    return ""


def extract_ocr_text(img_path, paddle_engine=None, easy_engine=None):
    if paddle_engine is not None:
        text = extract_with_paddle(paddle_engine, img_path)
        if text:
            return text
    if easy_engine is not None:
        text = extract_with_easyocr(easy_engine, img_path)
        if text:
            return text
    return ""


def run_ocr(csv_path, image_root, out_csv):
    df = pd.read_csv(csv_path)
    paddle_engine = _init_paddle()
    easy_engine = None if paddle_engine is not None else _init_easyocr()

    if paddle_engine is None and easy_engine is None:
        print("[WARN] PaddleOCR & EasyOCR not available. Proceeding with empty OCR text.")

    ocr_texts = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_name = os.path.basename(str(row.get('image_link', '')))
        img_path = os.path.join(image_root, img_name)
        if not os.path.exists(img_path):
            ocr_texts.append("")
            continue
        try:
            Image.open(img_path).convert('RGB')
        except Exception:
            ocr_texts.append("")
            continue
        ocr_texts.append(extract_ocr_text(img_path, paddle_engine, easy_engine))

    df['ocr_text'] = ocr_texts
    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved OCR-augmented CSV → {out_csv}")

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv_path', type=str, required=True)
    ap.add_argument('--image_root', type=str, default='data/images')
    ap.add_argument('--out_csv', type=str, required=True)
    args = ap.parse_args()
    run_ocr(args.csv_path, args.image_root, args.out_csv)