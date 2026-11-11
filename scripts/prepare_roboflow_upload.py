"""Roboflow çapraz kontrolü için yükleme klasörlerini hazırlar (issue #2).

Ham Kaggle verisinden (bizim düzeltmelerimiz UYGULANMADAN) üç klasör üretir:
    data/roboflow_upload/{train,valid,test}/  →  <ad>.jpg + <ad>.json (LabelMe)

Tek değişiklik: LabelMe JSON'larının içindeki base64 görüntü kopyası (`imageData`) çıkarılır — Roboflow görüntüyü
yanındaki .jpg'den okur, yükleme ~yarıya iner. Poligonlara, etiketlere ve dosya adlarına dokunulmaz.
"Test Disease Severity Level" klasörü (test'in kopyası) dahil edilmez.

    .venv/bin/python scripts/prepare_roboflow_upload.py
"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402

SPLIT_DIR = {"train": "train", "val": "valid", "test": "test"}   # Roboflow'un bölme adları


def main() -> None:
    out = C.DATA / "roboflow_upload"
    if out.exists():
        shutil.rmtree(out)
    total = 0
    for split, rdir in SPLIT_DIR.items():
        d = out / rdir
        d.mkdir(parents=True)
        n = 0
        for js in sorted((C.RAW / split).glob("*.json")):
            j = json.loads(js.read_text())
            j["imageData"] = None
            (d / js.name).write_text(json.dumps(j, ensure_ascii=False))
            shutil.copy2(js.with_suffix(".jpg"), d / f"{js.stem}.jpg")
            n += 1
        size = sum(p.stat().st_size for p in d.iterdir()) / 1e6
        print(f"{rdir:6s} {n:5d} görüntü + etiket  ({size:.0f} MB)  →  {d}")
        total += n
    print(f"Toplam {total} görüntü.")


if __name__ == "__main__":
    main()
