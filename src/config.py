"""Proje genelinde tek kaynak: yollar, sınıflar, gruplar, sabitler.

Hem yerelde (Mac) hem Colab'da çalışır: kök klasör ODEV13_ROOT ortam değişkeniyle değiştirilebilir.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("ODEV13_ROOT", Path(__file__).resolve().parents[1]))
DATA = ROOT / "data"
RAW = DATA / "raw" / "strawberry"          # Kaggle zip'inin açıldığı yer
PROCESSED = DATA / "processed"             # cls/, yoloseg/, coco/
DERIVED = DATA / "derived"                 # istatistik, sorun listeleri, doğrulama çıktıları (panel bunları okur)
RUNS = ROOT / "runs"

KAGGLE_REF = "usmanafzaal/strawberry-disease-detection-dataset"
KAGGLE_ZIP_SHA256 = "e728dd56eaab4e11e64d46da77b06eb377beb264c53d0a32c1eaf44d051de525"

SPLITS = ("train", "val", "test")          # resmi bölme (ödev kuralı: varsa resmi bölme kullanılır)
SEED = 42

# Sınıf sırası sabittir (alfabetik); tüm formatlarda aynı kimlik numaraları kullanılır.
CLASSES = (
    "Angular Leafspot",
    "Anthracnose Fruit Rot",
    "Blossom Blight",
    "Gray Mold",
    "Leaf Spot",
    "Powdery Mildew Fruit",
    "Powdery Mildew Leaf",
)
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}
CLASS_SLUG = {c: c.lower().replace(" ", "_") for c in CLASSES}   # klasör adları için

# Türkçe adlar (rapor ve panel için)
CLASS_TR = {
    "Angular Leafspot": "Köşeli yaprak lekesi",
    "Anthracnose Fruit Rot": "Antraknoz meyve çürüklüğü",
    "Blossom Blight": "Çiçek yanıklığı",
    "Gray Mold": "Kurşuni küf",
    "Leaf Spot": "Yaprak lekesi",
    "Powdery Mildew Fruit": "Külleme (meyve)",
    "Powdery Mildew Leaf": "Külleme (yaprak)",
}

# "Dikkat" maddesi: meyve / yaprak (/ çiçek) grupları
GROUP_OF = {
    "Angular Leafspot": "yaprak",
    "Leaf Spot": "yaprak",
    "Powdery Mildew Leaf": "yaprak",
    "Anthracnose Fruit Rot": "meyve",
    "Gray Mold": "meyve",
    "Powdery Mildew Fruit": "meyve",
    "Blossom Blight": "çiçek",
}
GROUPS = ("yaprak", "meyve", "çiçek")

# Görüntü düzeyi etiket kuralı (raporda açıkça yazılır)
IMAGE_LABEL_RULE = (
    "Görüntüdeki tüm örnekler tek sınıftansa o sınıf; birden fazla sınıf varsa "
    "toplam poligon alanı en büyük olan sınıf."
)

# Algısal hash (pHash) ile yakın-yinelenen eşiği — görsel kontrolle seçildi (data/derived/phash_calibration.csv):
# mesafe <= 12'de rastgele örneklenen bölmeler arası çiftlerin hepsi aynı sahne; 14'te yanlış eşleşmeler başlıyor.
# 14–16 arasında da gerçek yinelenenler kaldığı için 12 ile ölçülen sızıntı bir ALT SINIRDIR.
PHASH_THRESHOLD = 12
PHASH_MAX_STORE = 20       # bu mesafeye kadar olan bütün çiftler kaydedilir (panelde kaydırıcı için)


def ensure_dirs() -> None:
    for p in (DATA, PROCESSED, DERIVED, RUNS):
        p.mkdir(parents=True, exist_ok=True)
