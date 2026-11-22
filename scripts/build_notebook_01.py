"""notebooks/01_veri_hazirlama.ipynb dosyasını üretir (notebook'u elle değil bu betikle düzenliyoruz).

    .venv/bin/python scripts/build_notebook_01.py
"""
from pathlib import Path

import sys

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import IMAGE_LABEL_RULE as RULE  # noqa: E402
from _nb_common import setup_code  # noqa: E402

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md("""# 01 · Veri Hazırlama — Çilek Hastalığı (Görev 13)

**Amaç:** Strawberry Disease Detection Dataset'i (Afzaal ve ark., 2021) indirmek, etiketleri denetleyip kurallı biçimde düzeltmek,
keşifsel analiz yapmak, bölmeler arası sızıntıyı ölçmek ve etiketleri üç biçime dönüştürüp dönüşümü doğrulamak:

1. Ultralytics **sınıflandırma** klasörleri (`data/processed/cls`)
2. **YOLO-seg** (`data/processed/yoloseg`)
3. **COCO** — RF-DETR / Roboflow düzeni (`data/processed/coco`)

Tüm hesaplar `src/` altındaki fonksiyonlarla yapılır; yerel panel (`app/`) ve rapor aynı çıktıları (`data/derived/`) kullanır.
Notebook baştan sona **Runtime → Run all** ile çalışır; GPU gerekmez."""),
    md("## 0 · Ortam"),
    code(setup_code() + """
import platform, random
from src import config as C
random.seed(C.SEED)
import numpy as np; np.random.seed(C.SEED)

import pandas as pd, matplotlib, shapely, imagehash, pycocotools, PIL
print("Proje kökü :", ROOT)
print("Python     :", platform.python_version(), "| Colab:", IN_COLAB)
print("numpy", np.__version__, "| pandas", pd.__version__, "| shapely", shapely.__version__,
      "| imagehash", imagehash.__version__, "| Pillow", PIL.__version__)"""),
    md("""## 1 · İndirme

Veri Kaggle'dan `kagglehub` ile iner (`usmanafzaal/strawberry-disease-detection-dataset`, sürüm 1).
Veri seti herkese açıktır; Kaggle anahtarı **gerekmez** (anahtarsız da iner). Proje sahibi Colab 🔑 **Secrets**'ta
`KAGGLE_USERNAME`, `KAGGLE_KEY` ve `GITHUB_TOKEN` tutar; bunlar yoksa defter misafir kipinde çalışır (Drive'a dokunmaz).
Yerelde `~/.kaggle/kaggle.json` kullanılır. Veri zaten iniyse bu adım atlanır."""),
    code("""from src.pipeline import download
raw = download()
print("Ham veri:", raw)
for d in sorted(p for p in raw.iterdir() if p.is_dir()):
    print(f"  {d.name:32s} {len(list(d.rglob('*.jpg'))):5d} jpg  {len(list(d.rglob('*.json'))):5d} json")"""),
    md("""## 2 · Tüm hat: denetim → istatistik → sızıntı → dönüşüm → doğrulama

`src.pipeline.run()` sırasıyla:
- LabelMe JSON'ları okur, poligonları kurallarla düzeltir (kendini kesen → `make_valid` + en büyük parça; görüntü dışı → kırpma;
  3'ten az nokta / sıfır alan → atma) ve her işlemi kaydeder,
- bölme ve sınıf istatistiklerini hesaplar,
- MD5 ile birebir, pHash ile yakın-yinelenen görüntüleri bulur, eşiğe göre sızıntıyı ölçer ve **temiz test** listesini çıkarır,
- üç biçime dönüştürür ve her örneği geri okuyarak IoU ile doğrular."""),
    code("""from src.pipeline import run
summary = run(threshold=C.PHASH_THRESHOLD)"""),
    md("## 3 · Keşifsel analiz"),
    code("""S = summary["stats"]
tab = pd.DataFrame({sp: {"görüntü": S[sp]["n_images"], "örnek (poligon)": S[sp]["n_instances"],
                         "görüntü başına örnek (ort.)": round(S[sp]["instances_per_image"]["mean"], 2),
                         "en çok örnek": S[sp]["instances_per_image"]["max"],
                         "çok sınıflı görüntü": S[sp]["n_multiclass_images"],
                         "görüntü boyutları": ", ".join(S[sp]["image_sizes"])} for sp in C.SPLITS})
tab"""),
    code("""cls_tab = pd.DataFrame({(sp, k): S[sp][key] for sp in C.SPLITS
                        for k, key in (("görüntü", "images_per_label"), ("örnek", "instances_per_class"))})
cls_tab.index.name = "sınıf"
cls_tab"""),
    code("""import matplotlib.pyplot as plt
FIG = ROOT / "rapor" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
SPLIT_COLOR = {"train": "#2a78d6", "val": "#eb6834", "test": "#1baf7a"}

order = sorted(C.CLASSES, key=lambda c: S["train"]["images_per_label"][c])
fig, ax = plt.subplots(figsize=(7, 3.8))
h = 0.26
for k, sp in enumerate(C.SPLITS):
    y = np.arange(len(order)) + (k - 1) * h
    ax.barh(y, [S[sp]["images_per_label"][c] for c in order], height=h * 0.9, color=SPLIT_COLOR[sp], label=sp)
ax.set_yticks(np.arange(len(order)), order); ax.set_xlabel("görüntü sayısı (görüntü düzeyi etiket)")
ax.legend(frameon=False, loc="lower right"); ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="x", alpha=.25)
plt.tight_layout(); plt.savefig(FIG / "sinif_dagilimi.pdf"); plt.savefig(FIG / "sinif_dagilimi.png", dpi=200); plt.show()"""),
    md("### Örnek görüntüler (her sınıftan biri, poligon bindirmeli)"),
    code("""from PIL import Image, ImageDraw
from src.dataset import load_record
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
COL = {c: PAL[i] for i, c in enumerate(C.CLASSES)}

def overlay(split, stem):
    r = load_record(C.RAW / split / f"{stem}.json", split)
    im = Image.open(r.image_path).convert("RGBA"); lay = Image.new("RGBA", im.size); d = ImageDraw.Draw(lay)
    for i in r.instances:
        rgb = tuple(int(COL[i.label][k:k + 2], 16) for k in (1, 3, 5))
        d.polygon([tuple(p) for p in i.points], fill=rgb + (70,), outline=rgb + (255,), width=3)
    return Image.alpha_composite(im, lay).convert("RGB")

fig, axes = plt.subplots(1, 7, figsize=(16, 2.8))
for ax, c in zip(axes, C.CLASSES):
    stem = sorted(p.stem for p in (C.RAW / "train").glob(f"{C.CLASS_SLUG[c]}*.json"))[3]
    ax.imshow(overlay("train", stem)); ax.set_title(c, fontsize=8); ax.axis("off")
plt.tight_layout(); plt.savefig(FIG / "ornekler.png", dpi=110); plt.show()"""),
    md(f"""## 4 · Etiket denetimi

**Görüntü düzeyi etiket kuralı (sınıflandırma için):** {RULE}"""),
    code("""print("Görüntü düzeyi etiket kuralı:", C.IMAGE_LABEL_RULE)
print("İşlem sayıları:", summary["label_issue_counts"])
issues = pd.read_csv(C.DERIVED / "label_issues.csv"); issues"""),
    code("""multi = pd.read_csv(C.DERIVED / "multiclass_images.csv"); multi"""),
    code("""sv = summary["severity_check"]; sv"""),
    md("""## 5 · Yinelenen görüntüler ve sızıntı

pHash (64 bit) Hamming mesafesi. Eşik görsel kontrolle seçildi (`docs/phash_calibration.csv`): mesafe ≤ 12'de incelenen tüm bölmeler
arası çiftler aynı sahne, 14'te yanlış eşleşmeler başlıyor. 14–16 aralığında da gerçek yinelenenler kaldığı için ölçülen sızıntı
bir **alt sınırdır**. Resmi bölme değiştirilmez; test metrikleri hem tüm test hem **temiz test** üzerinde raporlanır."""),
    code("""print(summary["duplicates"])
lt = pd.read_csv(C.DERIVED / "leakage_table.csv")
fig, ax = plt.subplots(figsize=(6.5, 3.2))
ax.plot(lt.esik, lt.test_oran * 100, "-o", ms=3, lw=2, color="#1baf7a", label="test (benzeri train/val'de)")
ax.plot(lt.esik, lt.val_oran * 100, "-o", ms=3, lw=2, color="#eb6834", label="val (benzeri train'de)")
ax.axvline(C.PHASH_THRESHOLD, ls="--", color="#898781", lw=1); ax.text(C.PHASH_THRESHOLD - .3, 90, f"eşik={C.PHASH_THRESHOLD}", ha="right", color="#898781")
ax.set_xlabel("pHash Hamming mesafesi eşiği"); ax.set_ylabel("etkilenen görüntü (%)")
ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=.25)
plt.tight_layout(); plt.savefig(FIG / "sizinti_esik.pdf"); plt.savefig(FIG / "sizinti_esik.png", dpi=200); plt.show()
pd.read_csv(ROOT / "docs" / "phash_calibration.csv")"""),
    code("""pairs = pd.read_csv(C.DERIVED / "dup_pairs.csv")
ex = pairs[pairs.cross_split & (pairs.dist <= C.PHASH_THRESHOLD) & ((pairs.a_split == "test") | (pairs.b_split == "test"))].head(4)
fig, axes = plt.subplots(2, 4, figsize=(11, 5.6))
for k, (_, r) in enumerate(ex.iterrows()):
    for j, (sp, st) in enumerate(((r.a_split, r.a_stem), (r.b_split, r.b_stem))):
        a = axes[j][k]; a.imshow(Image.open(C.RAW / sp / f"{st}.jpg")); a.axis("off"); a.set_title(f"{sp}/{st}" + (f"  d={r.dist}" if j == 0 else ""), fontsize=8)
plt.tight_layout(); plt.savefig(FIG / "sizinti_ornek.png", dpi=110); plt.show()"""),
    md("## 6 · Dönüşüm doğrulaması"),
    code("""rt = summary["roundtrip"]
print(f"Örnek sayısı           : {rt['n_instances']}")
print(f"IoU (YOLO-seg) min/ort.: {rt['iou_yolo_min']:.4f} / {rt['iou_yolo_mean']:.5f}")
print(f"IoU (COCO)     min/ort.: {rt['iou_coco_min']:.4f} / {rt['iou_coco_mean']:.5f}")
print(f"IoU < 0.99             : {rt['n_iou_lt_0_99']}")
print(f"Sınıf / sayı tutarlı   : {rt['class_ok']} / {rt['count_ok']}")
print("pycocotools yükleme    :", rt["coco_load"])
assert rt["class_ok"] and rt["count_ok"] and rt["n_iou_lt_0_99"] == 0, "Dönüşüm doğrulaması başarısız!"
pd.DataFrame(rt["cls_counts"])"""),
    md("## 7 · Tekrarlanabilirlik kaydı"),
    code("""print("Bölme listeleri SHA-256:")
for k, v in summary["split_list_sha256"].items():
    print(f"  {k:10s} {v}")
print("Kaggle zip beklenen SHA-256:", summary["zip_sha256_beklenen"])
print("Süre:", summary["sure_sn"], "sn")"""),
]
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}, "colab": {"provenance": []}}
out = Path(__file__).resolve().parents[1] / "notebooks" / "01_veri_hazirlama.ipynb"
nbf.write(nb, out)
print("yazıldı:", out)
