"""notebooks/02_siniflandirma.ipynb dosyasını üretir.

    .venv/bin/python scripts/build_notebook_02.py
"""
import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _nb_common import setup_code  # noqa: E402

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md("""# 02 · Sınıflandırma — YOLO26-cls (nano ve small)

**Soru:** Bir çilek fotoğrafına bakıp 7 hastalıktan hangisi olduğunu ne kadar doğru söyleyebiliriz? Küçük (**n**ano) ve
daha büyük (**s**mall) model arasında doğruluk–hız farkı ne?

- Veri: `01_veri_hazirlama` notebook'unun ürettiği `data/processed/cls` (resmi bölme; görüntü düzeyi etiket = en büyük alanlı sınıf).
- Modeller: Ultralytics **YOLO26n-cls** ve **YOLO26s-cls** (ImageNet ön-eğitimli), her biri **3 farklı seed** ile.
- Ablasyon: nano modelde giriş çözünürlüğü 224 → 384 (küçük lekeler için çözünürlüğün etkisi).
- Değerlendirme kendi kodumuzla (`src/metrics.py`), **tüm test (743)** ve **temiz test (692, sızıntısız)** üzerinde ayrı ayrı.
- Model seçimi yalnız doğrulama (val) kümesiyle; test en sonda bir kez.

**Colab'da çalıştırma:** Çalışma zamanı → Çalışma zamanı türünü değiştir → **T4 GPU**. Eğitim hücreleri bağlantı koparsa
yeniden çalıştırıldığında **kaldığı yerden devam eder**; biten koşular atlanır. Uzun süreceği için Mac'te `caffeinate -dimsu` açık kalsın."""),
    md("## 0 · Ortam"),
    code(setup_code("ultralytics") + """
import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
from src import config as C, cls
from src import metrics as M

# DUMAN TESTİ: yerelde (GPU yokken) kodu birkaç saniyede uçtan uca denemek için. Sonuçlar runs/cls_duman/'a yazılır,
# asıl sonuçlara ve panele asla karışmaz. Colab'da otomatik kapalı.
SMOKE = os.environ.get("ODEV13_SMOKE", "0" if torch.cuda.is_available() else "1") == "1"
DEVICE = 0 if torch.cuda.is_available() else "cpu"
cls.use_smoke_output(SMOKE)

gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "yok"
print("Proje kökü:", ROOT, "| GPU:", gpu, "| DUMAN TESTİ:", SMOKE)
if torch.cuda.is_available() and "T4" not in gpu:
    print("⚠️  GPU T4 değil! Adil hız karşılaştırması için Çalışma zamanı türünü T4 yapın.")
env = cls.write_env(); env"""),
    md("## 1 · Veri\nSınıflandırma klasörleri yoksa 01 notebook'undaki hat burada yeniden çalıştırılır (Kaggle'dan indirip ~15 sn'de üretir)."),
    code("""if not (cls.CLS_DATA / "train").exists():
    from src.pipeline import download, run
    download(); run(verbose=False)

counts = pd.DataFrame({sp: {c: len(list((cls.CLS_DATA / sp / C.CLASS_SLUG[c]).glob("*.jpg"))) for c in C.CLASSES} for sp in C.SPLITS})
counts.loc["TOPLAM"] = counts.sum(); counts"""),
    md("""## 2 · Deney düzeni

| Ayar | Değer | Basitçe |
|---|---|---|
| Modeller | YOLO26n-cls, YOLO26s-cls | küçük ve orta boy model |
| Ön-eğitim | ImageNet | genel fotoğraflarla önceden eğitilmiş, biz çileğe uyarlıyoruz |
| Giriş boyutu | 224 (ablasyon: 384) | fotoğraf modele bu boyuta küçültülerek verilir |
| Epoch / erken durdurma | 100 / 20 | val doğruluğu 20 tur iyileşmezse durur |
| Batch, optimizer | 64, `auto` (AdamW) | |
| Seed | 0, 1, 2 | aynı deneyi 3 kez; sonuç ± sapma ile verilir |
| Veri çoğaltma | Ultralytics varsayılanı | |"""),
    code("""if SMOKE:
    MAIN = [cls.RunSpec(sz, 0, imgsz=64, tag="duman") for sz in ("n", "s")]
    ABL = []
else:
    MAIN, ABL = cls.MAIN_RUNS, cls.ABLATION_RUNS
pd.DataFrame([{"koşu": s.name, "model": s.model, "imgsz": s.imgsz, "seed": s.seed, "bitti": cls.is_done(s)} for s in MAIN + ABL])"""),
    md("## 3 · Eğitim — ana deneyler (n × 3 seed, s × 3 seed)\nHücre yarıda kesilirse tekrar çalıştırın: biten koşular atlanır, yarım kalan kaldığı yerden sürer."),
    code("""for spec in MAIN:
    cls.train(spec, device=DEVICE, smoke=SMOKE)"""),
    md("## 4 · Eğitim — ablasyon (nano, imgsz 384 × 3 seed)\nZaman/kota yetmezse bu hücre atlanabilir; raporda belirtilir."),
    code("""for spec in ABL:
    cls.train(spec, device=DEVICE, smoke=SMOKE)"""),
    md("## 5 · Eğitim eğrileri (aşırı öğrenme kontrolü)"),
    code("""# Duman testi şekilleri rapor klasörüne değil runs/cls_duman/figures'a yazılır
FIG = (cls.OUT / "figures") if SMOKE else (ROOT / "rapor" / "figures"); FIG.mkdir(parents=True, exist_ok=True)
SFX = "_duman" if SMOKE else ""
fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
for spec in MAIN:
    log = cls.train_log(spec)
    if log.empty: continue
    c = "#2a78d6" if spec.size == "n" else "#eb6834"
    ax[0].plot(log["epoch"], log["train/loss"], color=c, alpha=.8, lw=1.5, label=f"{spec.size} s{spec.seed}")
    acc = [k for k in log.columns if "accuracy_top1" in k][0]
    ax[1].plot(log["epoch"], log[acc], color=c, alpha=.8, lw=1.5, label=f"{spec.size} s{spec.seed}")
ax[0].set_title("eğitim kaybı"); ax[1].set_title("doğrulama doğruluğu (top-1)")
for a in ax: a.set_xlabel("epoch"); a.grid(alpha=.25); a.spines[["top", "right"]].set_visible(False)
ax[1].legend(frameon=False, fontsize=8, ncol=2)
plt.tight_layout(); plt.savefig(FIG / f"cls_egriler{SFX}.png", dpi=200); plt.show()"""),
    md("## 6 · Test değerlendirmesi (tek sefer)\nTüm koşuların test tahminleri üretilir ve kaydedilir; metrikler hem tüm test hem temiz test için."),
    code("""res = cls.evaluate_all(MAIN + ABL, device=DEVICE, limit=60 if SMOKE else None)
summary = cls.summary_table(res)
summary"""),
    code("""# Sınıf bazında F1 — model başına 3 seed ortalaması (tüm test)
pc = pd.read_csv(cls.OUT / "per_class.csv").merge(res[["kosu", "model", "deney"]].drop_duplicates(), on="kosu")
pc_tab = (pc[(pc.alt_kume == "tum_test") & (pc.deney.isin(["ana", "duman"]))]
          .groupby(["sinif", "model"])[["precision", "recall", "f1"]].mean().unstack("model").round(3))
pc_tab"""),
    md("## 7 · İstatistiksel güven: bootstrap %95 GA ve McNemar (n ↔ s)"),
    code("""rows = []
for spec in MAIN:
    p = cls.load_preds(spec)
    for subset, d in (("tum_test", p), ("temiz_test", p[p.stem.isin(cls.clean_test_stems())])):
        acc, lo, hi = M.bootstrap_ci(d.y_true, d.y_pred, "accuracy", n_boot=500 if SMOKE else 2000)
        f1, flo, fhi = M.bootstrap_ci(d.y_true, d.y_pred, "macro_f1", n_boot=500 if SMOKE else 2000)
        rows.append({"koşu": spec.name, "alt_küme": subset, "doğruluk": f"{acc:.3f} [{lo:.3f}, {hi:.3f}]", "macro F1": f"{f1:.3f} [{flo:.3f}, {fhi:.3f}]"})
boot = pd.DataFrame(rows); boot.to_csv(cls.OUT / "bootstrap.csv", index=False); boot"""),
    code("""mc = []
for seed in sorted({s.seed for s in MAIN}):
    a = next(s for s in MAIN if s.size == "n" and s.seed == seed); b = next(s for s in MAIN if s.size == "s" and s.seed == seed)
    pa, pb = cls.load_preds(a), cls.load_preds(b)
    r = M.mcnemar_test(pa.y_true, pa.y_pred, pb.y_pred)
    mc.append({"seed": seed, "A": a.model, "B": b.model, **r})
mc = pd.DataFrame(mc); mc.to_csv(cls.OUT / "mcnemar.csv", index=False); mc"""),
    md("""## 8 · Karmaşıklık matrisleri ve meyve/yaprak karışması ("Dikkat" maddesi)
Her model için 3 seed'in test tahminleri **toplanarak** (3 × 743 karar) matris çizilir. Ardından 7 sınıf **yaprak / meyve / çiçek**
gruplarına indirgenir: gruplar arası hücreler, modelin bir yaprak hastalığını meyve hastalığı sanması gibi hataları gösterir."""),
    code("""from matplotlib.colors import LinearSegmentedColormap
BLUE = LinearSegmentedColormap.from_list("seq", ["#f7f6f2", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
short = ["Ang. Leafspot", "Anthracnose", "Blossom Bl.", "Gray Mold", "Leaf Spot", "PM Fruit", "PM Leaf"]
cms, groups = {}, []
for size in ("n", "s"):
    runs = [s for s in MAIN if s.size == size]
    p = pd.concat([cls.load_preds(s) for s in runs])
    cm = M.confusion(p.y_true, p.y_pred); cms[size] = cm
    cmn = cm / np.clip(cm.sum(1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(cmn, cmap=BLUE, vmin=0, vmax=1)
    for i in range(7):
        for j in range(7):
            if cm[i, j]: ax.text(j, i, f"{cmn[i, j]:.2f}\\n({int(cm[i, j])})", ha="center", va="center", fontsize=6.5,
                                 color="white" if cmn[i, j] > .55 else "#0b0b0b")
    ax.set_xticks(range(7), short, rotation=45, ha="right", fontsize=8); ax.set_yticks(range(7), short, fontsize=8)
    ax.set_xlabel("tahmin"); ax.set_ylabel("gerçek"); ax.set_title(f"{runs[0].model} — {len(runs)} seed toplamı, satır normalize", fontsize=9)
    plt.colorbar(im, fraction=.046); plt.tight_layout(); plt.savefig(FIG / f"cls_karmasiklik_{size}{SFX}.pdf"); plt.savefig(FIG / f"cls_karmasiklik_{size}{SFX}.png", dpi=200); plt.show()
    g = M.group_confusion(cm); g.insert(0, "model", runs[0].model); groups.append(g)
pd.concat(groups)"""),
    code("""rates = pd.DataFrame({cls.RunSpec(sz, 0).model: M.group_confusion_rate(cms[sz]) for sz in cms}).round(4)
rates.to_csv(cls.OUT / "grup_karisma.csv"); display(rates)
for sz in cms:
    print(f"\\nEn sık karışmalar — {cls.RunSpec(sz, 0).model}:"); display(M.top_confusions(cms[sz], 8))"""),
    md("## 9 · Verimlilik: parametre sayısı ve çıkarım süresi (T4, batch = 1)"),
    code("""eff = pd.DataFrame([cls.efficiency(s, device=DEVICE, n_warmup=3 if SMOKE else 20, n_images=10 if SMOKE else None)
                    for s in MAIN if s.seed == 0] + [cls.efficiency(s, device=DEVICE, n_warmup=3 if SMOKE else 20, n_images=10 if SMOKE else None)
                    for s in ABL if s.seed == 0 and cls.is_done(s)])
eff.to_csv(cls.OUT / "efficiency.csv", index=False); eff"""),
    md("""## 10 · Hata analizi — en kötü 10 tahmin
Seçim yalnız **doğrulama** doğruluğuna göre yapılır (en iyi val koşusu); o modelin test kümesinde **gerçek sınıfa en düşük
olasılığı verdiği** 10 yanlış tahmin gösterilir. Yorumlar raporda kategorilere ayrılır: arka plan, birden fazla hastalık,
erken evre/küçük leke, ışık/bulanıklık, olası etiket hatası."""),
    code("""from PIL import Image
best = cls.select_by_val(MAIN)
print("Val ile seçilen koşu:", best.name, "| val doğruluğu:", round(cls.best_val_acc(best), 4))
worst = cls.worst_predictions(cls.load_preds(best), 10)
worst.drop(columns="path").to_csv(cls.OUT / "worst10.csv", index=False)
fig, axes = plt.subplots(2, 5, figsize=(15, 6.8))
for ax, (_, r) in zip(axes.ravel(), worst.iterrows()):
    ax.imshow(Image.open(r.path)); ax.axis("off")
    ax.set_title(f"{r.stem}\\ngerçek: {r.gercek}\\ntahmin: {r.tahmin} ({r.tahmin_guveni:.2f})" + ("  [gruplar arası]" if r.gruplar_arasi else ""), fontsize=7.5)
for ax in axes.ravel()[len(worst):]: ax.axis("off")
plt.tight_layout(); plt.savefig(FIG / f"cls_en_kotu10{SFX}.png", dpi=150); plt.show()
worst.drop(columns="path")"""),
    md("## 11 · Kayıt"),
    code("""import json
out = {"duman_testi": SMOKE, "secilen_kosu_val": best.name, "env": env,
       "dosyalar": sorted(str(p.relative_to(ROOT)) for p in cls.OUT.glob("*.csv"))}
(cls.OUT / "summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
print(json.dumps(out, indent=2, ensure_ascii=False))
if SMOKE:
    print("\\n⚠️  DUMAN TESTİ — bu sayılar anlamsızdır (1 epoch, verinin %5'i). Gerçek sonuçlar Colab'da T4 ile üretilecek.")"""),
]

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}, "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"}}
out = Path(__file__).resolve().parents[1] / "notebooks" / "02_siniflandirma.ipynb"
nbf.write(nb, out)
print("yazıldı:", out)
