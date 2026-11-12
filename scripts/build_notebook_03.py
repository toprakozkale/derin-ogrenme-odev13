"""notebooks/03_segmentasyon.ipynb dosyasını üretir.

    .venv/bin/python scripts/build_notebook_03.py
"""
import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _nb_common import setup_code  # noqa: E402

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md("""# 03 · Segmentasyon — YOLO26-seg (nano, small) ↔ RF-DETR Seg Small

**Soru:** Fotoğraftaki hastalıklı bölgeleri piksel piksel ne kadar doğru bulabiliyoruz? İki farklı model ailesi —
Ultralytics **YOLO26-seg** (evrişimli, hızlı) ve **RF-DETR Seg** (dönüştürücü/transformer tabanlı) — aynı veride, aynı
koşullarda karşılaştırılıyor.

- Veri: `01_veri_hazirlama`'nın ürettiği YOLO-seg ve COCO dosyaları (aynı poligonlardan; resmi bölme).
- Modeller: **YOLO26n-seg**, **YOLO26s-seg** ve **RF-DETR Seg Small** — hepsi COCO ön-eğitimli, **3 seed**.
- Adil karşılaştırma: aynı giriş çözünürlüğü (**480**), aynı epoch sınırı (100) ve erken durdurma (20), aynı etkin batch (16),
  aynı T4, **aynı değerlendirme kodu** (`src/metrics.py`): iki modelin tahminleri de COCO biçimine çevrilip öyle ölçülür.
- Ölçütler: mask **mAP50**, **mAP50-95** (pycocotools), **mIoU**, **Dice**, sınıf bazında IoU; **tüm test (743)** ve
  **temiz test (692)**. Model seçimi yalnız doğrulama (val) ile; test en sonda bir kez.

**Colab'da çalıştırma:** Çalışma zamanı → Çalışma zamanı türünü değiştir → **T4 GPU**. Eğitim uzun sürer (birkaç oturuma
yayılabilir). Bağlantı koparsa **Çalışma zamanı → Oturumu yeniden başlat → Tümünü çalıştır**: biten koşular atlanır,
yarım kalan kaldığı yerden sürer. Mac'te `caffeinate -dimsu` açık kalsın."""),
    md("## 0 · Ortam"),
    code(setup_code("ultralytics", "rfdetr[train]") + """
import json, time
import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
from src import config as C, seg
from src import metrics as M

# DUMAN TESTİ: yerelde (GPU yokken) kodu birkaç dakikada uçtan uca denemek için. Sonuçlar runs/seg_duman/'a yazılır,
# asıl sonuçlara ve panele asla karışmaz. Colab'da (GPU varken) otomatik kapalı.
SMOKE = os.environ.get("ODEV13_SMOKE", "0" if torch.cuda.is_available() else "1") == "1"
DEVICE = 0 if torch.cuda.is_available() else "cpu"
seg.use_smoke_output(SMOKE)

gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "yok"
print("Proje kökü:", ROOT, "| GPU:", gpu, "| DUMAN TESTİ:", SMOKE)
if torch.cuda.is_available() and "T4" not in gpu:
    print("⚠️  GPU T4 değil! Adil hız karşılaştırması için Çalışma zamanı türünü T4 yapın.")
env = seg.write_env(); env"""),
    md("## 1 · Veri\nYOLO-seg ve COCO dosyaları yoksa 01 notebook'undaki hat burada yeniden çalıştırılır (Kaggle'dan indirip ~15 sn'de üretir)."),
    code("""if not (seg.YOLO_DATA / "data.yaml").exists() or not (seg.COCO_DATA / "train" / "_annotations.coco.json").exists():
    from src.pipeline import download, run
    download(); run(verbose=False)

# Bölme × sınıf: örnek (poligon) sayıları — aynı dosyalardan iki biçim
rows = {}
for sp in C.SPLITS:
    gt = seg.load_gt(sp)
    rows[sp] = pd.Series([a["category_id"] - 1 for a in gt["annotations"]]).value_counts().reindex(range(7), fill_value=0).to_numpy()
counts = pd.DataFrame(rows, index=list(C.CLASSES)); counts.loc["TOPLAM örnek"] = counts.sum()
counts.loc["görüntü"] = [len(seg.load_gt(sp)["images"]) for sp in C.SPLITS]; counts"""),
    md("""## 2 · Deney düzeni

| Ayar | YOLO26-seg (n, s) | RF-DETR Seg Small | Basitçe |
|---|---|---|---|
| Ön-eğitim | COCO | COCO | genel nesnelerle önceden eğitilmiş; biz çilek hastalığına uyarlıyoruz |
| Giriş boyutu | 480 | 480 | iki modelde aynı (görüntüler 419×419) |
| Epoch / erken durdurma | 100 / 20 | 100 / 20 | val maske mAP'si 20 tur iyileşmezse durur |
| Batch | 16 | 4 × 4 birikim = 16 | aynı etkin batch |
| Optimizer, öğrenme hızı | `auto` (Ultralytics) | AdamW 1e-4 (paket varsayılanı) | her aile kendi önerilen ayarıyla |
| En iyi ağırlık | val fitness (box + mask mAP) | val maske mAP (EMA/normal) | seçim yalnız val ile |
| Seed | 0, 1, 2 | 0, 1, 2 | |

Sıra seed'e göre: önce üç modelin seed 0'ı biter, sonra seed 1, sonra seed 2 — kota yetmezse elde yine tam bir karşılaştırma olur."""),
    code("""if SMOKE:
    RUNS = [seg.RunSpec("yolo", "n", 0, imgsz=192, tag="duman"), seg.RunSpec("rfdetr", "small", 0, imgsz=192, tag="duman")]
else:
    RUNS = seg.MAIN_RUNS
YOLO_RUNS = [s for s in RUNS if s.arch == "yolo"]; RF_RUNS = [s for s in RUNS if s.arch == "rfdetr"]
pd.DataFrame([{"koşu": s.name, "model": s.model, "imgsz": s.imgsz, "seed": s.seed, "bitti": seg.is_finished(s)} for s in RUNS])"""),
    md("""## 3 · Eğitim
Hücre yarıda kesilirse (bağlantı koptu, kota bitti) tekrar çalıştırın: biten koşular atlanır, yarım kalan kaldığı yerden sürer.
**Ayarları (özellikle çözünürlüğü) yarıda değiştirmeyin** — RF-DETR farklı ayarla kaldığı yerden devam edemez.
RF-DETR bitince yalnız en iyi ağırlığı tutar (Drive'da yer kalsın diye ~800 MB'lık ara dosyalar silinir)."""),
    code("""for spec in RUNS:
    t0 = time.time()
    seg.train(spec, device=DEVICE, smoke=SMOKE)
    print(f"   ⏱ {spec.name}: bu oturumda {(time.time() - t0) / 60:.1f} dk\\n")"""),
    code("""# Süre özeti (DONE işaretlerinden): Colab'da bir sonraki seed'leri planlamak için
sure = []
for s in RUNS:
    p = s.dir / "DONE"
    if p.exists():
        d = json.loads(p.read_text()); log = seg.train_log(s)
        sure.append({"koşu": s.name, "epoch": len(log), "son_oturum_dk": round(d["sure_sn_bu_oturum"] / 60, 1),
                     "en_iyi_val_mask_mAP50-95": round(seg.best_val_map(s), 4)})
pd.DataFrame(sure)"""),
    md("## 4 · Eğitim eğrileri (aşırı öğrenme kontrolü)"),
    code("""# Duman testi şekilleri rapor klasörüne değil runs/seg_duman/figures'a yazılır
FIG = (seg.OUT / "figures") if SMOKE else (ROOT / "rapor" / "figures"); FIG.mkdir(parents=True, exist_ok=True)
SFX = "_duman" if SMOKE else ""
COL = {"yolo26n-seg": "#2a78d6", "yolo26s-seg": "#eb6834", "rfdetr-seg-small": "#1f9e6e"}
fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
for s in RUNS:
    log = seg.train_log(s)
    if log.empty: continue
    lab = f"{s.model} s{s.seed}"
    ax[0].plot(log["epoch"], log["egitim_kaybi"] / log["egitim_kaybi"].iloc[0], color=COL.get(s.model, "gray"), alpha=.8, lw=1.4, label=lab)
    ax[1].plot(log["epoch"], log["val_mask_map50_95"], color=COL.get(s.model, "gray"), alpha=.8, lw=1.4, label=lab)
ax[0].set_title("eğitim kaybı (ilk epoch'a göre oran)"); ax[1].set_title("doğrulama maske mAP50-95")
for a in ax: a.set_xlabel("epoch"); a.grid(alpha=.25); a.spines[["top", "right"]].set_visible(False)
ax[1].legend(frameon=False, fontsize=7, ncol=3)
plt.tight_layout(); plt.savefig(FIG / f"seg_egriler{SFX}.png", dpi=200); plt.show()"""),
    md("""## 5 · Test değerlendirmesi (tek sefer, ortak kod)
Her koşu için: val ve test tahminleri üretilir → **val'de** piksel metrikleri için güven eşiği seçilir (mIoU'yu en yükselten) →
testte mask mAP (pycocotools), mIoU, Dice, sınıf bazında IoU ve IoU ≥ 0,5 eşleşmeli örnek karışma matrisi hesaplanır."""),
    code("""res = seg.evaluate_all(RUNS, device=DEVICE, limit=12 if SMOKE else None)
summary = seg.summary_table(res); summary.to_csv(seg.OUT / "ozet.csv", index=False)
summary"""),
    code("""# Çapraz kontrol: bizim kodumuzla hesaplanan val mask mAP50-95 ↔ kütüphanenin eğitimde raporladığı en iyi değer.
# Yakın olmalı; çok farklıysa sınıf eşlemesinde ya da maske dönüşümünde hata vardır (o zaman sonuçlara güvenmeyin).
res[res.alt_kume == "tum_test"][["kosu", "val_mask_map50_95_kutuphane", "val_mask_map50_95_bizim", "esik"]].round(4)"""),
    code("""# Sınıf bazında: AP50-95 ve IoU — model başına seed ortalaması (tüm test)
pc = pd.read_csv(seg.OUT / "per_class.csv").merge(res[["kosu", "model"]].drop_duplicates(), on="kosu")
pc_tab = (pc[pc.alt_kume == "tum_test"].groupby(["sinif", "model"])[["ap50", "ap50_95", "iou", "dice"]].mean()
          .unstack("model").round(3))
pc_tab"""),
    code("""fig, ax = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
short = ["Ang. Leafspot", "Anthracnose", "Blossom Bl.", "Gray Mold", "Leaf Spot", "PM Fruit", "PM Leaf"]
models = list(dict.fromkeys(res.model)); w = 0.8 / len(models)
for k, (metric, title) in enumerate((("ap50_95", "maske AP50-95"), ("iou", "piksel IoU"))):
    for j, m in enumerate(models):
        v = pc[(pc.alt_kume == "tum_test") & (pc.model == m)].groupby("sinif")[metric].mean().reindex(C.CLASSES)
        ax[k].bar(np.arange(7) + (j - (len(models) - 1) / 2) * w, v, w, color=COL.get(m, "gray"), label=m)
    ax[k].set_xticks(range(7), short, rotation=35, ha="right", fontsize=8); ax[k].set_title(title + " (seed ort., tüm test)")
    ax[k].grid(axis="y", alpha=.25); ax[k].spines[["top", "right"]].set_visible(False); ax[k].set_ylim(0, 1)
ax[0].legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.savefig(FIG / f"seg_sinif_bazinda{SFX}.png", dpi=200); plt.show()"""),
    md("""## 6 · Sınıf karışması: meyve/yaprak ("Dikkat" maddesi, segmentasyonda)
Her gerçek örnek, IoU ≥ 0,5 ile örtüşen bir tahminle eşleştirilir (sınıfa bakmadan). Eşleşen çiftlerde gerçek ↔ tahmin sınıfı
karışma matrisine yazılır; eşleşmeyen gerçek örnek **"kaçırıldı"**, eşleşmeyen tahmin **"fazladan"** sütun/satırına gider.
Seed'ler toplanır. Ardından 7 sınıf **yaprak / meyve / çiçek** gruplarına indirgenir."""),
    code("""from matplotlib.colors import LinearSegmentedColormap
BLUE = LinearSegmentedColormap.from_list("seq", ["#f7f6f2", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
labels = short + ["yok"]
cms, groups, rates = {}, [], {}
for m in models:
    runs = [s for s in RUNS if s.model == m and (seg.OUT / "preds" / f"{s.name}_test_ornek_karisma.csv").exists()]
    if not runs: continue
    cm = sum(seg.load_instance_confusion(s) for s in runs); cms[m] = cm
    cmn = cm / np.clip(cm.sum(1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(6, 5.2))
    im = ax.imshow(cmn, cmap=BLUE, vmin=0, vmax=1)
    for i in range(8):
        for j in range(8):
            if cm[i, j]: ax.text(j, i, f"{cmn[i, j]:.2f}\\n({int(cm[i, j])})", ha="center", va="center", fontsize=6,
                                 color="white" if cmn[i, j] > .55 else "#0b0b0b")
    ax.set_xticks(range(8), short + ["kaçırıldı"], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(8), short + ["fazladan"], fontsize=8)
    ax.set_xlabel("tahmin"); ax.set_ylabel("gerçek"); ax.set_title(f"{m} — {len(runs)} seed toplamı, satır normalize", fontsize=9)
    plt.colorbar(im, fraction=.046); plt.tight_layout()
    plt.savefig(FIG / f"seg_karisma_{m}{SFX}.png", dpi=200); plt.savefig(FIG / f"seg_karisma_{m}{SFX}.pdf"); plt.show()
    g = M.group_confusion(cm[:7, :7]); g.insert(0, "model", m); groups.append(g)
    rates[m] = M.group_confusion_rate(cm[:7, :7])
display(pd.concat(groups))
gr = pd.DataFrame(rates).round(4); gr.to_csv(seg.OUT / "grup_karisma.csv"); gr"""),
    code("""for m, cm in cms.items():
    print(f"\\nEn sık sınıf karışmaları (eşleşen örnekler) — {m}:"); display(M.top_confusions(cm[:7, :7], 8))"""),
    md("## 7 · Verimlilik: parametre, ms/görüntü, GPU belleği (T4, batch = 1)"),
    code("""thr_of = res[res.alt_kume == "tum_test"].set_index("kosu")["esik"].to_dict()
eff_runs = [s for s in RUNS if s.seed == min(x.seed for x in RUNS) and seg.is_done(s)]
eff = pd.DataFrame([seg.efficiency(s, device=DEVICE, n_warmup=3 if SMOKE else 20, n_images=8 if SMOKE else None,
                                   thr=thr_of.get(s.name)) for s in eff_runs])
eff.to_csv(seg.OUT / "efficiency.csv", index=False); eff"""),
    code("""# Doğruluk–hız: her model için tüm test mask mAP50-95 (seed ort.) ↔ uçtan uca ms/görüntü
t = res[res.alt_kume == "tum_test"].groupby("model")["mask_map50_95"].agg(["mean", "std"]).join(eff.set_index("model")[["ms_uctan_uca", "parametre"]])
fig, ax = plt.subplots(figsize=(5.6, 3.8))
for m, r in t.iterrows():
    ax.errorbar(r.ms_uctan_uca, r["mean"], yerr=r["std"] if r["std"] == r["std"] else 0, fmt="o", color=COL.get(m, "gray"), capsize=3)
    ax.annotate(f"{m}\\n{r.parametre / 1e6:.1f} M parametre", (r.ms_uctan_uca, r["mean"]), textcoords="offset points", xytext=(8, -4), fontsize=8)
ax.set_xlabel("uçtan uca süre (ms/görüntü, batch 1)"); ax.set_ylabel("maske mAP50-95 (tüm test)")
ax.grid(alpha=.25); ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(FIG / f"seg_dogruluk_hiz{SFX}.png", dpi=200); plt.show(); t.round(4)"""),
    md("""## 8 · Hata analizi — en kötü 10 test görüntüsü
Her aileden **doğrulama** mAP'si en yüksek koşu seçilir (test bakılmaz). İkisinin görüntü başına IoU ortalaması en düşük 10 test
görüntüsü: **gerçek | YOLO | RF-DETR** yan yana. Yorumlar raporda hata türlerine ayrılır: kaçırılan küçük leke, yanlış sınıf,
sınır taşması, birleşen örnekler, olası etiket hatası."""),
    code("""from PIL import Image
from matplotlib.patches import Patch
best_y, best_r = seg.select_by_val(YOLO_RUNS), seg.select_by_val(RF_RUNS)
print("Val ile seçilen:", best_y.name, round(seg.best_val_map(best_y), 4), "|", best_r.name, round(seg.best_val_map(best_r), 4))
worst = seg.worst_images([best_y, best_r], 10); worst.to_csv(seg.OUT / "worst10.csv", index=False)
PAL = np.array([[230, 159, 0], [86, 180, 233], [0, 158, 115], [117, 72, 178], [0, 114, 178], [213, 94, 0], [204, 121, 167]]) / 255
def overlay(img, sem, a=.55):
    out = np.asarray(img, float) / 255
    for c in range(7):
        m = sem == c
        out[m] = (1 - a) * out[m] + a * PAL[c]
    return out
folder = seg.COCO_DATA / "test"; fn = {im["id"]: im["file_name"] for im in seg.load_gt("test")["images"]}
fig, axes = plt.subplots(len(worst), 3, figsize=(8.4, 2.8 * len(worst)))
axes = np.atleast_2d(axes)
for row, (_, r) in zip(axes, worst.iterrows()):
    img = Image.open(folder / fn[r.image_id]).convert("RGB")
    panels = [("gerçek", seg.gt_semantic("test", r.image_id)),
              (f"YOLO (IoU {r.iou_yolo:.2f})", seg.masks_for(best_y, "test", r.image_id, thr_of[best_y.name])),
              (f"RF-DETR (IoU {r.iou_rfdetr:.2f})", seg.masks_for(best_r, "test", r.image_id, thr_of[best_r.name]))]
    for ax, (title, sem) in zip(row, panels):
        ax.imshow(overlay(img, sem)); ax.axis("off"); ax.set_title(title, fontsize=8)
    row[0].set_title(f"{r.stem}\\ngerçek: {r.gercek_siniflar}", fontsize=7.5)
fig.legend(handles=[Patch(color=PAL[i], label=short[i]) for i in range(7)], loc="lower center", ncol=4, frameon=False, fontsize=8)
plt.tight_layout(rect=(0, 0.02, 1, 1)); plt.savefig(FIG / f"seg_en_kotu10{SFX}.png", dpi=130); plt.show()
worst"""),
    md("## 9 · Kayıt"),
    code("""out = {"duman_testi": SMOKE, "secilen_yolo_val": best_y.name, "secilen_rfdetr_val": best_r.name, "env": env,
       "dosyalar": sorted(str(p.relative_to(ROOT)) for p in seg.OUT.glob("*.csv"))}
(seg.OUT / "summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
print(json.dumps(out, indent=2, ensure_ascii=False))
if SMOKE:
    print("\\n⚠️  DUMAN TESTİ — bu sayılar anlamsızdır (1 epoch, küçük alt küme). Gerçek sonuçlar Colab'da T4 ile üretilecek.")"""),
]

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}, "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"}}
out = Path(__file__).resolve().parents[1] / "notebooks" / "03_segmentasyon.ipynb"
nbf.write(nb, out)
print("yazıldı:", out)
