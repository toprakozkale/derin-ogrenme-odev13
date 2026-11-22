"""Colab'dan alınan runs/seg sonuçlarından segmentasyon rapor şekillerini yerelde yeniden üretir (yeniden eğitim/tahmin YOK).

03 notebook'u şekilleri Drive'daki rapor/figures'a yazar; Drive'dan yalnız runs/seg indirildiğinde şekiller bu betikle,
notebook'taki aynı çizim koduyla, kaydedilmiş tahminlerden (runs/seg/preds) çizilir. Önce tutarlılık denetimi yapar:
kaydedilmiş test tahminlerinden yeniden hesaplanan maske mAP50-95, results.csv ile aynı olmalı (yerel COCO verisi Colab'dakiyle aynı mı?).

    .venv/bin/python scripts/make_seg_figures.py            # denetim + bütün şekiller
    .venv/bin/python scripts/make_seg_figures.py --hizli    # denetimi yalnız val ile seçilen iki koşuda yap
"""
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from PIL import Image  # noqa: E402

from src import config as C  # noqa: E402
from src import metrics as M  # noqa: E402
from src import seg  # noqa: E402

FIG = ROOT / "rapor" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
RUNS = [s for s in seg.MAIN_RUNS if seg.is_finished(s)]
YOLO_RUNS = [s for s in RUNS if s.arch == "yolo"]
RF_RUNS = [s for s in RUNS if s.arch == "rfdetr"]
COL = {"yolo26n-seg": "#2a78d6", "yolo26s-seg": "#eb6834", "rfdetr-seg-small": "#1f9e6e"}
SHORT = ["Ang. Leafspot", "Anthracnose", "Blossom Bl.", "Gray Mold", "Leaf Spot", "PM Fruit", "PM Leaf"]
res = pd.read_csv(seg.OUT / "results.csv")
models = list(dict.fromkeys(res.model))


def check_consistency(specs) -> None:
    gt = seg.load_gt("test")
    for s in specs:
        d = json.loads(gzip.decompress(seg.preds_path(s, "test").read_bytes()))
        ours = M.coco_mask_map(gt, d["results"], img_ids=d["image_ids"])["map50_95"]
        ref = float(res[(res.kosu == s.name) & (res.alt_kume == "tum_test")].mask_map50_95.iloc[0])
        assert abs(ours - ref) < 1e-6, f"{s.name}: yeniden hesaplanan {ours:.6f} ≠ results.csv {ref:.6f}"
        print(f"✓ {s.name}: test maske mAP50-95 {ours:.4f} (results.csv ile aynı)")


def egriler():
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    for s in RUNS:
        log = seg.train_log(s)
        if log.empty:
            continue
        lab = f"{s.model} s{s.seed}"
        ax[0].plot(log["epoch"], log["egitim_kaybi"] / log["egitim_kaybi"].iloc[0], color=COL.get(s.model, "gray"), alpha=.8, lw=1.4, label=lab)
        ax[1].plot(log["epoch"], log["val_mask_map50_95"], color=COL.get(s.model, "gray"), alpha=.8, lw=1.4, label=lab)
    ax[0].set_title("eğitim kaybı (ilk epoch'a göre oran)"); ax[1].set_title("doğrulama maske mAP50-95")
    for a in ax:
        a.set_xlabel("epoch"); a.grid(alpha=.25); a.spines[["top", "right"]].set_visible(False)
    ax[1].legend(frameon=False, fontsize=7, ncol=3)
    plt.tight_layout(); plt.savefig(FIG / "seg_egriler.png", dpi=200); plt.close()


def sinif_bazinda():
    pc = pd.read_csv(seg.OUT / "per_class.csv").merge(res[["kosu", "model"]].drop_duplicates(), on="kosu")
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    w = 0.8 / len(models)
    for k, (metric, title) in enumerate((("ap50_95", "maske AP50-95"), ("iou", "piksel IoU"))):
        for j, m in enumerate(models):
            v = pc[(pc.alt_kume == "tum_test") & (pc.model == m)].groupby("sinif")[metric].mean().reindex(C.CLASSES)
            ax[k].bar(np.arange(7) + (j - (len(models) - 1) / 2) * w, v, w, color=COL.get(m, "gray"), label=m)
        ax[k].set_xticks(range(7), SHORT, rotation=35, ha="right", fontsize=8); ax[k].set_title(title + " (seed ort., tüm test)")
        ax[k].grid(axis="y", alpha=.25); ax[k].spines[["top", "right"]].set_visible(False); ax[k].set_ylim(0, 1)
    ax[0].legend(frameon=False, fontsize=8)
    plt.tight_layout(); plt.savefig(FIG / "seg_sinif_bazinda.png", dpi=200); plt.close()


def karisma():
    blue = LinearSegmentedColormap.from_list("seq", ["#f7f6f2", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    for m in models:
        runs = [s for s in RUNS if s.model == m]
        cm = sum(seg.load_instance_confusion(s) for s in runs)
        cmn = cm / np.clip(cm.sum(1, keepdims=True), 1, None)
        fig, ax = plt.subplots(figsize=(6, 5.2))
        im = ax.imshow(cmn, cmap=blue, vmin=0, vmax=1)
        for i in range(8):
            for j in range(8):
                if cm[i, j]:
                    ax.text(j, i, f"{cmn[i, j]:.2f}\n({int(cm[i, j])})", ha="center", va="center", fontsize=6,
                            color="white" if cmn[i, j] > .55 else "#0b0b0b")
        ax.set_xticks(range(8), SHORT + ["kaçırıldı"], rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(8), SHORT + ["fazladan"], fontsize=8)
        ax.set_xlabel("tahmin"); ax.set_ylabel("gerçek"); ax.set_title(f"{m} — {len(runs)} seed toplamı, satır normalize", fontsize=9)
        plt.colorbar(im, fraction=.046); plt.tight_layout()
        plt.savefig(FIG / f"seg_karisma_{m}.png", dpi=200); plt.savefig(FIG / f"seg_karisma_{m}.pdf"); plt.close()


def dogruluk_hiz():
    eff = pd.read_csv(seg.OUT / "efficiency.csv")
    t = res[res.alt_kume == "tum_test"].groupby("model")["mask_map50_95"].agg(["mean", "std"]).join(
        eff.set_index("model")[["ms_uctan_uca", "parametre"]])
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for m, r in t.iterrows():
        ax.errorbar(r.ms_uctan_uca, r["mean"], yerr=r["std"], fmt="o", color=COL.get(m, "gray"), capsize=3)
        ax.annotate(f"{m}\n{r.parametre / 1e6:.1f} M parametre", (r.ms_uctan_uca, r["mean"]), textcoords="offset points",
                    xytext=(8, -4), fontsize=8)
    ax.set_xlabel("uçtan uca süre (ms/görüntü, batch 1)"); ax.set_ylabel("maske mAP50-95 (tüm test)")
    ax.set_xlim(right=t.ms_uctan_uca.max() * 1.35)
    ax.grid(alpha=.25); ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(FIG / "seg_dogruluk_hiz.png", dpi=200); plt.close()


def en_kotu10():
    thr_of = res[res.alt_kume == "tum_test"].set_index("kosu")["esik"].to_dict()
    best_y, best_r = seg.select_by_val(YOLO_RUNS), seg.select_by_val(RF_RUNS)
    worst = pd.read_csv(seg.OUT / "worst10.csv")
    pal = np.array([[230, 159, 0], [86, 180, 233], [0, 158, 115], [117, 72, 178], [0, 114, 178], [213, 94, 0], [204, 121, 167]]) / 255

    def overlay(img, sem, a=.55):
        out = np.asarray(img, float) / 255
        for c in range(7):
            m = sem == c
            out[m] = (1 - a) * out[m] + a * pal[c]
        return out

    folder = seg.COCO_DATA / "test"
    fn = {im["id"]: im["file_name"] for im in seg.load_gt("test")["images"]}
    # Rapor sayfasına sığsın diye 2 × 5 yerleşim: her görüntü için gerçek | YOLO | RF-DETR üçlüsü yan yana
    fig, axes = plt.subplots(5, 6, figsize=(12, 10.6))
    for k, (_, r) in enumerate(worst.iterrows()):
        row, col0 = k // 2, (k % 2) * 3
        img = Image.open(folder / fn[r.image_id]).convert("RGB")
        panels = [(f"{r.stem}\ngerçek: {r.gercek_siniflar}", seg.gt_semantic("test", r.image_id)),
                  (f"YOLO26s (IoU {r.iou_yolo:.2f})", seg.masks_for(best_y, "test", r.image_id, thr_of[best_y.name])),
                  (f"RF-DETR (IoU {r.iou_rfdetr:.2f})", seg.masks_for(best_r, "test", r.image_id, thr_of[best_r.name]))]
        for j, (title, sem) in enumerate(panels):
            ax = axes[row, col0 + j]
            ax.imshow(overlay(img, sem)); ax.axis("off"); ax.set_title(title, fontsize=7)
    fig.legend(handles=[Patch(color=pal[i], label=SHORT[i]) for i in range(7)], loc="lower center", ncol=7, frameon=False, fontsize=8)
    plt.tight_layout(rect=(0, 0.03, 1, 1)); plt.savefig(FIG / "seg_en_kotu10.png", dpi=150); plt.close()
    print("Val ile seçilen:", best_y.name, "|", best_r.name)


if __name__ == "__main__":
    best = [seg.select_by_val(YOLO_RUNS), seg.select_by_val(RF_RUNS)]
    check_consistency(best if "--hizli" in sys.argv else RUNS)
    for f in (egriler, sinif_bazinda, karisma, dogruluk_hiz, en_kotu10):
        f(); print("✓", f.__name__)
    print("Şekiller:", sorted(p.name for p in FIG.glob("seg_*.png")))
