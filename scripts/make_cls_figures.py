"""Colab'dan alınan runs/cls sonuçlarından rapor şekillerini yerelde yeniden üretir (yeniden eğitim/tahmin YOK).

02 notebook'u şekilleri Drive'daki rapor/figures'a yazar; Drive'dan yalnız runs/cls indirildiğinde şekiller
bu betikle aynı kodla, kaydedilmiş test tahminlerinden (runs/cls/preds) çizilir. Ayrıca bir tutarlılık
denetimi yapar: tahmin dosyalarından yeniden hesaplanan doğruluk results.csv ile aynı olmalı.

    .venv/bin/python scripts/make_cls_figures.py
"""
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
from PIL import Image  # noqa: E402

from src import cls  # noqa: E402
from src import config as C  # noqa: E402
from src import metrics as M  # noqa: E402

FIG = ROOT / "rapor" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
MAIN, ABL = cls.MAIN_RUNS, cls.ABLATION_RUNS


def check_consistency() -> None:
    res = pd.read_csv(cls.OUT / "results.csv")
    clean = cls.clean_test_stems()
    assert len(clean) == 692, f"temiz test {len(clean)} (692 olmalı) — önce scripts/prepare_data.py"
    for spec in MAIN + ABL:
        p = cls.load_preds(spec)
        assert len(p) == 743, (spec.name, len(p))
        for subset, d in (("tum_test", p), ("temiz_test", p[p.stem.isin(clean)])):
            acc = float((d.y_true == d.y_pred).mean())
            ref = res[(res.kosu == spec.name) & (res.alt_kume == subset)].accuracy.item()
            assert abs(acc - ref) < 1e-9, (spec.name, subset, acc, ref)
    best = cls.select_by_val(MAIN)
    print(f"✓ tutarlılık: 9 koşu × 2 alt küme doğruluğu results.csv ile aynı · val ile seçilen: {best.name}")


def curves() -> None:
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    for spec in MAIN:
        log = cls.train_log(spec)
        c = "#2a78d6" if spec.size == "n" else "#eb6834"
        acc = [k for k in log.columns if "accuracy_top1" in k][0]
        ax[0].plot(log["epoch"], log["train/loss"], color=c, alpha=.8, lw=1.5, label=f"{spec.size} s{spec.seed}")
        ax[1].plot(log["epoch"], log[acc], color=c, alpha=.8, lw=1.5, label=f"{spec.size} s{spec.seed}")
    ax[0].set_title("eğitim kaybı"); ax[1].set_title("doğrulama doğruluğu (top-1)")
    for a in ax:
        a.set_xlabel("epoch"); a.grid(alpha=.25); a.spines[["top", "right"]].set_visible(False)
    ax[1].legend(frameon=False, fontsize=8, ncol=2)
    plt.tight_layout(); plt.savefig(FIG / "cls_egriler.png", dpi=200); plt.close()


def confusions() -> None:
    blue = LinearSegmentedColormap.from_list("seq", ["#f7f6f2", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    short = ["Ang. Leafspot", "Anthracnose", "Blossom Bl.", "Gray Mold", "Leaf Spot", "PM Fruit", "PM Leaf"]
    for size in ("n", "s"):
        runs = [s for s in MAIN if s.size == size]
        p = pd.concat([cls.load_preds(s) for s in runs])
        cm = M.confusion(p.y_true, p.y_pred)
        cmn = cm / np.clip(cm.sum(1, keepdims=True), 1, None)
        fig, ax = plt.subplots(figsize=(5.6, 4.8))
        im = ax.imshow(cmn, cmap=blue, vmin=0, vmax=1)
        for i in range(7):
            for j in range(7):
                if cm[i, j]:
                    ax.text(j, i, f"{cmn[i, j]:.2f}\n({int(cm[i, j])})", ha="center", va="center", fontsize=6.5,
                            color="white" if cmn[i, j] > .55 else "#0b0b0b")
        ax.set_xticks(range(7), short, rotation=45, ha="right", fontsize=8); ax.set_yticks(range(7), short, fontsize=8)
        ax.set_xlabel("tahmin"); ax.set_ylabel("gerçek")
        ax.set_title(f"{runs[0].model} — {len(runs)} seed toplamı, satır normalize", fontsize=9)
        plt.colorbar(im, fraction=.046); plt.tight_layout()
        plt.savefig(FIG / f"cls_karmasiklik_{size}.pdf"); plt.savefig(FIG / f"cls_karmasiklik_{size}.png", dpi=200); plt.close()


def worst10() -> None:
    best = cls.select_by_val(MAIN)
    worst = cls.worst_predictions(cls.load_preds(best), 10)
    saved = pd.read_csv(cls.OUT / "worst10.csv")
    assert worst.stem.tolist() == saved.stem.tolist(), "en kötü 10 listesi Colab'dakiyle aynı değil"
    fig, axes = plt.subplots(2, 5, figsize=(15, 6.8))
    for ax, (_, r) in zip(axes.ravel(), worst.iterrows()):
        local = C.PROCESSED / "cls" / "test" / C.CLASS_SLUG[r.gercek] / Path(r.path).name  # Colab yolu → yerel yol
        ax.imshow(Image.open(local)); ax.axis("off")
        ax.set_title(f"{r.stem}\ngerçek: {r.gercek}\ntahmin: {r.tahmin} ({r.tahmin_guveni:.2f})"
                     + ("  [gruplar arası]" if r.gruplar_arasi else ""), fontsize=7.5)
    plt.tight_layout(); plt.savefig(FIG / "cls_en_kotu10.png", dpi=150); plt.close()


if __name__ == "__main__":
    check_consistency()
    curves(); confusions(); worst10()
    print("✓ şekiller:", ", ".join(sorted(p.name for p in FIG.glob("cls_*"))))
