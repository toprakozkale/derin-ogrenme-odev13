"""Değerlendirme metrikleri — sınıflandırma (Aşama 2) ve segmentasyon (Aşama 3) için ortak.

Tüm metrikler kendi kodumuz + scikit-learn/statsmodels ile hesaplanır (model kütüphanesinin kendi raporuna
güvenilmez); böylece farklı modeller aynı kodla ölçülür.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support,
                             roc_auc_score)

from . import config as C


# ----------------------------------------------------------------------------- sınıflandırma

def cls_metrics(y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray | None = None,
                n_classes: int = len(C.CLASSES)) -> dict:
    """Doğruluk, sınıf bazında P/R/F1, macro/weighted F1, (ops.) makro OvR ROC-AUC."""
    labels = list(range(n_classes))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "per_class": pd.DataFrame({"precision": p, "recall": r, "f1": f, "support": s}, index=list(C.CLASSES[:n_classes])),
        "n": int(len(y_true)),
    }
    if proba is not None:
        present = np.unique(y_true)
        try:
            out["macro_roc_auc_ovr"] = float(roc_auc_score(y_true, proba[:, present] / proba[:, present].sum(1, keepdims=True),
                                                           multi_class="ovr", average="macro", labels=present)) \
                if len(present) > 2 else float("nan")
        except ValueError:
            out["macro_roc_auc_ovr"] = float("nan")
    return out


def confusion(y_true, y_pred, n_classes: int = len(C.CLASSES), normalize: bool = False) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes))).astype(float)
    if normalize:
        cm = cm / np.clip(cm.sum(1, keepdims=True), 1, None)
    return cm


def group_confusion(cm: np.ndarray) -> pd.DataFrame:
    """7×7 matrisi yaprak / meyve / çiçek gruplarına indirger (hocanın "Dikkat" maddesi)."""
    g = [C.GROUP_OF[c] for c in C.CLASSES]
    idx = {name: [i for i, x in enumerate(g) if x == name] for name in C.GROUPS}
    m = np.array([[cm[np.ix_(idx[a], idx[b])].sum() for b in C.GROUPS] for a in C.GROUPS])
    return pd.DataFrame(m, index=[f"gerçek: {x}" for x in C.GROUPS], columns=[f"tahmin: {x}" for x in C.GROUPS])


def group_confusion_rate(cm: np.ndarray) -> dict:
    """Gruplar arası karışma: gerçek grubu X olan örneklerin yüzde kaçı başka bir gruba atandı."""
    gc = group_confusion(cm).to_numpy()
    rows = gc.sum(1)
    out = {f"{a}->{b}": float(gc[i, j] / rows[i]) if rows[i] else float("nan")
           for i, a in enumerate(C.GROUPS) for j, b in enumerate(C.GROUPS) if i != j}
    out["gruplar_arasi_toplam"] = float((gc.sum() - np.trace(gc)) / gc.sum()) if gc.sum() else float("nan")
    return out


def top_confusions(cm: np.ndarray, k: int = 6) -> pd.DataFrame:
    rows = []
    for i, a in enumerate(C.CLASSES):
        for j, b in enumerate(C.CLASSES):
            if i != j and cm[i, j] > 0:
                rows.append({"gerçek": a, "tahmin": b, "adet": int(cm[i, j]),
                             "oran": cm[i, j] / max(cm[i].sum(), 1),
                             "grup": f"{C.GROUP_OF[a]}→{C.GROUP_OF[b]}"})
    return pd.DataFrame(rows).sort_values("adet", ascending=False).head(k).reset_index(drop=True) if rows else pd.DataFrame()


# ----------------------------------------------------------------------------- istatistik

def bootstrap_ci(y_true, y_pred, metric: str = "accuracy", n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = C.SEED) -> tuple[float, float, float]:
    """Test örneklerini yerine koyarak yeniden örnekleyip metriğin %95 güven aralığı (yüzdelik yöntemi)."""
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    fn = {"accuracy": lambda t, p: (t == p).mean(),
          "macro_f1": lambda t, p: f1_score(t, p, labels=list(range(len(C.CLASSES))), average="macro", zero_division=0)}[metric]
    vals = np.array([fn(y_true[i], y_pred[i]) for i in (rng.integers(0, n, n) for _ in range(n_boot))])
    return float(fn(y_true, y_pred)), float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def mcnemar_test(y_true, pred_a, pred_b) -> dict:
    """İki modelin aynı örneklerdeki doğru/yanlış tablosu üzerinden McNemar testi (kesin binom sürümü)."""
    from statsmodels.stats.contingency_tables import mcnemar
    a_ok = np.asarray(pred_a) == np.asarray(y_true)
    b_ok = np.asarray(pred_b) == np.asarray(y_true)
    table = [[int(np.sum(a_ok & b_ok)), int(np.sum(a_ok & ~b_ok))],
             [int(np.sum(~a_ok & b_ok)), int(np.sum(~a_ok & ~b_ok))]]
    res = mcnemar(table, exact=True)
    return {"yalniz_A_dogru": table[0][1], "yalniz_B_dogru": table[1][0], "ikisi_dogru": table[0][0],
            "ikisi_yanlis": table[1][1], "p_degeri": float(res.pvalue)}


def mean_std(values) -> str:
    v = np.asarray(values, dtype=float)
    return f"{v.mean():.4f} ± {v.std(ddof=1):.4f}" if len(v) > 1 else f"{v.mean():.4f}"


# ----------------------------------------------------------------------------- segmentasyon (Aşama 3)
# Adaletin kalbi: YOLO-seg ve RF-DETR tahminleri aynı biçime (COCO "results": RLE maske + sınıf + skor) çevrilir ve
# BU fonksiyonlarla ölçülür. Kütüphanelerin kendi raporladığı değerler yalnız ek tablodur.

BG = len(C.CLASSES)                      # arka plan indeksi (7) — piksel haritalarında "hastalık yok"
SEG_LABELS = list(C.CLASSES) + ["arka plan"]


def mask_to_rle(mask: np.ndarray) -> dict:
    """Bool maske → COCO RLE (counts JSON'a yazılabilsin diye str)."""
    from pycocotools import mask as mu
    rle = mu.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def rle_to_mask(rle: dict) -> np.ndarray:
    from pycocotools import mask as mu
    r = dict(rle)
    if isinstance(r["counts"], str):
        r["counts"] = r["counts"].encode("ascii")
    return mu.decode(r).astype(bool)


def semantic_map(masks, classes, order_key=None, shape=None) -> np.ndarray:
    """Örnek maskelerini tek bir piksel sınıf haritasına çevirir. Boş piksel = BG.

    Çakışmada order_key'i büyük olan örnek kazanır (tahminde skor; gerçekte -alan → küçük örnek büyüğün üstünde kalır).
    """
    if shape is None:
        shape = masks[0].shape
    sem = np.full(shape, BG, dtype=np.uint8)
    order = np.argsort(order_key, kind="stable") if order_key is not None else range(len(masks))
    for k in order:
        sem[masks[k]] = classes[k]
    return sem


def pixel_confusion(gt_sem: np.ndarray, pred_sem: np.ndarray, n: int = BG + 1) -> np.ndarray:
    """Piksel düzeyinde (n × n) karışma: satır = gerçek, sütun = tahmin; son satır/sütun arka plan."""
    idx = gt_sem.ravel().astype(np.int64) * n + pred_sem.ravel().astype(np.int64)
    return np.bincount(idx, minlength=n * n).reshape(n, n)


def seg_semantic_metrics(cm: np.ndarray) -> dict:
    """Toplanmış piksel karışma matrisinden sınıf bazında IoU ve Dice, mIoU, ortalama Dice.

    mIoU ve mDice yalnız 7 hastalık sınıfının ortalamasıdır (arka plan hariç; dahil olan da ayrıca verilir).
    Testte hiç pikseli olmayan ve hiç tahmin edilmeyen sınıf ortalamaya girmez (NaN).
    Ayrıca sınıftan bağımsız "hastalık ↔ arka plan" IoU'su (fg_iou): leke doğru yerde mi, sınıfı ne olursa olsun.
    """
    cm = cm.astype(float)
    tp = np.diag(cm)
    fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = np.where(tp + fp + fn > 0, tp / (tp + fp + fn), np.nan)
        dice = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), np.nan)
    fg_tp = cm[:BG, :BG].sum()
    fg_den = cm.sum() - cm[BG, BG]
    per = pd.DataFrame({"iou": iou, "dice": dice, "gt_piksel": cm.sum(1).astype(int), "tahmin_piksel": cm.sum(0).astype(int)},
                       index=SEG_LABELS[:len(cm)])
    return {"miou": float(np.nanmean(iou[:BG])), "mdice": float(np.nanmean(dice[:BG])),
            "miou_arka_plan_dahil": float(np.nanmean(iou)), "fg_iou": float(fg_tp / fg_den) if fg_den else float("nan"),
            "per_class": per}


def image_iou(gt_sem: np.ndarray, pred_sem: np.ndarray) -> float:
    """Tek görüntünün notu: gerçekte ya da tahminde görünen hastalık sınıflarının ortalama IoU'su (arka plan hariç).

    İkisinde de hastalık yoksa 1. En kötü 10 görüntü bu notla seçilir.
    """
    present = sorted((set(np.unique(gt_sem)) | set(np.unique(pred_sem))) - {BG})
    if not present:
        return 1.0
    vals = []
    for c in present:
        g, p = gt_sem == c, pred_sem == c
        vals.append(np.logical_and(g, p).sum() / np.logical_or(g, p).sum())
    return float(np.mean(vals))


def match_instances(gt_rles: list, gt_cls, pr_rles: list, pr_cls, iou_thr: float = 0.5) -> list[tuple[int, int, float]]:
    """Sınıftan bağımsız açgözlü eşleştirme (IoU büyükten küçüğe), her örnek en çok bir kez.

    Dönüş: (gerçek sınıf, tahmin sınıfı, IoU) üçlüleri. Eşleşmeyen gerçek örnek → (sınıf, BG, 0) = "kaçırıldı";
    eşleşmeyen tahmin → (BG, sınıf, 0) = "fazladan tahmin". Sınıf karışma matrisi bu çiftlerden kurulur.
    """
    from pycocotools import mask as mu

    def _enc(rs):
        return [{**r, "counts": r["counts"].encode("ascii") if isinstance(r["counts"], str) else r["counts"]} for r in rs]

    g_n, p_n = len(gt_rles), len(pr_rles)
    out: list[tuple[int, int, float]] = []
    used_g, used_p = set(), set()
    if g_n and p_n:
        ious = np.asarray(mu.iou(_enc(pr_rles), _enc(gt_rles), [0] * g_n)).reshape(p_n, g_n)
        for flat in np.argsort(-ious, axis=None, kind="stable"):
            p, g = divmod(int(flat), g_n)
            if ious[p, g] < iou_thr:
                break
            if p in used_p or g in used_g:
                continue
            used_p.add(p); used_g.add(g)
            out.append((int(gt_cls[g]), int(pr_cls[p]), float(ious[p, g])))
    out += [(int(gt_cls[g]), BG, 0.0) for g in range(g_n) if g not in used_g]
    out += [(BG, int(pr_cls[p]), 0.0) for p in range(p_n) if p not in used_p]
    return out


def instance_confusion(pairs: list[tuple[int, int, float]], n: int = BG + 1) -> np.ndarray:
    """Eşleştirme çiftlerinden (n × n) matris; [BG, BG] hücresi tanımsızdır ve 0 kalır."""
    cm = np.zeros((n, n), dtype=int)
    for g, p, _ in pairs:
        cm[g, p] += 1
    return cm


def coco_mask_map(gt: dict, results: list[dict], img_ids=None) -> dict:
    """pycocotools ile maske mAP50 ve mAP50-95 (+ sınıf bazında AP). gt: COCO sözlüğü (kategori 1..7 + 0 üst kategori).

    results: [{"image_id", "category_id" (1..7), "segmentation" (RLE), "score"}]. img_ids verilirse yalnız o görüntüler
    (ör. temiz test) değerlendirilir.
    """
    import contextlib, copy, io
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    cat_ids = list(range(1, BG + 1))
    with contextlib.redirect_stdout(io.StringIO()):
        G = COCO()
        G.dataset = copy.deepcopy(gt)        # pycocotools sözlükleri yerinde değiştirir (bbox/area ekler) → kopya
        # DİKKAT: COCOeval eşleşmeyi gerçek örneğin id'siyle kaydeder ve 0'ı "eşleşmedi" sayar → id'si 0 olan örnek
        # hiç doğru bulunamaz. Bizim dönüşümümüz (ve Roboflow) id'leri 0'dan başlatır; burada 1'den yeniden numaralanır.
        for i, a in enumerate(G.dataset["annotations"]):
            a["id"] = i + 1
        G.createIndex()
    ids = sorted(img_ids) if img_ids is not None else sorted(G.getImgIds())
    keep = set(ids)
    res = [{k: (dict(v) if isinstance(v, dict) else v) for k, v in r.items() if k not in ("bbox", "area", "id")}
           for r in results if r["image_id"] in keep]
    nan_pc = pd.DataFrame({"ap50": np.nan, "ap50_95": np.nan}, index=list(C.CLASSES))
    if not res:
        return {"map50": 0.0, "map50_95": 0.0, "per_class": nan_pc.fillna(0.0)}
    with contextlib.redirect_stdout(io.StringIO()):
        D = G.loadRes(res)
        E = COCOeval(G, D, "segm")
        E.params.catIds = cat_ids
        E.params.imgIds = ids
        E.evaluate(); E.accumulate(); E.summarize()
    prec = E.eval["precision"]          # [IoU eşiği T, recall R, sınıf K, alan A, maxDets M]
    ap50_95, ap50 = [], []
    for k in range(len(cat_ids)):
        p = prec[:, :, k, 0, -1]
        ap50_95.append(float(p[p > -1].mean()) if (p > -1).any() else np.nan)
        p0 = prec[0, :, k, 0, -1]
        ap50.append(float(p0[p0 > -1].mean()) if (p0 > -1).any() else np.nan)
    return {"map50_95": float(E.stats[0]), "map50": float(E.stats[1]),
            "per_class": pd.DataFrame({"ap50": ap50, "ap50_95": ap50_95}, index=list(C.CLASSES))}
