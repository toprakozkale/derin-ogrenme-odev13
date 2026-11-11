"""Değerlendirme metrikleri — sınıflandırma (Aşama 2) ve ileride segmentasyon (Aşama 3) için ortak.

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
