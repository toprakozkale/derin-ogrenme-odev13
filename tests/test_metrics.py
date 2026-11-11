"""Metrik fonksiyonlarının elle hesaplanmış değerlere karşı testleri."""
import numpy as np
import pytest

from src import config as C
from src.metrics import (bootstrap_ci, cls_metrics, confusion, group_confusion, group_confusion_rate, mcnemar_test,
                         top_confusions)

AL, AF, BB, GM, LS, PMF, PML = range(7)   # alfabetik sınıf kimlikleri


def test_dogruluk_ve_f1_bilinen_degerler():
    y = np.array([LS, LS, LS, PML, PML, GM])
    p = np.array([LS, LS, PML, PML, PML, GM])
    m = cls_metrics(y, p)
    assert m["accuracy"] == pytest.approx(5 / 6)
    pc = m["per_class"]
    # Leaf Spot: P=2/2, R=2/3 ; PM Leaf: P=2/3, R=2/2
    assert pc.loc["Leaf Spot", "precision"] == pytest.approx(1.0)
    assert pc.loc["Leaf Spot", "recall"] == pytest.approx(2 / 3)
    assert pc.loc["Powdery Mildew Leaf", "precision"] == pytest.approx(2 / 3)
    assert pc.loc["Leaf Spot", "support"] == 3


def test_grup_matrisi_yaprak_meyve_cicek():
    # gerçek Gray Mold (meyve) → tahmin Leaf Spot (yaprak); gerçek Blossom Blight (çiçek) → doğru
    y = np.array([GM, GM, BB, LS])
    p = np.array([LS, GM, BB, AL])          # LS→AL yaprak içi karışma: grup düzeyinde doğru
    cm = confusion(y, p)
    g = group_confusion(cm)
    assert g.loc["gerçek: meyve", "tahmin: yaprak"] == 1
    assert g.loc["gerçek: yaprak", "tahmin: yaprak"] == 1
    r = group_confusion_rate(cm)
    assert r["meyve->yaprak"] == pytest.approx(0.5)
    assert r["gruplar_arasi_toplam"] == pytest.approx(1 / 4)
    top = top_confusions(cm)
    assert set(top["grup"]) == {"meyve→yaprak", "yaprak→yaprak"}


def test_bootstrap_araligi_noktayi_kapsar():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 7, 300)
    p = np.where(rng.random(300) < 0.8, y, (y + 1) % 7)
    point, lo, hi = bootstrap_ci(y, p, n_boot=500)
    assert lo <= point <= hi and hi - lo < 0.15


def test_mcnemar_tablosu():
    y = np.zeros(10, int)
    a = np.array([0] * 8 + [1] * 2)          # 8 doğru
    b = np.array([0] * 5 + [1] * 5)          # 5 doğru; A'nın doğru B'nin yanlış olduğu 3 örnek
    r = mcnemar_test(y, a, b)
    assert r["yalniz_A_dogru"] == 3 and r["yalniz_B_dogru"] == 0 and r["ikisi_dogru"] == 5
    assert 0 < r["p_degeri"] <= 1


def test_sinif_sirasi_alfabetik():
    assert list(C.CLASSES) == sorted(C.CLASSES)
