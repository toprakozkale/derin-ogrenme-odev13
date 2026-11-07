"""Veri hattının sentetik verilerle testleri (gerçek veri gerektirmez).

Çalıştırma:  .venv/bin/python -m pytest -q tests
"""
import json

import numpy as np
import pytest
from PIL import Image

from src import config as C
from src.convert import mask_iou, polygon_to_mask, verify_roundtrip, write_cls, write_coco, write_yolo_seg
from src.dataset import fix_polygon, load_all
from src.dedup import image_hashes, leaked, near_duplicate_pairs

W = H = 100


def test_mask_iou_bilinen_deger():
    a = np.zeros((10, 10), bool); a[0:4, 0:4] = True        # 16 piksel
    b = np.zeros((10, 10), bool); b[2:6, 2:6] = True        # 16 piksel, kesişim 4
    assert mask_iou(a, b) == pytest.approx(4 / 28)
    assert mask_iou(a, a) == 1.0


def test_gecerli_poligon_degismez():
    sq = [[10, 10], [50, 10], [50, 50], [10, 50]]
    pts, actions, a_raw, a = fix_polygon(sq, W, H)
    assert actions == [] and a == pytest.approx(1600) and a_raw == pytest.approx(1600)
    assert np.allclose(pts, sq)


def test_goruntu_disi_kirpilir():
    pts, actions, _, a = fix_polygon([[-5, -5], [50, -5], [50, 50], [-5, 50]], W, H)
    assert "goruntu_disi:kirpildi" in actions
    assert pts.min() >= 0 and a == pytest.approx(2500)


def test_kendini_kesen_onarilir_buyuk_parca_kalir():
    # "papyon" şekli: iki üçgen, sol üçgen daha büyük
    bow = [[0, 0], [60, 60], [60, 40], [0, 100]]
    pts, actions, _, a = fix_polygon(bow, W, H)
    assert pts is not None
    assert any(s.startswith("kendini_kesen:onarildi") for s in actions)


def test_uc_noktadan_az_atilir():
    pts, actions, _, _ = fix_polygon([[0, 0], [10, 10]], W, H)
    assert pts is None and actions == ["3ten_az_nokta:atildi"]


def _make_fake_dataset(root):
    """Her bölmede 2 görüntü; biri iki sınıflı (görüntü etiketi alanı büyük olan olmalı)."""
    rng = np.random.default_rng(0)
    shapes_a = [  # Leaf Spot alanı 400, Powdery Mildew Fruit alanı 1600 → görüntü etiketi PMF
        {"label": "Leaf Spot", "points": [[5, 5], [25, 5], [25, 25], [5, 25]], "shape_type": "polygon"},
        {"label": "Powdery Mildew Fruit", "points": [[40, 40], [80, 40], [80, 80], [40, 80]], "shape_type": "polygon"},
    ]
    shapes_b = [
        {"label": "Gray Mold", "points": [[10, 10], [90, 12], [70, 95], [-3, 60]], "shape_type": "polygon"},  # taşan nokta
    ]
    for s_i, split in enumerate(C.SPLITS):
        d = root / split; d.mkdir(parents=True)
        for k, shapes in enumerate((shapes_a, shapes_b)):
            stem = f"img_{split}_{k}"
            arr = (rng.random((H, W, 3)) * 255).astype("uint8")
            Image.fromarray(arr).save(d / f"{stem}.jpg")
            (d / f"{stem}.json").write_text(json.dumps({
                "version": "4.5.6", "flags": {}, "shapes": shapes, "imagePath": f"{stem}.jpg",
                "imageData": None, "imageHeight": H, "imageWidth": W}))


def test_uctan_uca_donusum_ve_geri_donus(tmp_path):
    raw = tmp_path / "raw"; _make_fake_dataset(raw)
    recs = load_all(raw)
    a = next(r for r in recs["train"] if r.stem.endswith("_0"))
    assert a.image_label == "Powdery Mildew Fruit"           # alan kuralı
    b = next(r for r in recs["train"] if r.stem.endswith("_1"))
    assert b.instances[0].actions == ["goruntu_disi:kirpildi"]

    cls_counts = write_cls(recs, tmp_path / "cls")
    assert sum(cls_counts["train"].values()) == 2
    yaml = write_yolo_seg(recs, tmp_path / "yolo")
    assert "nc: 7" in yaml.read_text()
    coco = write_coco(recs, tmp_path / "coco")
    assert (tmp_path / "coco" / "valid" / "_annotations.coco.json").exists()

    rows = verify_roundtrip(recs, tmp_path / "yolo", coco)
    assert len(rows) == 3 * len(C.SPLITS)
    assert all(r["class_ok"] and r["count_ok"] for r in rows)
    assert min(r["iou_yolo"] for r in rows) > 0.99
    assert min(r["iou_coco"] for r in rows) > 0.99


def test_yinelenen_tespiti(tmp_path):
    raw = tmp_path / "raw"; _make_fake_dataset(raw)
    # test bölmesine train'deki bir görüntünün birebir kopyasını koy
    src = raw / "train" / "img_train_0.jpg"
    (raw / "test" / "kopya.jpg").write_bytes(src.read_bytes())
    (raw / "test" / "kopya.json").write_text((raw / "train" / "img_train_0.json").read_text().replace("img_train_0", "kopya"))
    recs = load_all(raw)
    pairs = near_duplicate_pairs(image_hashes(recs), max_dist=0)
    hit = pairs[(pairs["a_stem"] == "img_train_0") & (pairs["b_stem"] == "kopya")]
    assert len(hit) == 1 and bool(hit["same_md5"].iloc[0])
    assert leaked(pairs, 0, "test", ("train", "val")) == {"kopya"}


def test_polygon_to_mask_alan():
    m = polygon_to_mask(np.array([[10, 10], [30, 10], [30, 30], [10, 30]], float), H, W)
    assert abs(m.sum() - 400) <= 45   # rasterleştirme kenar pikselleri
