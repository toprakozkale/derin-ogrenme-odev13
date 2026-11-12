"""Segmentasyon metriklerinin (src/metrics.py) elle hesaplanmış değerlere karşı testleri.

20×20'lik yapay görüntüler: gerçek maske 10×10 (100 piksel), tahmin bunun içinde 9×8 (72 piksel) → IoU = 0,72.
"""
import numpy as np
import pytest

from src import config as C
from src.metrics import (BG, coco_mask_map, group_confusion, image_iou, instance_confusion, mask_to_rle,
                         match_instances, pixel_confusion, rle_to_mask, seg_semantic_metrics, semantic_map)

AL, AF, BB, GM, LS, PMF, PML = range(7)
H = W = 20


def box(r0, r1, c0, c1):
    m = np.zeros((H, W), bool)
    m[r0:r1, c0:c1] = True
    return m


GT_MASK = box(0, 10, 0, 10)          # 100 piksel
PR_MASK = box(0, 9, 0, 8)            # 72 piksel, tamamı gerçeğin içinde → IoU 0,72


def coco_gt(images: list[list[tuple[int, np.ndarray]]]) -> dict:
    cats = [{"id": 0, "name": "diseases", "supercategory": "none"}] + [
        {"id": i + 1, "name": c, "supercategory": "diseases"} for i, c in enumerate(C.CLASSES)]
    imgs, anns = [], []
    for img_id, insts in enumerate(images):
        imgs.append({"id": img_id, "file_name": f"{img_id}.jpg", "width": W, "height": H})
        for cls, m in insts:
            ys, xs = np.where(m)
            anns.append({"id": len(anns), "image_id": img_id, "category_id": cls + 1, "segmentation": mask_to_rle(m),
                         "area": float(m.sum()), "bbox": [float(xs.min()), float(ys.min()), float(np.ptp(xs) + 1),
                                                          float(np.ptp(ys) + 1)], "iscrowd": 0})
    return {"images": imgs, "annotations": anns, "categories": cats}


def test_rle_gidis_donus():
    assert np.array_equal(rle_to_mask(mask_to_rle(PR_MASK)), PR_MASK)


def test_piksel_iou_dice_bilinen_degerler():
    gt = semantic_map([GT_MASK], [LS])
    pr = semantic_map([PR_MASK], [LS])
    m = seg_semantic_metrics(pixel_confusion(gt, pr))
    pc = m["per_class"]
    assert pc.loc["Leaf Spot", "iou"] == pytest.approx(0.72)
    assert pc.loc["Leaf Spot", "dice"] == pytest.approx(2 * 72 / (72 + 100))
    assert pc.loc["arka plan", "iou"] == pytest.approx(300 / 328)
    assert m["miou"] == pytest.approx(0.72)            # testte görünmeyen sınıflar ortalamaya girmez
    assert m["fg_iou"] == pytest.approx(0.72)
    assert image_iou(gt, pr) == pytest.approx(0.72)


def test_cakismada_yuksek_skor_kazanir_yanlis_sinif_cezalanir():
    # Aynı bölgede iki tahmin: düşük skorlu doğru sınıf (LS), yüksek skorlu yanlış sınıf (AL) → piksel AL olur
    sem = semantic_map([GT_MASK, GT_MASK], [LS, AL], order_key=[0.3, 0.9])
    assert (sem[GT_MASK] == AL).all() and (sem[~GT_MASK] == BG).all()
    m = seg_semantic_metrics(pixel_confusion(semantic_map([GT_MASK], [LS]), sem))
    assert m["per_class"].loc["Leaf Spot", "iou"] == 0
    assert m["fg_iou"] == pytest.approx(1.0)            # leke yeri doğru, sınıfı yanlış


def test_ornek_eslestirme_karisma_kacirilan_fazladan():
    g_rles = [mask_to_rle(GT_MASK), mask_to_rle(box(12, 18, 12, 18))]
    p_rles = [mask_to_rle(PR_MASK), mask_to_rle(box(0, 3, 15, 20))]
    pairs = match_instances(g_rles, [GM, BB], p_rles, [PMF, LS])
    # GM ↔ PMF (IoU 0,72) eşleşir; BB kaçırılır; LS tahmini fazladan
    assert (GM, PMF) in [(g, p) for g, p, _ in pairs]
    assert any(g == BB and p == BG for g, p, _ in pairs)
    assert any(g == BG and p == LS for g, p, _ in pairs)
    cm = instance_confusion(pairs)
    assert cm[GM, PMF] == 1 and cm[BB, BG] == 1 and cm[BG, LS] == 1 and cm.sum() == 3
    gc = group_confusion(cm[:BG, :BG])
    assert gc.loc["gerçek: meyve", "tahmin: meyve"] == 1     # Gray Mold → PM Fruit: grup içi karışma


def test_coco_mask_map_bilinen_degerler():
    gt = coco_gt([[(LS, GT_MASK)], [(GM, GT_MASK)]])
    # Görüntü 0: LS için IoU 0,72 → yalnız 0,50…0,70 eşiklerinde (5/10) doğru. Görüntü 1: GM kusursuz.
    res = [{"image_id": 0, "category_id": LS + 1, "segmentation": mask_to_rle(PR_MASK), "score": 0.9},
           {"image_id": 1, "category_id": GM + 1, "segmentation": mask_to_rle(GT_MASK), "score": 0.8}]
    m = coco_mask_map(gt, res)
    assert m["map50"] == pytest.approx(1.0)
    assert m["per_class"].loc["Leaf Spot", "ap50_95"] == pytest.approx(0.5)
    assert m["per_class"].loc["Gray Mold", "ap50_95"] == pytest.approx(1.0)
    assert m["map50_95"] == pytest.approx(0.75)
    assert np.isnan(m["per_class"].loc["Blossom Blight", "ap50"])     # testte olmayan sınıf ortalamaya girmez
    # Alt küme (temiz test benzeri): yalnız görüntü 1
    assert coco_mask_map(gt, res, img_ids=[1])["map50_95"] == pytest.approx(1.0)
    # Yanlış sınıf → AP 0
    bad = [{**res[1], "category_id": PMF + 1}]
    assert coco_mask_map(gt, bad, img_ids=[1])["map50"] == pytest.approx(0.0)
