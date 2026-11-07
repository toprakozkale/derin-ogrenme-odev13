"""LabelMe → (1) Ultralytics sınıflandırma klasörleri, (2) YOLO-seg, (3) COCO (RF-DETR / Roboflow düzeni).

Ayrıca her formattan maskeyi geri üretip kaynak poligonla IoU karşılaştıran doğrulama fonksiyonu içerir.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils

from . import config as C
from .dataset import Record

COCO_SPLIT_DIR = {"train": "train", "val": "valid", "test": "test"}   # RF-DETR / Roboflow klasör adları
COCO_SUPERCATEGORY = "diseases"


def coco_cat_id(label: str) -> int:
    """COCO kategori kimliği. Roboflow düzeninde 0 = üst kategori, sınıflar 1..7 (YOLO kimliği + 1)."""
    return C.CLASS_TO_ID[label] + 1


# ----------------------------------------------------------------------------- maske yardımcıları

def polygon_to_rle(points: np.ndarray, height: int, width: int):
    flat = np.asarray(points, dtype=float).reshape(-1).tolist()
    return mask_utils.merge(mask_utils.frPyObjects([flat], height, width))


def polygon_to_mask(points: np.ndarray, height: int, width: int) -> np.ndarray:
    return mask_utils.decode(polygon_to_rle(points, height, width)).astype(bool)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 1.0


# ----------------------------------------------------------------------------- 1) sınıflandırma

def write_cls(records_by_split: dict[str, list[Record]], out: Path) -> dict:
    """Ultralytics -cls klasör yapısı: out/{train,val,test}/<sinif_slug>/<stem>.jpg"""
    if out.exists():
        shutil.rmtree(out)
    counts = {}
    for split, recs in records_by_split.items():
        for c in C.CLASSES:
            (out / split / C.CLASS_SLUG[c]).mkdir(parents=True, exist_ok=True)
        for r in recs:
            lab = r.image_label
            if lab is None:
                continue
            shutil.copy2(r.image_path, out / split / C.CLASS_SLUG[lab] / r.image_path.name)
            counts.setdefault(split, {}).setdefault(lab, 0)
            counts[split][lab] += 1
    return counts


# ----------------------------------------------------------------------------- 2) YOLO-seg

def yolo_seg_lines(rec: Record) -> list[str]:
    lines = []
    for inst in rec.instances:
        xy = np.column_stack([inst.points[:, 0] / rec.width, inst.points[:, 1] / rec.height]).clip(0, 1)
        lines.append(f"{C.CLASS_TO_ID[inst.label]} " + " ".join(f"{x:.6f} {y:.6f}" for x, y in xy))
    return lines


def parse_yolo_seg_line(line: str, width: int, height: int) -> tuple[int, np.ndarray]:
    vals = line.split()
    cid = int(vals[0])
    xy = np.asarray(vals[1:], dtype=float).reshape(-1, 2) * [width, height]
    return cid, xy


def write_yolo_seg(records_by_split: dict[str, list[Record]], out: Path) -> Path:
    if out.exists():
        shutil.rmtree(out)
    for split, recs in records_by_split.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for r in recs:
            shutil.copy2(r.image_path, out / "images" / split / r.image_path.name)
            (out / "labels" / split / f"{r.stem}.txt").write_text("\n".join(yolo_seg_lines(r)) + "\n")
    yaml_path = out / "data.yaml"
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(C.CLASSES))
    yaml_path.write_text(
        "# Otomatik üretildi: src/convert.py — elle düzenlemeyin\n"
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"nc: {len(C.CLASSES)}\nnames:\n{names}\n"
    )
    return yaml_path


# ----------------------------------------------------------------------------- 3) COCO

def coco_dict(recs: list[Record]) -> dict:
    cats = [{"id": 0, "name": COCO_SUPERCATEGORY, "supercategory": "none"}] + [
        {"id": coco_cat_id(c), "name": c, "supercategory": COCO_SUPERCATEGORY} for c in C.CLASSES
    ]
    images, anns = [], []
    ann_id = 0
    for img_id, r in enumerate(recs):
        images.append({"id": img_id, "file_name": r.image_path.name, "width": r.width, "height": r.height})
        for inst in r.instances:
            rle = polygon_to_rle(inst.points, r.height, r.width)
            x0, y0 = inst.points.min(0)
            x1, y1 = inst.points.max(0)
            anns.append({
                "id": ann_id, "image_id": img_id, "category_id": coco_cat_id(inst.label),
                "segmentation": [np.round(inst.points.reshape(-1), 3).tolist()],
                "area": float(mask_utils.area(rle)),
                "bbox": [round(float(x0), 3), round(float(y0), 3), round(float(x1 - x0), 3), round(float(y1 - y0), 3)],
                "iscrowd": 0,
            })
            ann_id += 1
    return {
        "info": {"description": "Strawberry Disease Detection Dataset (Afzaal et al., 2021) — LabelMe'den dönüştürüldü"},
        "licenses": [], "categories": cats, "images": images, "annotations": anns,
    }


def write_coco(records_by_split: dict[str, list[Record]], out: Path) -> dict[str, Path]:
    if out.exists():
        shutil.rmtree(out)
    paths = {}
    for split, recs in records_by_split.items():
        d = out / COCO_SPLIT_DIR[split]
        d.mkdir(parents=True, exist_ok=True)
        for r in recs:
            shutil.copy2(r.image_path, d / r.image_path.name)
        p = d / "_annotations.coco.json"
        p.write_text(json.dumps(coco_dict(recs)))
        paths[split] = p
    return paths


# ----------------------------------------------------------------------------- doğrulama

def verify_roundtrip(records_by_split: dict[str, list[Record]], yolo_root: Path, coco_paths: dict[str, Path]) -> list[dict]:
    """Her örnek için: kaynak (düzeltilmiş) poligon maskesi ↔ YOLO txt'den ve COCO json'dan geri üretilen maske IoU'su.

    Ayrıca sınıf kimliği eşleşmesini ve örnek sayısı tutarlılığını kontrol eder.
    """
    from pycocotools.coco import COCO
    import contextlib, io

    rows = []
    for split, recs in records_by_split.items():
        with contextlib.redirect_stdout(io.StringIO()):
            coco = COCO(str(coco_paths[split]))
        name_to_img = {im["file_name"]: im["id"] for im in coco.dataset["images"]}
        for r in recs:
            lines = (yolo_root / "labels" / split / f"{r.stem}.txt").read_text().strip().splitlines()
            anns = coco.loadAnns(coco.getAnnIds(imgIds=[name_to_img[r.image_path.name]]))
            ok_count = len(lines) == len(r.instances) == len(anns)
            for k, inst in enumerate(r.instances):
                src = polygon_to_mask(inst.points, r.height, r.width)
                cid_y, xy = parse_yolo_seg_line(lines[k], r.width, r.height)
                iou_y = mask_iou(src, polygon_to_mask(xy, r.height, r.width))
                iou_c = mask_iou(src, coco.annToMask(anns[k]).astype(bool))
                raw_mask_iou = (mask_iou(src, polygon_to_mask(inst.raw_points, r.height, r.width))
                                if inst.actions and len(inst.raw_points) >= 3 else 1.0)
                rows.append({
                    "split": split, "stem": r.stem, "index": k, "label": inst.label,
                    "iou_yolo": iou_y, "iou_coco": iou_c,
                    "class_ok": (cid_y == C.CLASS_TO_ID[inst.label]) and (anns[k]["category_id"] == coco_cat_id(inst.label)),
                    "count_ok": ok_count,
                    "fixed": bool(inst.actions), "iou_fixed_vs_raw": raw_mask_iou,
                })
    return rows
