"""Ham LabelMe JSON etiketlerini okuma, denetleme ve kurallı düzeltme.

Her düzeltme kayıt altına alınır (hangi dosya, ne yapıldı, alan ne kadar değişti) — rapordaki
"etiket denetimi" tablosu bu kayıtlardan üretilir.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon
from shapely.validation import make_valid

from . import config as C


@dataclass
class Instance:
    label: str
    points: np.ndarray            # (N, 2) piksel koordinatı, düzeltilmiş
    raw_points: np.ndarray        # (N, 2) JSON'daki orijinal koordinatlar
    actions: list[str] = field(default_factory=list)
    area_raw: float = 0.0
    area: float = 0.0


@dataclass
class Record:
    split: str
    stem: str
    image_path: Path
    width: int
    height: int
    instances: list[Instance]
    dropped: list[dict] = field(default_factory=list)   # atılan örnekler ve nedeni
    notes: list[str] = field(default_factory=list)      # görüntü düzeyindeki notlar

    @property
    def image_label(self) -> str | None:
        """Görüntü düzeyi sınıf: toplam poligon alanı en büyük sınıf (bkz. config.IMAGE_LABEL_RULE)."""
        area = class_areas(self)
        return max(area, key=area.get) if area else None

    @property
    def classes(self) -> set[str]:
        return {i.label for i in self.instances}


def class_areas(rec: Record) -> dict[str, float]:
    area: dict[str, float] = defaultdict(float)
    for inst in rec.instances:
        area[inst.label] += inst.area
    return dict(area)


def _dedupe_consecutive(pts: np.ndarray) -> np.ndarray:
    keep = np.ones(len(pts), bool)
    keep[1:] = np.any(np.abs(np.diff(pts, axis=0)) > 1e-9, axis=1)
    pts = pts[keep]
    if len(pts) > 1 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    return pts


def fix_polygon(points, width: int, height: int) -> tuple[np.ndarray | None, list[str], float, float]:
    """Poligonu kurallara göre düzeltir.

    Kurallar (sırayla):
      1. 3'ten az nokta → at.
      2. Görüntü dışına taşan koordinatlar görüntü sınırına kırpılır.
      3. Kendini kesen (geçersiz) poligon shapely.make_valid ile onarılır; birden çok parçaya ayrılırsa
         en büyük alanlı parça tutulur (kalan alan oranı kayda geçer).
      4. Alanı 1 pikselden küçük kalırsa → at.
    Dönüş: (düzeltilmiş noktalar veya None, yapılan işlemler, ham alan, son alan)
    """
    actions: list[str] = []
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(pts) < 3:
        return None, ["3ten_az_nokta:atildi"], 0.0, 0.0
    area_raw = float(Polygon(pts).area) if len(pts) >= 3 else 0.0

    if (pts[:, 0].min() < 0) or (pts[:, 1].min() < 0) or (pts[:, 0].max() > width) or (pts[:, 1].max() > height):
        pts = np.column_stack([np.clip(pts[:, 0], 0, width), np.clip(pts[:, 1], 0, height)])
        actions.append("goruntu_disi:kirpildi")

    pts = _dedupe_consecutive(pts)
    if len(pts) < 3:
        return None, actions + ["3ten_az_nokta:atildi"], area_raw, 0.0

    poly = Polygon(pts)
    if not poly.is_valid:
        fixed = make_valid(poly)
        parts = [g for g in getattr(fixed, "geoms", [fixed]) if g.geom_type == "Polygon" and g.area > 0]
        # MultiPolygon içinde iç içe koleksiyonlar olabilir
        if not parts and hasattr(fixed, "geoms"):
            for g in fixed.geoms:
                parts += [p for p in getattr(g, "geoms", []) if p.geom_type == "Polygon" and p.area > 0]
        if not parts:
            return None, actions + ["kendini_kesen:onarilamadi:atildi"], area_raw, 0.0
        total = sum(p.area for p in parts)
        best = max(parts, key=lambda p: p.area)
        actions.append(f"kendini_kesen:onarildi:parca={len(parts)}:tutulan_oran={best.area / total:.3f}")
        pts = _dedupe_consecutive(np.asarray(best.exterior.coords, dtype=float))
        poly = Polygon(pts)

    if poly.area < 1.0:
        return None, actions + ["sifir_alan:atildi"], area_raw, float(poly.area)
    return pts, actions, area_raw, float(poly.area)


def load_record(json_path: Path, split: str) -> Record:
    j = json.loads(Path(json_path).read_text())
    stem = Path(json_path).stem
    img = Path(json_path).with_suffix(".jpg")
    W, H = int(j["imageWidth"]), int(j["imageHeight"])
    rec = Record(split=split, stem=stem, image_path=img, width=W, height=H, instances=[])
    if j.get("imagePath") != img.name:
        rec.notes.append(f"imagePath_farkli:{j.get('imagePath')}")
    if not img.exists():
        rec.notes.append("goruntu_yok")
    for k, s in enumerate(j.get("shapes", [])):
        label = s.get("label")
        if label not in C.CLASS_TO_ID:
            rec.dropped.append({"index": k, "label": label, "reason": "bilinmeyen_sinif"})
            continue
        if s.get("shape_type", "polygon") != "polygon":
            rec.dropped.append({"index": k, "label": label, "reason": f"sekil_turu:{s.get('shape_type')}"})
            continue
        raw = np.asarray(s.get("points", []), dtype=float).reshape(-1, 2)
        pts, actions, a_raw, a = fix_polygon(raw, W, H)
        if pts is None:
            rec.dropped.append({"index": k, "label": label, "reason": ";".join(actions)})
            continue
        rec.instances.append(Instance(label=label, points=pts, raw_points=raw, actions=actions, area_raw=a_raw, area=a))
    if not rec.instances:
        rec.notes.append("ornek_yok")
    return rec


def load_split(root: Path, split: str) -> list[Record]:
    return [load_record(p, split) for p in sorted((Path(root) / split).glob("*.json"))]


def load_all(root: Path = C.RAW) -> dict[str, list[Record]]:
    return {s: load_split(root, s) for s in C.SPLITS}


# ----------------------------------------------------------------------------- istatistik

def split_stats(records: list[Record]) -> dict:
    inst_per_class = Counter(i.label for r in records for i in r.instances)
    img_label = Counter(r.image_label for r in records if r.image_label)
    n_inst = [len(r.instances) for r in records]
    area_ratio = [i.area / (r.width * r.height) for r in records for i in r.instances]
    return {
        "n_images": len(records),
        "n_instances": int(sum(n_inst)),
        "instances_per_class": {c: int(inst_per_class.get(c, 0)) for c in C.CLASSES},
        "images_per_label": {c: int(img_label.get(c, 0)) for c in C.CLASSES},
        "instances_per_image": {
            "min": int(min(n_inst)), "mean": float(np.mean(n_inst)), "median": float(np.median(n_inst)), "max": int(max(n_inst)),
            "hist": {str(k): int(v) for k, v in sorted(Counter(n_inst).items())},
        },
        "instance_area_ratio": {
            "p05": float(np.percentile(area_ratio, 5)), "median": float(np.median(area_ratio)),
            "p95": float(np.percentile(area_ratio, 95)),
            "small_lt_1pct": int(sum(a < 0.01 for a in area_ratio)),
        },
        "image_sizes": {f"{w}x{h}": int(k) for (w, h), k in Counter((r.width, r.height) for r in records).items()},
        "n_multiclass_images": int(sum(len(r.classes) > 1 for r in records)),
    }
