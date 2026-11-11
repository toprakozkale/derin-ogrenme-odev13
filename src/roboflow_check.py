"""Roboflow çapraz kontrolü (issue #2): Roboflow'un ham LabelMe'den ürettiği COCO ↔ bizim dönüşümümüz.

Roboflow'a ham Kaggle verisi yüklendi (bizim düzeltmelerimiz uygulanmadan), ön işleme/çoğaltma kapalı sürüm
COCO Segmentation olarak dışa aktarıldı → data/roboflow_export/coco/{train,valid,test}/_annotations.coco.json

Karşılaştırma:
  - Görüntü eşleştirme: Roboflow `extra.name` (orijinal dosya adı) ↔ bizim kayıtlarımız.
  - Roboflow'un yüklemede atladığı görüntüler: hangileri, ve projede zaten bulunan bir kopyası var mı (MD5 / pHash).
  - Görüntü başına örnek sayısı ve sınıf çok-kümesi.
  - Örnek bazında maske IoU'su: aynı sınıftaki örnekler arasında en iyi eşleşme (açgözlü, IoU'ya göre).
"""
from __future__ import annotations

import contextlib
import io
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .convert import mask_iou, polygon_to_mask
from .dataset import load_all

EXPORT = C.DATA / "roboflow_export" / "coco"
RF_SPLIT = {"train": "train", "val": "valid", "test": "test"}


def _rf_records(split: str) -> dict[str, dict]:
    """Roboflow COCO'sunu orijinal dosya köküne göre sözlüğe çevirir: stem → {w, h, anns:[(label, mask)]}"""
    from pycocotools.coco import COCO
    p = EXPORT / RF_SPLIT[split] / "_annotations.coco.json"
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(p))
    names = {c["id"]: c["name"] for c in coco.dataset["categories"]}
    out = {}
    for im in coco.dataset["images"]:
        orig = (im.get("extra") or {}).get("name") or im["file_name"].split("_jpg.rf.")[0] + ".jpg"
        anns = coco.loadAnns(coco.getAnnIds(imgIds=[im["id"]]))
        out[Path(orig).stem] = {"w": im["width"], "h": im["height"],
                                "anns": [(names[a["category_id"]], coco.annToMask(a).astype(bool)) for a in anns]}
    return out


def _match(ours: list[tuple[str, np.ndarray]], theirs: list[tuple[str, np.ndarray]]) -> list[dict]:
    """Aynı sınıftaki örnekleri IoU'ya göre açgözlü eşleştirir."""
    rows, used = [], set()
    pairs = sorted(((mask_iou(a[1], b[1]), i, j) for i, a in enumerate(ours) for j, b in enumerate(theirs)
                    if a[0] == b[0]), reverse=True)
    matched_i = set()
    for iou, i, j in pairs:
        if i in matched_i or j in used:
            continue
        matched_i.add(i); used.add(j)
        rows.append({"index": i, "label": ours[i][0], "iou": iou})
    for i, (lab, _) in enumerate(ours):
        if i not in matched_i:
            rows.append({"index": i, "label": lab, "iou": 0.0})
    return rows


def compare(dup_pairs: pd.DataFrame | None = None, hashes: pd.DataFrame | None = None) -> dict:
    recs = load_all(C.RAW)
    dup_pairs = dup_pairs if dup_pairs is not None else pd.read_csv(C.DERIVED / "dup_pairs.csv")
    hashes = hashes if hashes is not None else pd.read_csv(C.DERIVED / "image_hashes.csv")
    md5_of = {(r.split, r.stem): r.md5 for r in hashes.itertuples()}

    inst_rows, img_rows, missing = [], [], []
    for split, rs in recs.items():
        rf = _rf_records(split)
        for r in rs:
            if r.stem not in rf:
                # Neden atlandı? Projede birebir (MD5) ya da algısal olarak aynı (pHash=0) bir eşi var mı?
                same_md5 = [f"{s}/{st}" for (s, st), m in md5_of.items() if m == md5_of[(split, r.stem)] and (s, st) != (split, r.stem)]
                p = dup_pairs[(dup_pairs["dist"] == 0) & (
                    ((dup_pairs.a_split == split) & (dup_pairs.a_stem == r.stem)) |
                    ((dup_pairs.b_split == split) & (dup_pairs.b_stem == r.stem)))]
                twins = sorted({f"{x.b_split}/{x.b_stem}" if (x.a_split, x.a_stem) == (split, r.stem) else f"{x.a_split}/{x.a_stem}"
                                for x in p.itertuples()})
                missing.append({"split": split, "stem": r.stem, "md5_esi": "; ".join(same_md5),
                                "phash0_esi": "; ".join(twins), "ornek_sayisi": len(r.instances)})
                continue
            t = rf[r.stem]
            ours = [(i.label, polygon_to_mask(i.points, r.height, r.width)) for i in r.instances]
            m = _match(ours, t["anns"])
            for row in m:
                row.update(split=split, stem=r.stem, fixed=bool(r.instances[row["index"]].actions))
            inst_rows += m
            img_rows.append({"split": split, "stem": r.stem, "boyut_ayni": (t["w"], t["h"]) == (r.width, r.height),
                             "n_bizim": len(ours), "n_roboflow": len(t["anns"]),
                             "siniflar_ayni": Counter(l for l, _ in ours) == Counter(l for l, _ in t["anns"])})
    inst = pd.DataFrame(inst_rows); img = pd.DataFrame(img_rows); miss = pd.DataFrame(missing)
    inst.to_csv(C.DERIVED / "roboflow_instances.csv", index=False)
    img.to_csv(C.DERIVED / "roboflow_images.csv", index=False)
    miss.to_csv(C.DERIVED / "roboflow_missing.csv", index=False)

    rf_counts = {s: len(_rf_records(s)) for s in C.SPLITS}
    summary = {
        "roboflow_goruntu": rf_counts, "roboflow_toplam": sum(rf_counts.values()),
        "atlanan_goruntu": len(miss),
        "atlanan_bolmeye_gore": miss["split"].value_counts().to_dict() if len(miss) else {},
        "atlananlarin_esi_var": int(((miss["md5_esi"] != "") | (miss["phash0_esi"] != "")).sum()) if len(miss) else 0,
        "eslesen_goruntu": len(img),
        "boyut_ayni": int(img["boyut_ayni"].sum()), "ornek_sayisi_ayni": int((img.n_bizim == img.n_roboflow).sum()),
        "siniflar_ayni": int(img["siniflar_ayni"].sum()),
        "ornek_karsilastirilan": len(inst),
        "iou_ortalama": float(inst["iou"].mean()), "iou_min": float(inst["iou"].min()),
        "iou_p01": float(inst["iou"].quantile(0.01)),
        "iou_lt_0_99": int((inst["iou"] < 0.99).sum()), "iou_lt_0_95": int((inst["iou"] < 0.95).sum()),
        "duzeltilen_ornek_iou_min": float(inst.loc[inst.fixed, "iou"].min()) if inst["fixed"].any() else None,
        "duzeltilmemis_iou_min": float(inst.loc[~inst.fixed, "iou"].min()),
    }
    (C.DERIVED / "roboflow_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    # Roboflow dışa aktarımı git'e girmez; sonucun kalıcı kopyası docs/roboflow/ altında saklanır (panel ve rapor buradan okur)
    keep = C.ROOT / "docs" / "roboflow"
    keep.mkdir(parents=True, exist_ok=True)
    (keep / "roboflow_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    miss.to_csv(keep / "roboflow_missing.csv", index=False)
    inst.sort_values("iou").head(20).to_csv(keep / "roboflow_en_dusuk_iou.csv", index=False)
    return summary
