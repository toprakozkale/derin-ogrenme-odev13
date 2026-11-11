"""Aşama 1 veri hazırlama hattı: indir → denetle/düzelt → istatistik → yinelenen/sızıntı → dönüştür → doğrula.

Notebook (01_veri_hazirlama) ve `scripts/prepare_data.py` bu fonksiyonları çağırır; panel (app/) yalnızca
`data/derived/` altındaki çıktıları okur. Böylece rapor, notebook ve panel aynı sayıları gösterir.
"""
from __future__ import annotations

import hashlib
import json
import time
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd

from . import config as C
from .convert import verify_roundtrip, write_cls, write_coco, write_yolo_seg
from .dataset import load_all, split_stats
from .dedup import image_hashes, leaked, leakage_table, near_duplicate_pairs


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def download(dest_dir: Path = C.RAW.parent) -> Path:
    """Kaggle'dan zip'i indirir (kagglehub ya da Kaggle API), SHA-256 doğrular, açar. Zaten açıksa atlar."""
    if (C.RAW / "train").exists():
        return C.RAW
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / "strawberry-disease-detection-dataset.zip"
    if not zip_path.exists():
        # kagglehub (Colab'da hazır) veriyi kendi önbelleğine indirip açar; içinden train/ içeren klasörü bulup bağlarız
        import kagglehub
        src = Path(kagglehub.dataset_download(C.KAGGLE_REF))
        hits = [p.parent for p in src.rglob("train") if p.is_dir() and (p.parent / "test").is_dir()]
        if not hits:
            raise FileNotFoundError(f"İndirilen klasörde train/ bulunamadı: {src}")
        C.RAW.parent.mkdir(parents=True, exist_ok=True)
        if not C.RAW.exists():
            C.RAW.symlink_to(hits[0], target_is_directory=True)
        return C.RAW
    sha = _sha256(zip_path)
    if sha != C.KAGGLE_ZIP_SHA256:
        print(f"UYARI: zip SHA-256 farklı ({sha}); veri seti sürümü değişmiş olabilir.")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(C.RAW)
    return C.RAW


def severity_check(raw: Path = C.RAW) -> dict:
    sev = sorted((raw / "Test Disease Severity Level").rglob("*.jpg"))
    same = sum(
        (raw / "test" / p.name).exists() and (raw / "test" / p.name).read_bytes() == p.read_bytes()
        and (raw / "test" / (p.stem + ".json")).read_bytes() == p.with_suffix(".json").read_bytes()
        for p in sev
    )
    levels = Counter(p.parent.name for p in sev)
    return {"n_images": len(sev), "identical_to_test": int(same), "levels": dict(levels),
            "karar": "test/ kümesinin birebir kopyası; ek veri olarak kullanılmaz" if same == len(sev) else "İNCELE"}


def run(threshold: int = C.PHASH_THRESHOLD, write_formats: bool = True, verbose: bool = True) -> dict:
    t0 = time.time()
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    C.ensure_dirs()
    D = C.DERIVED
    (D / "splits").mkdir(parents=True, exist_ok=True)

    log("1) Etiketler okunuyor ve denetleniyor…")
    recs = load_all(C.RAW)

    # Düzeltme / atma kayıtları
    issue_rows = []
    for split, rs in recs.items():
        for r in rs:
            for k, inst in enumerate(r.instances):
                for a in inst.actions:
                    issue_rows.append({"split": split, "stem": r.stem, "label": inst.label, "islem": a,
                                       "alan_ham": round(inst.area_raw, 1), "alan_son": round(inst.area, 1)})
            for d in r.dropped:
                issue_rows.append({"split": split, "stem": r.stem, "label": d["label"], "islem": d["reason"],
                                   "alan_ham": None, "alan_son": None})
            for n in r.notes:
                issue_rows.append({"split": split, "stem": r.stem, "label": None, "islem": n, "alan_ham": None, "alan_son": None})
    pd.DataFrame(issue_rows, columns=["split", "stem", "label", "islem", "alan_ham", "alan_son"]).to_csv(D / "label_issues.csv", index=False)

    # Çok sınıflı görüntüler ve dosya adı ↔ etiket uyuşmazlıkları
    multi = []
    for split, rs in recs.items():
        for r in rs:
            prefix = r.stem.rstrip("0123456789")
            slugs = {C.CLASS_SLUG[c] for c in r.classes}
            if len(r.classes) > 1 or prefix not in slugs:
                multi.append({"split": split, "stem": r.stem, "dosya_oneki": prefix,
                              "siniflar": "; ".join(sorted(r.classes)), "goruntu_etiketi": r.image_label,
                              "tur": "cok_sinifli" if len(r.classes) > 1 else "ad_etiket_uyusmazligi"})
    pd.DataFrame(multi).to_csv(D / "multiclass_images.csv", index=False)

    log("2) İstatistikler…")
    stats = {s: split_stats(rs) for s, rs in recs.items()}

    log("3) Yinelenen / yakın-yinelenen görüntüler (pHash)…")
    h = image_hashes(recs)
    h.to_csv(D / "image_hashes.csv", index=False)
    pairs = near_duplicate_pairs(h)
    pairs.to_csv(D / "dup_pairs.csv", index=False)
    n_test, n_val = len(recs["test"]), len(recs["val"])
    leakage_table(pairs, n_test, n_val).to_csv(D / "leakage_table.csv", index=False)
    md5_groups = h[h.duplicated("md5", keep=False)].groupby("md5")
    exact = [{"md5": m, "uyeler": "; ".join(f"{a}/{b}" for a, b in zip(g["split"], g["stem"])),
              "bolmeler_arasi": g["split"].nunique() > 1} for m, g in md5_groups]
    pd.DataFrame(exact).to_csv(D / "exact_duplicates.csv", index=False)

    leaked_test = leaked(pairs, threshold, "test", ("train", "val"))
    leaked_val = leaked(pairs, threshold, "val", ("train",))

    log("4) Bölme listeleri…")
    split_sha = {}
    for s, rs in recs.items():
        p = D / "splits" / f"{s}.txt"
        p.write_text("\n".join(r.image_path.name for r in rs) + "\n")
        split_sha[s] = _sha256(p)
    clean = [r.image_path.name for r in recs["test"] if r.stem not in leaked_test]
    p = D / "splits" / "test_clean.txt"
    p.write_text("\n".join(clean) + "\n")
    split_sha["test_clean"] = _sha256(p)

    roundtrip_summary = None
    if write_formats:
        log("5) Dönüşümler: sınıflandırma klasörleri, YOLO-seg, COCO…")
        cls_counts = write_cls(recs, C.PROCESSED / "cls")
        yaml_path = write_yolo_seg(recs, C.PROCESSED / "yoloseg")
        coco_paths = write_coco(recs, C.PROCESSED / "coco")

        log("6) Doğrulama: her formattan maske geri üretilip kaynakla karşılaştırılıyor…")
        rt = pd.DataFrame(verify_roundtrip(recs, C.PROCESSED / "yoloseg", coco_paths))
        rt.to_csv(D / "roundtrip.csv", index=False)
        roundtrip_summary = {
            "n_instances": int(len(rt)),
            "iou_yolo_min": float(rt["iou_yolo"].min()), "iou_yolo_mean": float(rt["iou_yolo"].mean()),
            "iou_coco_min": float(rt["iou_coco"].min()), "iou_coco_mean": float(rt["iou_coco"].mean()),
            "n_iou_lt_0_99": int(((rt["iou_yolo"] < 0.99) | (rt["iou_coco"] < 0.99)).sum()),
            "class_ok": bool(rt["class_ok"].all()), "count_ok": bool(rt["count_ok"].all()),
            "fixed_instances": int(rt["fixed"].sum()),
            "fixed_iou_vs_raw_min": float(rt.loc[rt["fixed"], "iou_fixed_vs_raw"].min()) if rt["fixed"].any() else None,
        }
        # pycocotools ile yüklenebilirlik
        from pycocotools.coco import COCO
        import contextlib, io
        coco_load = {}
        for s, cp in coco_paths.items():
            with contextlib.redirect_stdout(io.StringIO()):
                cc = COCO(str(cp))
            coco_load[s] = {"images": len(cc.getImgIds()), "annotations": len(cc.getAnnIds()), "categories": len(cc.getCatIds())}
        roundtrip_summary["coco_load"] = coco_load
        roundtrip_summary["cls_counts"] = cls_counts
        # Colab'da veri proje klasörünün dışında (ODEV13_DATA) olabilir; o durumda tam yol yazılır
        roundtrip_summary["yolo_yaml"] = (str(yaml_path.relative_to(C.ROOT))
                                          if yaml_path.is_relative_to(C.ROOT) else str(yaml_path))

    summary = {
        "olusturma_zamani": time.strftime("%Y-%m-%d %H:%M:%S"),
        "kaggle_ref": C.KAGGLE_REF,
        "zip_sha256_beklenen": C.KAGGLE_ZIP_SHA256,
        "classes": list(C.CLASSES),
        "image_label_rule": C.IMAGE_LABEL_RULE,
        "stats": stats,
        "severity_check": severity_check(C.RAW),
        "label_issue_counts": dict(Counter(":".join(r["islem"].split(":")[:2]) for r in issue_rows)),
        "n_multiclass_or_mismatch": len(multi),
        "duplicates": {
            "exact_groups": len(exact), "exact_cross_split_groups": int(sum(e["bolmeler_arasi"] for e in exact)),
            "phash_threshold": threshold,
            "test_leaked": len(leaked_test), "test_clean": len(clean), "val_leaked": len(leaked_val),
        },
        "split_list_sha256": split_sha,
        "roundtrip": roundtrip_summary,
        "sure_sn": round(time.time() - t0, 1),
    }
    (D / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    log(f"Bitti ({summary['sure_sn']} sn). Özet: {D / 'summary.json'}")
    return summary
