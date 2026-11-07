"""Yinelenen ve yakın-yinelenen görüntü tespiti (bölmeler arası sızıntı kontrolü).

- Birebir aynı dosya: MD5.
- Yakın-yinelenen (aynı sahnenin kaydırılmış/kırpılmış/yeniden sıkıştırılmış hali): 64 bit algısal hash (pHash),
  Hamming mesafesi. Eşik panelde görsel kontrolle seçilir (config.PHASH_THRESHOLD).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

from . import config as C


def image_hashes(records_by_split) -> pd.DataFrame:
    rows = []
    for split, recs in records_by_split.items():
        for r in recs:
            ph = imagehash.phash(Image.open(r.image_path))
            rows.append({
                "split": split, "stem": r.stem,
                "md5": hashlib.md5(r.image_path.read_bytes()).hexdigest(),
                "phash": int(str(ph), 16),
            })
    return pd.DataFrame(rows)


def near_duplicate_pairs(h: pd.DataFrame, max_dist: int = C.PHASH_MAX_STORE) -> pd.DataFrame:
    """Hamming mesafesi <= max_dist olan tüm görüntü çiftleri (aynı bölme içi ve bölmeler arası)."""
    v = h["phash"].to_numpy(dtype=np.uint64)
    rows = []
    for i in range(len(v) - 1):
        d = np.bitwise_count(v[i] ^ v[i + 1:]).astype(int)
        for k in np.nonzero(d <= max_dist)[0]:
            j = i + 1 + k
            rows.append((i, j, int(d[k])))
    p = pd.DataFrame(rows, columns=["i", "j", "dist"])
    a, b = h.iloc[p["i"]].reset_index(drop=True), h.iloc[p["j"]].reset_index(drop=True)
    out = pd.DataFrame({
        "a_split": a["split"], "a_stem": a["stem"], "b_split": b["split"], "b_stem": b["stem"],
        "dist": p["dist"], "same_md5": a["md5"].values == b["md5"].values,
    })
    out["cross_split"] = out["a_split"] != out["b_split"]
    return out.sort_values(["dist", "a_split", "a_stem"]).reset_index(drop=True)


def leaked(pairs: pd.DataFrame, threshold: int, target: str, sources: tuple[str, ...]) -> set[str]:
    """`target` bölmesinde olup `sources` bölmelerinden birinde mesafe <= threshold benzeri bulunan görüntüler."""
    p = pairs[(pairs["dist"] <= threshold) & pairs["cross_split"]]
    hit = set()
    hit |= set(p.loc[(p["a_split"] == target) & p["b_split"].isin(sources), "a_stem"])
    hit |= set(p.loc[(p["b_split"] == target) & p["a_split"].isin(sources), "b_stem"])
    return hit


def leakage_table(pairs: pd.DataFrame, n_test: int, n_val: int, thresholds=range(0, C.PHASH_MAX_STORE + 1)) -> pd.DataFrame:
    rows = []
    for t in thresholds:
        lt = leaked(pairs, t, "test", ("train", "val"))
        lv = leaked(pairs, t, "val", ("train",))
        rows.append({"esik": t, "test_sizan": len(lt), "test_oran": len(lt) / n_test,
                     "val_sizan": len(lv), "val_oran": len(lv) / n_val})
    return pd.DataFrame(rows)
