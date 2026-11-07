"""Aşama 1 veri hazırlamayı komut satırından baştan sona çalıştırır.

    .venv/bin/python scripts/prepare_data.py            # varsayılan eşikle
    .venv/bin/python scripts/prepare_data.py --esik 8   # pHash eşiğini değiştir
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C  # noqa: E402
from src.pipeline import download, run  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--esik", type=int, default=C.PHASH_THRESHOLD, help="pHash yakın-yinelenen eşiği")
    ap.add_argument("--donusum-yok", action="store_true", help="format dönüşümlerini atla (yalnızca analiz)")
    a = ap.parse_args()
    download()
    s = run(threshold=a.esik, write_formats=not a.donusum_yok)
    print(json.dumps({k: s[k] for k in ("duplicates", "roundtrip", "label_issue_counts")}, indent=2, ensure_ascii=False, default=str))
