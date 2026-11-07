"""Panelin ortak yardımcıları: veri yükleme (önbellekli), renkler, poligon çizimi, "Basitçe" kutusu."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config as C  # noqa: E402
from src.dataset import load_record  # noqa: E402

# Doğrulanmış kategorik palet (dataviz referans paleti, sabit sıra). 7 sınıf → ilk 7 slot.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
CLASS_COLOR = {c: SERIES[i] for i, c in enumerate(C.CLASSES)}
SPLIT_COLOR = {"train": SERIES[0], "val": SERIES[1], "test": SERIES[2]}
SPLIT_TR = {"train": "Eğitim (train)", "val": "Doğrulama (val)", "test": "Test"}
STATUS_ICON = {"tamam": "✅", "devam": "⏳", "bekliyor": "⬜"}


def basitce(text: str) -> None:
    """Teknik olmayan açıklama kutusu."""
    st.info(text, icon="💡")


def data_ready() -> bool:
    return (C.DERIVED / "summary.json").exists()


def require_data() -> None:
    if not data_ready():
        st.warning("Veri hazırlama çıktıları bulunamadı. Önce şunu çalıştırın:\n\n"
                   "`.venv/bin/python scripts/prepare_data.py`")
        st.stop()


# Önbellek anahtarına dosyanın değişme zamanı (mtime) eklenir: dosya yeniden üretilince panel
# eski değeri değil yenisini gösterir (sunucuyu yeniden başlatmaya gerek kalmaz, sayfayı yenilemek yeter).
def _mtime(p: Path) -> float:
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_data(show_spinner=False)
def _json(path: str, mtime: float) -> dict:
    return json.loads(Path(path).read_text())


@st.cache_data(show_spinner=False)
def _csv(path: str, mtime: float) -> pd.DataFrame:
    p = Path(path)
    return pd.read_csv(p) if p.exists() and p.stat().st_size > 1 else pd.DataFrame()


@st.cache_data(show_spinner=False)
def _yaml(path: str, mtime: float) -> dict:
    return yaml.safe_load(Path(path).read_text())


def summary() -> dict:
    p = C.DERIVED / "summary.json"
    return _json(str(p), _mtime(p))


def csv(name: str) -> pd.DataFrame:
    p = C.DERIVED / name
    return _csv(str(p), _mtime(p))


def doc_csv(name: str) -> pd.DataFrame:
    p = ROOT / "docs" / name
    return _csv(str(p), _mtime(p))


def durum() -> dict:
    p = ROOT / "docs" / "durum.yaml"
    return _yaml(str(p), _mtime(p))


@st.cache_data(show_spinner=False)
def stems(split: str) -> list[str]:
    return sorted(p.stem for p in (C.RAW / split).glob("*.json"))


@st.cache_data(show_spinner=False, max_entries=512)
def overlay(split: str, stem: str, show_raw: bool = False, size: int = 300) -> Image.Image:
    """Görüntünün üstüne sınıf rengiyle yarı saydam poligonlar çizer."""
    rec = load_record(C.RAW / split / f"{stem}.json", split)
    img = Image.open(rec.image_path).convert("RGBA")
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for inst in rec.instances:
        pts = inst.raw_points if show_raw else inst.points
        xy = [tuple(p) for p in pts]
        if len(xy) < 3:
            continue
        rgb = tuple(int(CLASS_COLOR[inst.label][i:i + 2], 16) for i in (1, 3, 5))
        d.polygon(xy, fill=rgb + (70,), outline=rgb + (255,), width=3)
    out = Image.alpha_composite(img, layer).convert("RGB")
    return out.resize((size, size))


def plain(split: str, stem: str, size: int = 300) -> Image.Image:
    return Image.open(C.RAW / split / f"{stem}.jpg").convert("RGB").resize((size, size))


def class_legend() -> None:
    chips = " ".join(
        f"<span style='display:inline-flex;align-items:center;gap:6px;margin:0 14px 6px 0'>"
        f"<span style='width:12px;height:12px;border-radius:3px;background:{CLASS_COLOR[c]};display:inline-block'></span>"
        f"{c} <span style='opacity:.65'>({C.CLASS_TR[c]})</span></span>"
        for c in C.CLASSES)
    st.markdown(f"<div style='font-size:0.85rem'>{chips}</div>", unsafe_allow_html=True)
