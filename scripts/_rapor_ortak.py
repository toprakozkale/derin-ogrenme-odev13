"""Raporların (1. rapor CLS, 2. rapor SEG) ortak yapı taşları: Türkçe sayı biçimleri, iki sütunlu IEEE düzeni,
başlık/paragraf/tablo/şekil yardımcıları ve yazar kimliği. Rapor betikleri `from _rapor_ortak import *` ile kullanır;
içe aktarıldığında boş bir belge (`doc`) hazırlanır — her betik tek bir rapor üretir."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_COLOR_INDEX
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config as C  # noqa: E402
from src import metrics as M  # noqa: E402

# ---------------------------------------------------------------- kimlik (kullanıcı verdi, 21 Kas 2026)
YAZAR = "Toprak Özkale"
OGRENCI_NO = "250401006"
KURUM = "Iğdır Üniversitesi, Lisansüstü Eğitim Enstitüsü"
DERS = "Derin Öğrenme-Sinir Ağları"
HOCA = "Gültekin Işık"  # unvan bilinmiyor → yazılmadı

# ---------------------------------------------------------------- sayı biçimleri (Türkçe: ondalık virgül, binlik nokta)
def f(x: float, d: int = 3) -> str:
    return f"{x:.{d}f}".replace(".", ",")


def pct(x: float, d: int = 1) -> str:
    return "%" + f(100 * x, d)


def n(x: int) -> str:
    return f"{int(x):,}".replace(",", ".")


def ga(s: str) -> str:
    """'0.931 [0.913, 0.949]' → '0,931 [0,913; 0,949]'"""
    return s.replace(', ', '; ').replace('.', ',')


# Türkçe ek: sayının okunuşundaki son sözcüğe göre ünlü uyumu ve sertleşme (0,867'den · %46'sı · 51'i · 0,99995'tir)
_BIRLER = ["sıfır", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz"]
_ONLAR = ["", "on", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş", "seksen", "doksan"]


def _son_sozcuk(sayi: str) -> str:
    g = sayi.replace("%", "").replace(".", "")
    g = g.split(",")[-1] if "," in g else g
    if int(g) == 0:
        return "sıfır"
    r = g[::-1]
    i = next(k for k, ch in enumerate(r) if ch != "0")
    return _BIRLER[int(r[i])] if i == 0 else _ONLAR[int(r[i])] if i == 1 else "yüz" if i == 2 else "bin"


def ek(sayi: str, tur: str) -> str:
    """tur: 'iyelik' (-sı), 'belirtme' (-ı), 'iyelik_belirtme' (-sını), 'den', 'e', 'dir'"""
    w = _son_sozcuk(sayi)
    v = [c for c in w if c in "aeıioöuü"][-1]
    h2 = "a" if v in "aıou" else "e"
    h4 = {"a": "ı", "ı": "ı", "o": "u", "u": "u", "e": "i", "i": "i", "ö": "ü", "ü": "ü"}[v]
    unlu = w[-1] in "aeıioöuü"
    d = "t" if w[-1] in "fstkçşhp" else "d"
    e_ = {"iyelik": ("s" if unlu else "") + h4, "belirtme": ("y" if unlu else "") + h4,
          "den": d + h2 + "n", "e": ("y" if unlu else "") + h2, "dir": d + h4 + "r",
          "iyelik_belirtme": ("s" if unlu else "") + h4 + "n" + h4}[tur]
    return f"{sayi}'{e_}"


def ms(mean: float, std: float, d: int = 3) -> str:
    return f"{f(mean, d)} ± {f(std, d)}"


# ---------------------------------------------------------------- belge kurulumu
doc = Document()
GOVDE_PT = 10
FONT = "Times New Roman"

stl = doc.styles["Normal"]
stl.font.name = FONT
stl.font.size = Pt(GOVDE_PT)
rpr = stl.element.get_or_add_rPr()
rfonts = rpr.find(qn("w:rFonts"))
for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
    rfonts.set(qn(a), FONT)
lang = OxmlElement("w:lang")
lang.set(qn("w:val"), "tr-TR")
lang.set(qn("w:eastAsia"), "tr-TR")
rpr.append(lang)
stl.paragraph_format.space_after = Pt(0)
stl.paragraph_format.space_before = Pt(0)
stl.paragraph_format.line_spacing = 1.0

# Word'de otomatik heceleme (iki yana yaslı dar sütunlarda boşlukları azaltır)
settings = doc.settings.element
ah = OxmlElement("w:autoHyphenation")
settings.append(ah)

sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
sec.top_margin = sec.bottom_margin = Cm(1.9)
sec.left_margin = sec.right_margin = Cm(1.6)
METIN_GENISLIK = 21.0 - 2 * 1.6
SUTUN_ARA = 0.5
SUTUN = (METIN_GENISLIK - SUTUN_ARA) / 2


def sutun_ayarla(section, sayi: int):
    sp = section._sectPr
    for c in sp.findall(qn("w:cols")):
        sp.remove(c)
    cols = OxmlElement("w:cols")
    cols.set(qn("w:num"), str(sayi))
    cols.set(qn("w:space"), str(int(SUTUN_ARA / 2.54 * 1440)))
    sp.append(cols)


def yeni_bolum(sayi: int):
    s = doc.add_section(WD_SECTION.CONTINUOUS)
    sutun_ayarla(s, sayi)
    return s


def sayfa_no(section):
    p = section.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run()
    for tag, txt in (("begin", None), (None, "PAGE"), ("end", None)):
        if tag:
            e = OxmlElement("w:fldChar")
            e.set(qn("w:fldCharType"), tag)
        else:
            e = OxmlElement("w:instrText")
            e.set(qn("xml:space"), "preserve")
            e.text = txt
        r._r.append(e)
    r.font.size = Pt(8)


sayfa_no(sec)


def run(p, text, bold=False, italic=False, size=None, smallcaps=False, highlight=False, sup=False):
    r = p.add_run(text)
    r.bold, r.italic = bold, italic
    if size:
        r.font.size = Pt(size)
    if smallcaps:
        r.font.small_caps = True
    if highlight:
        r.font.highlight_color = WD_COLOR_INDEX.YELLOW
    if sup:
        r.font.superscript = True
    return r


def para(parcalar, align=WD_ALIGN_PARAGRAPH.JUSTIFY, girinti=True, size=None, after=0, before=0, keep=False):
    """parcalar: str ya da (metin, {bold/italic/...}) listesi. **kalın** kısaltması yok — açık yazılır."""
    p = doc.add_paragraph()
    p.alignment = align
    pf = p.paragraph_format
    if girinti:
        pf.first_line_indent = Cm(0.35)
    pf.space_after = Pt(after)
    pf.space_before = Pt(before)
    pf.keep_with_next = keep
    if isinstance(parcalar, str):
        parcalar = [parcalar]
    for prc in parcalar:
        if isinstance(prc, str):
            run(p, prc, size=size)
        else:
            run(p, prc[0], size=size, **prc[1])
    return p


ROMA = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
_h1 = [0]
_h2 = [0]


def h1(text, numarasiz=False):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    if numarasiz:
        run(p, text, smallcaps=True)
    else:
        _h1[0] += 1
        _h2[0] = 0
        run(p, f"{ROMA[_h1[0] - 1]}. ", smallcaps=True)
        run(p, text, smallcaps=True)
    return p


def h2(text):
    _h2[0] += 1
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    run(p, f"{chr(64 + _h2[0])}. {text}", italic=True)
    return p


_tablo = [0]
_sekil = [0]


def kenarlik(el, **kenarlar):
    """kenarlar: top/bottom/insideH ... = 'single' | 'nil' (kalınlık 4 = 0,5 pt; 8 = 1 pt)"""
    tag = "w:tblBorders" if el.tag == qn("w:tblPr") else "w:tcBorders"
    b = el.find(qn(tag))
    if b is None:
        b = OxmlElement(tag)
        el.append(b)
    for k, v in kenarlar.items():
        e = OxmlElement(f"w:{k}")
        if v == "nil":
            e.set(qn("w:val"), "nil")
        else:
            e.set(qn("w:val"), "single")
            e.set(qn("w:sz"), str(v))
            e.set(qn("w:color"), "000000")
        b.append(e)


def hecesiz(p):
    p._p.get_or_add_pPr().append(OxmlElement("w:suppressAutoHyphens"))


def tablo(baslik, basliklar, satirlar, genislikler, hizalar=None, kalin_satirlar=(), yazi=8, toplam_genislik=None):
    """IEEE: başlık tablonun üstünde, ortalı, 'TABLO I' + küçük büyük harf başlık."""
    _tablo[0] += 1
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.keep_with_next = True
    run(p, f"TABLO {ROMA[_tablo[0] - 1]}", size=8)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.keep_with_next = True
    run(p, baslik, size=8, smallcaps=True)
    hecesiz(p)

    t = doc.add_table(rows=1 + len(satirlar), cols=len(basliklar))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    tblPr = t._tbl.tblPr
    lay = OxmlElement("w:tblLayout")
    lay.set(qn("w:type"), "fixed")
    tblPr.append(lay)
    kenarlik(tblPr, top=8, bottom=8, left="nil", right="nil", insideH="nil", insideV="nil")
    hizalar = hizalar or (["l"] + ["c"] * (len(basliklar) - 1))
    H = {"l": WD_ALIGN_PARAGRAPH.LEFT, "c": WD_ALIGN_PARAGRAPH.CENTER, "r": WD_ALIGN_PARAGRAPH.RIGHT}
    for i, row in enumerate([basliklar] + satirlar):
        tr = t.rows[i]
        trPr = tr._tr.get_or_add_trPr()
        cs = OxmlElement("w:cantSplit")
        trPr.append(cs)
        for j, val in enumerate(row):
            cell = tr.cells[j]
            cell.width = Cm(genislikler[j])
            cp = cell.paragraphs[0]
            cp.alignment = H[hizalar[j]]
            cp.paragraph_format.keep_with_next = i < len(satirlar)
            hecesiz(cp)
            r_ = cp.add_run(str(val))
            r_.font.size = Pt(yazi)
            r_.bold = (i == 0) or (i - 1 in kalin_satirlar)
            tcPr = cell._tc.get_or_add_tcPr()
            mar = OxmlElement("w:tcMar")
            for side in ("left", "right"):
                e = OxmlElement(f"w:{side}")
                e.set(qn("w:w"), "40")
                e.set(qn("w:type"), "dxa")
                mar.append(e)
            tcPr.append(mar)
            if i == 0:
                kenarlik(tcPr, bottom=4)
    # sütun genişlikleri tablo ızgarasında da
    grid = t._tbl.tblGrid
    for j, gc in enumerate(grid.findall(qn("w:gridCol"))):
        gc.set(qn("w:w"), str(int(genislikler[j] / 2.54 * 1440)))
    bos = doc.add_paragraph()
    bos.paragraph_format.space_after = Pt(4)
    return t


def sekil(yol, aciklama, genislik):
    _sekil[0] += 1
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(yol), width=Cm(genislik))
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(6)
    run(p, f"Şekil {_sekil[0]}. ", size=8)
    run(p, aciklama, size=8)
    return _sekil[0]


def yer_tutucu(text):
    return (text, {"highlight": True})




def sayaclar() -> tuple[int, int]:
    """(tablo sayısı, şekil sayısı)"""
    return _tablo[0], _sekil[0]
