"""Raporlar ve teslim: rapor PDF'lerinin sayfa önizlemesi, indirme düğmeleri, linkler ve teslim kontrol listesi.

Kaynak: docs/durum.yaml → "raporlar" (kontrol listesi, dosya yolları, Colab linki) ve rapor/*.pdf, *.docx.
"""
from datetime import datetime
from pathlib import Path

import streamlit as st

from common import ROOT, STATUS_ICON, _mtime, basitce, durum

REPO_URL = "https://github.com/toprakozkale/derin-ogrenme-odev13"

st.title("📝 Raporlar ve Teslim")
basitce(
    "Hocaya gidecek raporların son hâli burada. Raporu Word'de elle değil, bir betikle üretiyoruz: betik bütün sayıları "
    "doğrudan sonuç dosyalarından okur, böylece rapordaki hiçbir sayı elle yazılmaz. Aşağıda her raporun sayfalarını "
    "görebilir, PDF ve Word dosyasını indirebilir ve teslimden önce yapılacakları takip edebilirsiniz."
)


@st.cache_data(show_spinner=False)
def _sayfalar(path: str, mtime: float, olcek: float = 1.4) -> list:
    """PDF sayfalarını görüntüye çevirir (dosya değişince önbellek yenilenir)."""
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(path)
    return [doc[i].render(scale=olcek).to_pil() for i in range(len(doc))]


d = durum()
raporlar = d.get("raporlar", [])
if not raporlar:
    st.warning("docs/durum.yaml içinde 'raporlar' bölümü yok.")
    st.stop()

sec = st.segmented_control("Rapor", [r["ad"] for r in raporlar], default=raporlar[0]["ad"])
r = next((x for x in raporlar if x["ad"] == sec), raporlar[0])

pdf, docx = ROOT / r["pdf"], ROOT / r["docx"]
due = datetime.strptime(r["teslim"], "%Y-%m-%d %H:%M")
kalan = due - datetime.now()
kontrol = r.get("kontrol", [])
n_ok = sum(s == "tamam" for _, s in kontrol)

c = st.columns(4)
c[0].metric("Teslime kalan", f"{kalan.days} gün" if kalan.total_seconds() > 0 else "Süre doldu",
            help=f"Son teslim: {due:%d.%m.%Y %H:%M}")
if pdf.exists():
    sayfalar = _sayfalar(str(pdf), _mtime(pdf))
    c[1].metric("Sayfa", len(sayfalar), help="Ödev 4–6 sayfa istiyor")
    c[2].metric("Son güncelleme", f"{datetime.fromtimestamp(pdf.stat().st_mtime):%d.%m %H:%M}")
else:
    sayfalar = []
    c[1].metric("Sayfa", "—")
    c[2].metric("Son güncelleme", "—")
c[3].metric("Kontrol listesi", f"{n_ok}/{len(kontrol)}")

st.subheader("Teslimden önce")
st.progress(n_ok / len(kontrol) if kontrol else 0.0)
for madde, s in kontrol:
    st.markdown(f"{STATUS_ICON[s]} {madde}")

st.subheader("Dosyalar ve linkler")
b = st.columns(3)
if pdf.exists():
    b[0].download_button("📄 PDF indir", pdf.read_bytes(), file_name=pdf.name, mime="application/pdf")
if docx.exists():
    b[1].download_button("📝 Word (.docx) indir", docx.read_bytes(), file_name=docx.name,
                         mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
st.markdown(f"- Colab defteri: {r['colab']}\n- Kod deposu: {REPO_URL}")
st.caption("Depo teslim gününe kadar gizli (private); linkler hocada ancak depo herkese açık yapılınca çalışır. "
           + (f"Rapor `{r['uretici']}` ile yeniden üretilir." if r.get("uretici") else ""))

st.subheader("Sayfalar")
if not sayfalar:
    st.info("Bu raporun PDF'i henüz yok.", icon="⏳")
else:
    sutun = st.columns(2)
    for i, img in enumerate(sayfalar):
        with sutun[i % 2]:
            st.image(img, caption=f"Sayfa {i + 1}", width="stretch")
