"""Ödev 13 paneli — giriş noktası.

Çalıştırma:  .venv/bin/streamlit run app/Ana_Sayfa.py
"""
import streamlit as st

st.set_page_config(page_title="Ödev 13 · Çilek Hastalığı", page_icon="🍓", layout="wide")

pages = {
    "Proje": [
        st.Page("sayfalar/genel.py", title="Genel Bakış", icon="🏠", default=True),
        st.Page("sayfalar/gunluk.py", title="İlerleme Günlüğü", icon="📒"),
        st.Page("sayfalar/raporlar.py", title="Raporlar ve Teslim", icon="📝"),
        st.Page("sayfalar/sekiller.py", title="Rapor Şekilleri", icon="📊"),
    ],
    "Aşama 1 · Veri": [
        st.Page("sayfalar/veri.py", title="Veri Seti", icon="🖼️"),
        st.Page("sayfalar/etiket.py", title="Etiket Denetimi ve Dönüşüm", icon="🧹"),
        st.Page("sayfalar/sizinti.py", title="Sızıntı Analizi", icon="🔍"),
    ],
    "Aşama 2–3 · Modeller": [
        st.Page("sayfalar/siniflandirma.py", title="Sınıflandırma", icon="🏷️"),
        st.Page("sayfalar/segmentasyon.py", title="Segmentasyon", icon="✂️"),
    ],
    "Yardım": [
        st.Page("sayfalar/sozluk.py", title="Sözlük", icon="📖"),
    ],
}
st.navigation(pages).run()
