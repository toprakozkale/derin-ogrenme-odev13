"""Ödev 13 paneli — giriş noktası.

Çalıştırma:  .venv/bin/streamlit run app/Ana_Sayfa.py
"""
import streamlit as st

st.set_page_config(page_title="Ödev 13 · Çilek Hastalığı", page_icon="🍓", layout="wide")

pages = {
    "Proje": [
        st.Page("pages/genel.py", title="Genel Bakış", icon="🏠", default=True),
    ],
    "Aşama 1 · Veri": [
        st.Page("pages/veri.py", title="Veri Seti", icon="🖼️"),
        st.Page("pages/etiket.py", title="Etiket Denetimi ve Dönüşüm", icon="🧹"),
        st.Page("pages/sizinti.py", title="Sızıntı Analizi", icon="🔍"),
    ],
    "Aşama 2–3 · Modeller": [
        st.Page("pages/siniflandirma.py", title="Sınıflandırma", icon="🏷️"),
        st.Page("pages/segmentasyon.py", title="Segmentasyon", icon="✂️"),
    ],
    "Yardım": [
        st.Page("pages/sozluk.py", title="Sözlük", icon="📖"),
    ],
}
st.navigation(pages).run()
