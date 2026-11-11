import subprocess

import pandas as pd
import streamlit as st

from common import ROOT, basitce

st.title("📒 İlerleme Günlüğü")
basitce(
    "Ödev boyunca ne yaptığımızın tarih sırasıyla kaydı: hangi işler bitti, hangi kararlar alındı, nerede sorun çıktı ve nasıl "
    "çözüldü. Hocaya sunumda 'nasıl çalıştınız?' sorusunun cevabı burada. Yeni kayıtlar en üstte."
)

text = (ROOT / "docs" / "gunluk.md").read_text()
# Başlığı ve giriş paragrafını panel başlığıyla tekrarlamamak için ilk "---" sonrasından göster
body = text.split("\n---\n", 1)[1] if "\n---\n" in text else text
st.markdown(body)

st.divider()
st.subheader("GitHub'a gönderilen değişiklikler")
try:
    log = subprocess.run(["git", "-C", str(ROOT), "log", "--date=format:%d.%m.%Y %H:%M", "--format=%h\t%ad\t%s"],
                         capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
    df = pd.DataFrame([l.split("\t", 2) for l in log], columns=["kod", "tarih", "açıklama"])
    st.dataframe(df, hide_index=True, width="stretch")
    st.caption("Her satır bir commit. Kod sütunu, GitHub'da o değişikliği bulmak için kullanılır.")
except Exception as e:  # git yoksa sayfa yine açılsın
    st.info(f"Git geçmişi okunamadı: {e}")
