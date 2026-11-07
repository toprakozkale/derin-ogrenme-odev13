from datetime import datetime

import pandas as pd
import streamlit as st

from common import STATUS_ICON, basitce, data_ready, durum, summary

d = durum()
st.title("🍓 Çilek Hastalığı: Sınıflandırma ve Bölütleme")
st.caption("Derin Öğrenme YL ödevi · Görev 13 · Strawberry Disease Detection Dataset (Afzaal ve ark., 2021)")

basitce(
    "Bir sera işletmesi, telefonla çekilen çilek fotoğraflarından **hangi hastalığın olduğunu** (sınıflandırma) ve "
    "**hastalıklı bölgenin tam olarak nerede olduğunu** (bölütleme) otomatik bulmak istiyor. Biz bunun için yapay zekâ "
    "modellerini eğitip karşılaştırıyoruz. Bu panel, ödevin her adımında ne yaptığımızı ve sonuçları gösterir."
)

# --- teslim geri sayımı
cols = st.columns(len(d["teslimler"]))
now = datetime.now()
for col, t in zip(cols, d["teslimler"]):
    due = datetime.strptime(t["tarih"], "%Y-%m-%d %H:%M")
    left = due - now
    with col:
        st.metric(t["ad"], f"{left.days} gün" if left.total_seconds() > 0 else "Süre doldu",
                  help=f"Son teslim: {due:%d.%m.%Y %H:%M}")
        st.caption(f"Son teslim: {due:%d %B %Y, %H:%M}".replace("November", "Kasım").replace("December", "Aralık"))

st.divider()

# --- puanlama
st.subheader("Puanlama ve durum")
rows = [{"Durum": STATUS_ICON[p["durum"]], "Bileşen": p["bilesen"], "Ağırlık": f"%{p['agirlik']}", "Not": p["not"]}
        for p in d["puanlama"]]
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
done = sum(p["agirlik"] for p in d["puanlama"] if p["durum"] == "tamam")
part = sum(p["agirlik"] for p in d["puanlama"] if p["durum"] == "devam")
st.caption(f"✅ tamam · ⏳ devam ediyor · ⬜ başlanmadı — tamamlanan ağırlık %{done}, üzerinde çalışılan %{part}")

# --- aşamalar
st.subheader("Aşamalar")
for a in d["asamalar"]:
    n_ok = sum(s == "tamam" for _, s in a["adimlar"])
    with st.expander(f"{STATUS_ICON[a['durum']]} {a['ad']} — {n_ok}/{len(a['adimlar'])} adım", expanded=a["durum"] == "devam"):
        st.progress(n_ok / len(a["adimlar"]))
        for ad, s in a["adimlar"]:
            st.markdown(f"{STATUS_ICON[s]} {ad}")

# --- veri özeti
if data_ready():
    s = summary()
    st.subheader("Veri özeti")
    st_ = s["stats"]
    c = st.columns(4)
    c[0].metric("Görüntü", f"{sum(v['n_images'] for v in st_.values()):,}".replace(",", "."))
    c[1].metric("Hastalıklı bölge (örnek)", f"{sum(v['n_instances'] for v in st_.values()):,}".replace(",", "."))
    c[2].metric("Hastalık sınıfı", len(s["classes"]))
    c[3].metric("Temiz test görüntüsü", s["duplicates"]["test_clean"],
                delta=f"-{s['duplicates']['test_leaked']} sızan", delta_color="off",
                help="Eğitim/doğrulama kümesinde neredeyse aynısı bulunmayan test görüntüleri")
    st.caption(f"Veri hazırlama son çalıştırma: {s['olusturma_zamani']} · süre {s['sure_sn']} sn")
