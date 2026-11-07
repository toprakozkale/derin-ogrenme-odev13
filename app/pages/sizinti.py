import altair as alt
import pandas as pd
import streamlit as st

from common import C, SERIES, SPLIT_TR, basitce, csv, doc_csv, plain, require_data, summary

require_data()
s = summary()
dup = s["duplicates"]
n_test, n_val = s["stats"]["test"]["n_images"], s["stats"]["val"]["n_images"]

st.title("🔍 Sızıntı Analizi")
basitce(
    "Modeli bir öğrenci gibi düşünün: **eğitim** fotoğrafları ders kitabı, **test** fotoğrafları final sınavı. "
    "Sınavdaki bir soru ders kitabında da varsa öğrenci ezberden cevaplar ve notu olduğundan yüksek çıkar. "
    "Bu veri setinde bazı fotoğrafların **neredeyse aynısı** (aynı yaprak, biraz kaydırılmış/kırpılmış) hem eğitimde hem "
    "testte var. Buna **sızıntı** denir. Bu sayfa sızıntının ne kadar olduğunu ve nasıl önlem aldığımızı gösterir."
)

c = st.columns(4)
c[0].metric("Seçilen benzerlik eşiği", dup["phash_threshold"], help="İki fotoğrafın 'parmak izi' en fazla bu kadar farklıysa aynı sahne sayılır")
c[1].metric("Sızan test fotoğrafı", f"{dup['test_leaked']} / {n_test}", f"%{dup['test_leaked'] / n_test * 100:.1f}", delta_color="off")
c[2].metric("Temiz test kümesi", dup["test_clean"], help="Eğitim/doğrulamada benzeri olmayan test fotoğrafları")
c[3].metric("Sızan doğrulama fotoğrafı", f"{dup['val_leaked']} / {n_val}", f"%{dup['val_leaked'] / n_val * 100:.1f}", delta_color="off")

st.subheader("Ne yapıyoruz?")
st.markdown(
    f"""
- Ödev kuralı gereği veri setinin **resmi bölmesini değiştirmiyoruz**.
- Test sonuçlarını **iki kez** raporlayacağız: tüm test kümesi ({n_test}) ve **temiz test** kümesi ({dup['test_clean']}).
  İki not arasındaki fark, sızıntının notu ne kadar şişirdiğini sayıyla gösterecek.
- Doğrulama kümesindeki sızıntı daha büyük (%{dup['val_leaked'] / n_val * 100:.0f}): eğitim sırasındaki doğrulama skorları iyimser olacak.
  Bunu raporda sınırlılık olarak yazacağız.
- Birebir aynı dosya ({dup['exact_groups']} grup) yalnızca **eğitim kümesinin kendi içinde** var; bölmeler arasında yok.
"""
)

st.subheader("Eşiğe göre sızıntı")
basitce(
    "Her fotoğrafın 64 bitlik bir 'parmak izi' (pHash) var. İki parmak izi arasındaki fark (mesafe) küçükse fotoğraflar "
    "birbirine çok benziyor demektir. Eşiği büyüttükçe daha çok çift 'aynı' sayılır, ama bir noktadan sonra gerçekten farklı "
    "fotoğraflar da karışmaya başlar."
)
lt = csv("leakage_table.csv")
long = pd.concat([
    pd.DataFrame({"Eşik": lt["esik"], "Küme": "Test", "Oran (%)": lt["test_oran"] * 100, "Adet": lt["test_sizan"]}),
    pd.DataFrame({"Eşik": lt["esik"], "Küme": "Doğrulama", "Oran (%)": lt["val_oran"] * 100, "Adet": lt["val_sizan"]}),
])
base = alt.Chart(long).encode(
    x=alt.X("Eşik:Q", title="Benzerlik eşiği (pHash mesafesi)", scale=alt.Scale(domain=[0, 20])),
    y=alt.Y("Oran (%):Q", title="Benzeri eğitimde olan fotoğraf (%)", axis=alt.Axis(gridOpacity=0.25)),
    color=alt.Color("Küme:N", scale=alt.Scale(domain=["Test", "Doğrulama"], range=[SERIES[2], SERIES[1]]),
                    legend=alt.Legend(orient="top", title=None)),
    tooltip=["Küme", "Eşik", alt.Tooltip("Oran (%):Q", format=".1f"), "Adet"],
)
rule = alt.Chart(pd.DataFrame({"x": [dup["phash_threshold"]]})).mark_rule(strokeDash=[4, 4], color="#898781").encode(x="x:Q")
label = alt.Chart(pd.DataFrame({"x": [dup["phash_threshold"]], "y": [95], "t": [f"seçilen eşik = {dup['phash_threshold']}"]})).mark_text(
    align="right", dx=-6, color="#898781").encode(x="x:Q", y="y:Q", text="t:N")
st.altair_chart((base.mark_line(strokeWidth=2) + base.mark_point(size=40, filled=True) + rule + label).properties(height=320), width="stretch")

st.subheader("Eşiği nasıl seçtik? (görsel kontrol)")
cal = doc_csv("phash_calibration.csv").rename(columns={"mesafe": "Mesafe", "incelenen_cift": "İncelenen çift",
                                                       "ayni_sahne": "Aynı sahne", "farkli": "Farklı fotoğraf", "not": "Not"})
st.dataframe(cal, hide_index=True, width="stretch")
st.caption("12'ye kadar incelenen tüm çiftler aynı sahne; 14'te yanlış eşleşmeler başlıyor. 14–16 arasında da gerçek kopyalar "
           "kaldığı için 12 eşiğiyle ölçülen sızıntı bir **alt sınırdır** (gerçek sızıntı biraz daha fazla olabilir).")

st.subheader("Çiftleri kendiniz inceleyin")
pairs = csv("dup_pairs.csv")
pairs = pairs[pairs["cross_split"]]
cc = st.columns([1, 1, 1])
kind = cc[0].selectbox("Hangi bölmeler?", ["test ↔ eğitim/doğrulama", "doğrulama ↔ eğitim", "hepsi"])
dist = cc[1].slider("Mesafe aralığı", 0, 20, (0, dup["phash_threshold"]))
page = cc[2].number_input("Sayfa", min_value=1, value=1, step=1)
if kind.startswith("test"):
    pairs = pairs[(pairs["a_split"] == "test") | (pairs["b_split"] == "test")]
elif kind.startswith("doğ"):
    pairs = pairs[((pairs["a_split"] == "val") & (pairs["b_split"] == "train")) |
                  ((pairs["a_split"] == "train") & (pairs["b_split"] == "val"))]
pairs = pairs[(pairs["dist"] >= dist[0]) & (pairs["dist"] <= dist[1])].reset_index(drop=True)
st.caption(f"{len(pairs)} çift bulundu. Her satırda solda ve sağda birbirine benzeyen iki fotoğraf var.")
per = 6
for _, r in pairs.iloc[(page - 1) * per: page * per].iterrows():
    a, b, m = st.columns([2, 2, 1])
    a.image(plain(r["a_split"], r["a_stem"], 260), caption=f"{SPLIT_TR[r['a_split']]} · {r['a_stem']}")
    b.image(plain(r["b_split"], r["b_stem"], 260), caption=f"{SPLIT_TR[r['b_split']]} · {r['b_stem']}")
    m.metric("Mesafe", int(r["dist"]))
    m.caption("✅ eşik içinde: sızıntı sayılır" if r["dist"] <= dup["phash_threshold"] else "eşik dışında: sayılmaz")
