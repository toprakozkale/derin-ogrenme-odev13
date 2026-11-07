import altair as alt
import pandas as pd
import streamlit as st

from common import C, SPLIT_COLOR, SPLIT_TR, basitce, class_legend, overlay, require_data, stems, summary

require_data()
s = summary()
S = s["stats"]

st.title("🖼️ Veri Seti")
basitce(
    "Elimizde **2.500 çilek fotoğrafı** var. Her fotoğrafta uzmanlar hastalıklı bölgelerin sınırını tek tek çizmiş "
    "(bunlara **örnek** ya da **poligon** diyoruz). Veri sahipleri fotoğrafları üçe bölmüş: model **eğitim** "
    "fotoğraflarından öğrenir, **doğrulama** ile eğitim sırasında kontrol edilir, **test** ile en sonda bir kez not alır."
)

fmt = lambda n: f"{n:,}".replace(",", ".")
c = st.columns(4)
for col, sp in zip(c, C.SPLITS):
    col.metric(f"{SPLIT_TR[sp]} · görüntü", fmt(S[sp]["n_images"]))
    col.caption(f"{fmt(S[sp]['n_instances'])} hastalıklı bölge")
c[3].metric("Görüntü boyutu (piksel)", "419×419")
c[3].caption("tüm görüntüler aynı boyutta")

st.subheader("Hastalıklara göre dağılım")
mode = st.segmented_control("Ne sayılsın?", ["Görüntü (sınıflandırma etiketi)", "Örnek (hastalıklı bölge)"],
                            default="Görüntü (sınıflandırma etiketi)")
key = "images_per_label" if mode.startswith("Görüntü") else "instances_per_class"
df = pd.DataFrame([{"Sınıf": cl, "Türkçe": C.CLASS_TR[cl], "Bölme": SPLIT_TR[sp], "Sayı": S[sp][key][cl]}
                   for sp in C.SPLITS for cl in C.CLASSES])
order = sorted(C.CLASSES, key=lambda cl: -S["train"][key][cl])
chart = (
    alt.Chart(df)
    .mark_bar(cornerRadiusEnd=4, height={"band": 0.8})
    .encode(
        y=alt.Y("Sınıf:N", sort=order, title=None, axis=alt.Axis(labelLimit=220)),
        yOffset=alt.YOffset("Bölme:N", sort=[SPLIT_TR[x] for x in C.SPLITS]),
        x=alt.X("Sayı:Q", title=None, axis=alt.Axis(grid=True, gridOpacity=0.25)),
        color=alt.Color("Bölme:N", sort=[SPLIT_TR[x] for x in C.SPLITS],
                        scale=alt.Scale(domain=[SPLIT_TR[x] for x in C.SPLITS], range=[SPLIT_COLOR[x] for x in C.SPLITS]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=["Sınıf", "Türkçe", "Bölme", "Sayı"],
    )
    .properties(height=420)
)
st.altair_chart(chart, width="stretch")
tr = S["train"]["images_per_label"]
st.caption(
    f"**Sınıflar dengesiz:** eğitimde en çok görüntü Leaf Spot'ta ({tr['Leaf Spot']}), en az Anthracnose Fruit Rot'ta "
    f"({tr['Anthracnose Fruit Rot']}) — yaklaşık {tr['Leaf Spot'] / tr['Anthracnose Fruit Rot']:.0f} kat fark. "
    "Az örnekli hastalıkları model daha zor öğrenir; bu yüzden yalnız doğruluk değil, sınıf bazında F1 de raporlanacak."
)
with st.expander("Tablo olarak göster"):
    st.dataframe(df.pivot_table(index=["Sınıf", "Türkçe"], columns="Bölme", values="Sayı").reset_index(), hide_index=True)

st.subheader("Bir fotoğrafta kaç hastalıklı bölge var?")
col1, col2 = st.columns([2, 1])
with col1:
    h = pd.DataFrame([{"Bölge sayısı": int(k), "Görüntü": v} for k, v in S["train"]["instances_per_image"]["hist"].items()])
    st.altair_chart(
        alt.Chart(h).mark_bar(cornerRadiusEnd=4, color=SPLIT_COLOR["train"])
        .encode(x=alt.X("Bölge sayısı:O", title="Fotoğraftaki bölge sayısı (eğitim kümesi)"),
                y=alt.Y("Görüntü:Q", title=None, axis=alt.Axis(gridOpacity=0.25)),
                tooltip=["Bölge sayısı", "Görüntü"])
        .properties(height=240), width="stretch")
with col2:
    ar = S["train"]["instance_area_ratio"]
    st.metric("Ortanca bölge büyüklüğü", f"%{ar['median'] * 100:.1f}".replace(".", ","), help="Bölge alanının fotoğraf alanına oranı (ortanca)")
    st.metric("Çok küçük bölgeler (<%1)", ar["small_lt_1pct"], help="Eğitim kümesinde fotoğrafın %1'inden küçük bölge sayısı")
    st.caption("Küçük lekeleri bulmak modeller için en zor kısım; hata analizinde ayrıca bakacağız.")

st.subheader("Örnek fotoğraflar")
basitce("Renkli alanlar uzmanların çizdiği hastalıklı bölgelerdir. Bir hastalık seçin, o hastalıktan örnekler görün.")
class_legend()
cc = st.columns([2, 1, 1])
cls = cc[0].selectbox("Hastalık", C.CLASSES, format_func=lambda x: f"{x} — {C.CLASS_TR[x]}")
split = cc[1].selectbox("Bölme", C.SPLITS, format_func=lambda x: SPLIT_TR[x])
seed = cc[2].number_input("Başka örnekler", min_value=0, max_value=999, value=0, step=1, help="Sayıyı değiştirince farklı fotoğraflar gelir")
cand = [x for x in stems(split) if x.rstrip("0123456789") == C.CLASS_SLUG[cls]]
pick = pd.Series(cand).sample(min(8, len(cand)), random_state=int(seed)).tolist() if cand else []
grid = st.columns(4)
for i, stem in enumerate(pick):
    with grid[i % 4]:
        st.image(overlay(split, stem), caption=f"{split}/{stem}", width="stretch")
