"""Aşama 2 — Sınıflandırma sonuçları. runs/cls/ altındaki dosyalar geldikçe kendiliğinden dolar."""
import os
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from common import C, ROOT, SERIES, basitce

from src import cls
from src import metrics as M

st.title("🏷️ Sınıflandırma (YOLO26-cls)")
basitce(
    "Sınıflandırmada model bir fotoğrafa bakıp **tek bir cevap** verir: \"bu fotoğrafta köşeli yaprak lekesi var\" gibi. "
    "İki boyutta model eğitiyoruz: küçük ve hızlı **nano (n)**, biraz daha büyük **small (s)**. Her birini 3 kez (farklı "
    "seed) eğitip sonuçları ortalama ± sapma olarak veriyoruz; böylece şans eseri iyi/kötü çıkan bir denemeye aldanmayız."
)

# Geliştirme için başka klasör gösterilebilir (ör. duman testi); varsayılan asıl sonuçlar
OUT = Path(os.environ.get("ODEV13_CLS_DIR", C.RUNS / "cls"))
cls.OUT = OUT
if "duman" in OUT.name:
    st.error("⚠️ DUMAN TESTİ çıktısı gösteriliyor (1 epoch, verinin %5'i) — sayılar anlamsızdır, yalnız sayfa denemesi içindir.")

SIZE_COLOR = {"yolo26n-cls": SERIES[0], "yolo26s-cls": SERIES[1]}

# ------------------------------------------------------------------ deney düzeni
with st.expander("🔧 Deney düzeni", expanded=not (OUT / "results.csv").exists()):
    a = cls.TRAIN_ARGS
    st.markdown(f"""
| Ayar | Değer | Basitçe |
|---|---|---|
| Modeller | `yolo26n-cls`, `yolo26s-cls` (ImageNet ön-eğitimli) | küçük ve orta boy model |
| Giriş boyutu | 224 piksel (ablasyon: 384) | fotoğraf modele bu boyuta küçültülerek verilir |
| Epoch / erken durdurma | en fazla {a['epochs']} / {a['patience']} | val doğruluğu {a['patience']} tur iyileşmezse durur |
| Batch / optimizer | {a['batch']} / {a['optimizer']} (AdamW) | |
| Seed | 0, 1, 2 | aynı deney 3 kez |
| Model seçimi | doğrulama (val) doğruluğu | test kümesine bakılmaz |
| Test | tüm test (743) + temiz test (692) | temiz test = eğitimde benzeri olmayan fotoğraflar |
""")

# ------------------------------------------------------------------ koşuların durumu
st.subheader("Koşuların durumu")
rows = []
for s in cls.MAIN_RUNS + cls.ABLATION_RUNS:
    log = cls.train_log(s)
    done = cls.is_done(s)
    rows.append({"durum": "✅ bitti" if done else ("⏳ yarım" if len(log) else "⬜ başlamadı"),
                 "koşu": s.name, "model": s.model, "imgsz": s.imgsz, "seed": s.seed,
                 "biten epoch": len(log), "en iyi val doğruluğu": round(cls.best_val_acc(s), 4) if len(log) else None})
status = pd.DataFrame(rows)
st.dataframe(status, hide_index=True, width="stretch")
n_done = (status["durum"] == "✅ bitti").sum()
st.progress(n_done / len(status), text=f"{n_done}/{len(status)} koşu bitti")

if not (OUT / "results.csv").exists():
    st.warning(
        "⏳ **Test sonuçları henüz yok.** Eğitimler Colab'da (T4) `02_siniflandirma` notebook'uyla yapılır; sonuçlar Drive'da "
        "`odev13/runs/cls/` klasörüne yazılır. Yerele almak için Drive'da o klasörü **İndir** (zip) yapıp şunu çalıştırın:\n\n"
        "`.venv/bin/python scripts/import_runs.py ~/Downloads/cls-....zip`")
    st.stop()

# ------------------------------------------------------------------ ana tablo
res = pd.read_csv(OUT / "results.csv")
st.subheader("Ana sonuçlar")
basitce("**Doğruluk:** doğru bilinen fotoğrafların oranı. **Macro F1:** her hastalığa eşit ağırlık veren not — az fotoğraflı "
        "hastalıklarda da iyi mi, onu gösterir. **Temiz test:** eğitimde benzeri olmayan fotoğraflar; tüm testle farkı sızıntının etkisi.")
st.dataframe(cls.summary_table(res).rename(columns={"deney": "deney", "alt_kume": "alt küme", "seed": "seed sayısı"}),
             hide_index=True, width="stretch")

agg = (res.groupby(["model", "deney", "imgsz", "alt_kume"])[["accuracy", "macro_f1"]]
       .agg(["mean", "std"]).reset_index())
agg.columns = ["model", "deney", "imgsz", "alt_kume", "acc", "acc_sd", "f1", "f1_sd"]
agg["etiket"] = agg["model"] + " · " + agg["imgsz"].astype(str)
long = pd.concat([
    agg.assign(metrik="Doğruluk", deger=agg["acc"], sd=agg["acc_sd"].fillna(0)),
    agg.assign(metrik="Macro F1", deger=agg["f1"], sd=agg["f1_sd"].fillna(0))])
long["lo"], long["hi"] = long["deger"] - long["sd"], long["deger"] + long["sd"]
long["alt_kume"] = long["alt_kume"].map({"tum_test": "Tüm test", "temiz_test": "Temiz test"})
base = alt.Chart(long).encode(
    y=alt.Y("etiket:N", title=None), yOffset=alt.YOffset("alt_kume:N"),
    color=alt.Color("alt_kume:N", scale=alt.Scale(domain=["Tüm test", "Temiz test"], range=[SERIES[0], SERIES[2]]),
                    legend=alt.Legend(orient="top", title=None)))
bars = base.mark_bar(cornerRadiusEnd=4).encode(
    x=alt.X("deger:Q", title=None, scale=alt.Scale(domain=[0, 1])),
    tooltip=["etiket", "alt_kume", alt.Tooltip("deger:Q", format=".4f", title="ortalama"), alt.Tooltip("sd:Q", format=".4f", title="std")])
err = base.mark_rule(color="#52514e").encode(x="lo:Q", x2="hi:Q")
st.altair_chart((bars + err).properties(height=110).facet(row=alt.Row("metrik:N", title=None)), width="stretch")
st.caption("Çubuk = 3 seed ortalaması, çizgi = ± 1 standart sapma.")

# ------------------------------------------------------------------ sınıf bazında
st.subheader("Hastalık bazında F1")
pc = pd.read_csv(OUT / "per_class.csv").merge(res[["kosu", "model", "deney"]].drop_duplicates(), on="kosu")
pc = pc[(pc["alt_kume"] == "tum_test") & (pc["deney"] != "abl384")]
pcm = pc.groupby(["sinif", "model"])[["precision", "recall", "f1"]].mean().reset_index()
st.altair_chart(
    alt.Chart(pcm).mark_bar(cornerRadiusEnd=4).encode(
        y=alt.Y("sinif:N", title=None, sort=list(C.CLASSES), axis=alt.Axis(labelLimit=220)),
        yOffset="model:N", x=alt.X("f1:Q", title="F1 (3 seed ortalaması, tüm test)", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("model:N", scale=alt.Scale(domain=list(SIZE_COLOR), range=list(SIZE_COLOR.values())),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=["sinif", "model", alt.Tooltip("precision:Q", format=".3f"), alt.Tooltip("recall:Q", format=".3f"),
                 alt.Tooltip("f1:Q", format=".3f")]).properties(height=380), width="stretch")

# ------------------------------------------------------------------ karmaşıklık + Dikkat maddesi
st.subheader("Hangi hastalık hangisiyle karıştı? (hocanın \"Dikkat\" maddesi)")
basitce("Satır = gerçek hastalık, sütun = modelin tahmini. Köşegen dışındaki koyu hücreler sık karışmalardır. Alttaki 3×3 tablo "
        "hastalıkları **yaprak / meyve / çiçek** gruplarına toplar: gruplar arası hücreler, modelin örneğin bir yaprak hastalığını "
        "meyve hastalığı sanması demektir.")
size = st.segmented_control("Model", ["n", "s"], default="n", format_func=lambda x: f"yolo26{x}-cls")
runs = [s for s in cls.MAIN_RUNS if s.size == size and (OUT / "preds" / f"{s.name}_test.csv").exists()]
if runs:
    p = pd.concat([cls.load_preds(s) for s in runs])
    cm = M.confusion(p.y_true, p.y_pred)
    cmn = cm / np.clip(cm.sum(1, keepdims=True), 1, None)
    short = {c: c.replace("Powdery Mildew", "PM").replace("Anthracnose Fruit Rot", "Anthracnose") for c in C.CLASSES}
    cdf = pd.DataFrame([{"gerçek": short[a], "tahmin": short[b], "oran": cmn[i, j], "adet": int(cm[i, j])}
                        for i, a in enumerate(C.CLASSES) for j, b in enumerate(C.CLASSES)])
    order = [short[c] for c in C.CLASSES]
    heat = alt.Chart(cdf).encode(x=alt.X("tahmin:N", sort=order, axis=alt.Axis(labelAngle=-40)), y=alt.Y("gerçek:N", sort=order))
    st.altair_chart(
        (heat.mark_rect(stroke="white", strokeWidth=2).encode(
            color=alt.Color("oran:Q", scale=alt.Scale(range=["#f7f6f2", "#cde2fb", "#6da7ec", "#256abf", "#0d366b"], domain=[0, 1]),
                            legend=alt.Legend(title="satır oranı")),
            tooltip=["gerçek", "tahmin", "adet", alt.Tooltip("oran:Q", format=".2f")])
         + heat.mark_text(fontSize=11).encode(
            text=alt.Text("oran:Q", format=".2f"),
            color=alt.condition("datum.oran > 0.55", alt.value("white"), alt.value("#0b0b0b"))))
        .properties(height=380), width="stretch")
    st.caption(f"{len(runs)} seed'in test tahminleri toplandı ({int(cm.sum())} karar).")
    c1, c2 = st.columns([3, 2])
    c1.dataframe(M.group_confusion(cm).astype(int), width="stretch")
    rate = M.group_confusion_rate(cm)
    c2.metric("Gruplar arası karışma", f"%{rate['gruplar_arasi_toplam'] * 100:.1f}".replace(".", ","))
    c2.caption(f"yaprak→meyve %{rate['yaprak->meyve']*100:.1f} · meyve→yaprak %{rate['meyve->yaprak']*100:.1f}".replace(".", ","))
    st.markdown("**En sık karışmalar**")
    st.dataframe(M.top_confusions(cm, 8), hide_index=True, width="stretch")

# ------------------------------------------------------------------ istatistik + verimlilik
st.subheader("İstatistiksel güven ve hız")
c1, c2 = st.columns(2)
if (OUT / "mcnemar.csv").exists():
    c1.markdown("**McNemar testi (nano ↔ small, aynı seed)**")
    c1.dataframe(pd.read_csv(OUT / "mcnemar.csv"), hide_index=True)
    c1.caption("p < 0,05 ise iki modelin hataları arasındaki fark tesadüf değildir.")
if (OUT / "efficiency.csv").exists():
    e = pd.read_csv(OUT / "efficiency.csv")
    c2.markdown("**Verimlilik (batch = 1)**")
    c2.dataframe(e[["model", "imgsz", "parametre", "ms_cikarim", "ms_toplam_duvar", "cihaz"]].round(2), hide_index=True)
    c2.caption("ms_cikarim: yalnız modelin hesap süresi · ms_toplam_duvar: ön/son işlem dahil gerçek süre.")
if (OUT / "bootstrap.csv").exists():
    with st.expander("Bootstrap %95 güven aralıkları (koşu bazında)"):
        st.dataframe(pd.read_csv(OUT / "bootstrap.csv"), hide_index=True, width="stretch")

# ------------------------------------------------------------------ eğitim eğrileri
st.subheader("Eğitim eğrileri")
curves = []
for s in cls.MAIN_RUNS:
    log = cls.train_log(s)
    if len(log):
        acc = next(c for c in log.columns if "accuracy_top1" in c)
        curves.append(pd.DataFrame({"epoch": log["epoch"], "val doğruluğu": log[acc], "kayıp": log["train/loss"],
                                    "model": s.model, "koşu": f"{s.model} s{s.seed}"}))
if curves:
    cv = pd.concat(curves)
    base = alt.Chart(cv).encode(x="epoch:Q", detail="koşu:N",
                                color=alt.Color("model:N", scale=alt.Scale(domain=list(SIZE_COLOR), range=list(SIZE_COLOR.values())),
                                                legend=alt.Legend(orient="top", title=None)), tooltip=["koşu", "epoch"])
    c1, c2 = st.columns(2)
    c1.altair_chart(base.mark_line(strokeWidth=2).encode(y=alt.Y("kayıp:Q", title="eğitim kaybı")).properties(height=240), width="stretch")
    c2.altair_chart(base.mark_line(strokeWidth=2).encode(y=alt.Y("val doğruluğu:Q", scale=alt.Scale(zero=False))).properties(height=240), width="stretch")
    st.caption("Kayıp düşerken val doğruluğu da artıyorsa model öğreniyor; val düşmeye başlarsa ezberleme (aşırı öğrenme) başlamıştır.")

# ------------------------------------------------------------------ en kötü 10
if (OUT / "worst10.csv").exists():
    st.subheader("En kötü 10 tahmin")
    basitce("Val ile seçilen en iyi modelin, gerçek hastalığa **en düşük olasılığı verdiği** 10 yanlış tahmini. Raporda her biri "
            "kategorize edilerek yorumlanacak: arka plan, birden fazla hastalık, erken evre/küçük leke, ışık/bulanıklık, olası etiket hatası.")
    w = pd.read_csv(OUT / "worst10.csv")
    grid = st.columns(5)
    for i, r in w.iterrows():
        img = C.PROCESSED / "cls" / "test" / C.CLASS_SLUG[r["gercek"]] / f"{r['stem']}.jpg"
        with grid[i % 5]:
            if img.exists():
                st.image(str(img), width="stretch")
            st.caption(f"**{r['stem']}**  \ngerçek: {r['gercek']}  \ntahmin: {r['tahmin']} ({r['tahmin_guveni']:.2f})"
                       + ("  \n⚠️ gruplar arası" if r["gruplar_arasi"] else ""))
