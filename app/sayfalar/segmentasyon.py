"""Aşama 3 — Segmentasyon sonuçları (YOLO26-seg ↔ RF-DETR Seg Small). runs/seg/ altındaki dosyalar geldikçe kendiliğinden dolar."""
import gzip
import json
import os
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from common import C, SERIES, basitce

from src import metrics as M
from src import seg

st.title("✂️ Segmentasyon: YOLO-seg ve RF-DETR")
basitce(
    "Bölütlemede (segmentasyon) model yalnızca hastalığın adını söylemez, **hastalıklı bölgenin sınırını piksel piksel "
    "çizer**. İki farklı model ailesini aynı fotoğraflar, aynı çözünürlük (480) ve **aynı ölçüm koduyla** karşılaştırıyoruz: "
    "hızıyla bilinen **YOLO26-seg** (nano ve small) ve dönüştürücü (transformer) tabanlı **RF-DETR Seg Small**. "
    "Her model 3 kez (farklı seed) eğitilir; sonuçlar ortalama ± sapma olarak verilir."
)

# Geliştirme için başka klasör gösterilebilir (ör. duman testi); varsayılan asıl sonuçlar
OUT = Path(os.environ.get("ODEV13_SEG_DIR", C.RUNS / "seg"))
seg.OUT = OUT
SMOKE = "duman" in OUT.name
if SMOKE:
    st.error("⚠️ DUMAN TESTİ çıktısı gösteriliyor (1 epoch, küçük alt küme) — sayılar anlamsızdır, yalnız sayfa denemesi içindir.")

MODEL_COLOR = {"yolo26n-seg": SERIES[0], "yolo26s-seg": SERIES[1], "rfdetr-seg-small": SERIES[2]}
SHORT = {c: c.replace("Powdery Mildew", "PM").replace("Anthracnose Fruit Rot", "Anthracnose") for c in C.CLASSES}
ORDER = [SHORT[c] for c in C.CLASSES]
SUBSET_TR = {"tum_test": "Tüm test", "temiz_test": "Temiz test"}


def model_scale():
    return alt.Scale(domain=list(MODEL_COLOR), range=list(MODEL_COLOR.values()))


# ------------------------------------------------------------------ deney düzeni
with st.expander("🔧 Deney düzeni", expanded=not (OUT / "results.csv").exists()):
    y, r = seg.YOLO_TRAIN_ARGS, seg.RF_TRAIN_ARGS
    st.markdown(f"""
| Ayar | YOLO26-seg (n, s) | RF-DETR Seg Small | Basitçe |
|---|---|---|---|
| Ön-eğitim | COCO | COCO | genel nesnelerle önceden eğitilmiş; biz çilek hastalığına uyarlıyoruz |
| Giriş boyutu | {seg.IMGSZ} | {seg.IMGSZ} | iki modelde aynı (fotoğraflar 419×419) |
| Epoch / erken durdurma | en fazla {y['epochs']} / {y['patience']} | en fazla {r['epochs']} / {r['early_stopping_patience']} | val maske mAP'si iyileşmezse durur |
| Batch | {y['batch']} | {r['batch_size']} × {r['grad_accum_steps']} birikim | aynı etkin batch (16) |
| Seed | {", ".join(map(str, seg.SEEDS))} | {", ".join(map(str, seg.SEEDS))} | aynı deney birkaç kez |
| Model seçimi | doğrulama (val) | doğrulama (val) | test kümesine bakılmaz |
| Test | tüm test (743) + temiz test (692) | aynı | temiz test = eğitimde benzeri olmayan fotoğraflar |
""")

# ------------------------------------------------------------------ koşuların durumu
st.subheader("Koşuların durumu")
found = [p for p in (OUT / "train").glob("*") if p.is_dir()] if (OUT / "train").exists() else []
specs = list(seg.MAIN_RUNS) if not SMOKE else []
specs += [s for s in (seg.parse_run(p.name) for p in sorted(found)) if s and s not in specs]
rows = []
for s in specs:
    log = seg.train_log(s)
    done = seg.is_finished(s)
    rows.append({"durum": "✅ bitti" if done else ("⏳ sürüyor / yarım" if len(log) else "⬜ başlamadı"),
                 "koşu": s.name, "model": s.model, "seed": s.seed, "biten epoch": len(log),
                 "en iyi val maske mAP50-95": round(seg.best_val_map(s), 4) if len(log) else None,
                 "ağırlık": "bu bilgisayarda" if s.weights.exists() else ("Drive'da" if done else "—")})
status = pd.DataFrame(rows)
if len(status):
    st.dataframe(status, hide_index=True, width="stretch")
    n_done = int((status["durum"] == "✅ bitti").sum())
    st.progress(n_done / len(status), text=f"{n_done}/{len(status)} koşu bitti")
    st.caption("Durum, Colab'dan indirilip `scripts/import_runs.py` ile alınan dosyalara göredir; Colab'da süren eğitim burada "
               "ancak sonuçlar alındıktan sonra görünür.")

# ------------------------------------------------------------------ eğitim eğrileri (sonuçlardan önce: yarım koşular da görünsün)
curves = []
for s in specs:
    log = seg.train_log(s)
    if len(log):
        curves.append(log.assign(model=s.model, koşu=f"{s.model} s{s.seed}"))
if curves:
    st.subheader("Eğitim eğrileri")
    cv = pd.concat(curves)
    cv["kayip_oran"] = cv["egitim_kaybi"] / cv.groupby("koşu")["egitim_kaybi"].transform("first")
    base = alt.Chart(cv).encode(x="epoch:Q", detail="koşu:N", tooltip=["koşu", "epoch"],
                                color=alt.Color("model:N", scale=model_scale(), legend=alt.Legend(orient="top", title=None)))
    c1, c2 = st.columns(2)
    c1.altair_chart(base.mark_line(strokeWidth=2).encode(y=alt.Y("kayip_oran:Q", title="eğitim kaybı (ilk epoch = 1)")).properties(height=240),
                    width="stretch")
    c2.altair_chart(base.mark_line(strokeWidth=2).encode(
        y=alt.Y("val_mask_map50_95:Q", title="val maske mAP50-95", scale=alt.Scale(zero=False))).properties(height=240),
        width="stretch")
    st.caption("İki aile kaybı farklı ölçeklerde hesapladığı için kayıp, her koşunun ilk epoch'una oranla gösterilir. "
               "Val maske mAP artarken kayıp düşüyorsa model öğreniyor; val düşmeye başlarsa ezberleme (aşırı öğrenme) başlamıştır.")

if not (OUT / "results.csv").exists():
    st.warning(
        "⏳ **Test sonuçları henüz yok.** Eğitimler Colab'da (T4) `03_segmentasyon` notebook'uyla yapılıyor; sonuçlar Drive'da "
        "`odev13/runs/seg/` klasörüne yazılır. Yerele almak için Drive'da o klasörü **İndir** (zip) yapıp şunu çalıştırın:\n\n"
        "`.venv/bin/python scripts/import_runs.py ~/Downloads/seg-....zip`")
    st.stop()

# ------------------------------------------------------------------ ana tablo
res = pd.read_csv(OUT / "results.csv")
models = [m for m in MODEL_COLOR if m in set(res["model"])]
st.subheader("Ana sonuçlar")
basitce("**Maske mAP50 / mAP50-95:** model her lekeyi buldu mu ve sınırını ne kadar doğru çizdi (50: yarı yarıya örtüşme yeter; "
        "50-95: daha sıkı örtüşme eşiklerinin ortalaması). **mIoU:** her hastalık için tahmin edilen ve gerçek bölgelerin "
        "örtüşme oranı (kesişim / birleşim), 7 hastalığın ortalaması. **Dice:** IoU'ya benzer, biraz daha cömert bir örtüşme notu. "
        "Hepsi 0–1 arası; büyük olan iyi.")
st.dataframe(seg.summary_table(res).rename(columns={"alt_kume": "alt küme", "seed": "seed sayısı"}),
             hide_index=True, width="stretch")

METRICS = {"mask_map50_95": "Maske mAP50-95", "mask_map50": "Maske mAP50", "miou": "mIoU", "mdice": "Dice"}
agg = res.groupby(["model", "alt_kume"])[list(METRICS)].agg(["mean", "std"])
long = []
for m, title in METRICS.items():
    t = agg[m].reset_index().rename(columns={"mean": "deger", "std": "sd"})
    long.append(t.assign(metrik=title))
long = pd.concat(long)
long["sd"] = long["sd"].fillna(0)
long["lo"], long["hi"] = long["deger"] - long["sd"], long["deger"] + long["sd"]
long["alt_kume"] = long["alt_kume"].map(SUBSET_TR)
base = alt.Chart(long).encode(
    y=alt.Y("model:N", title=None, sort=models), yOffset=alt.YOffset("alt_kume:N"),
    color=alt.Color("alt_kume:N", scale=alt.Scale(domain=list(SUBSET_TR.values()), range=[SERIES[0], SERIES[2]]),
                    legend=alt.Legend(orient="top", title=None)))
bars = base.mark_bar(cornerRadiusEnd=4).encode(
    x=alt.X("deger:Q", title=None, scale=alt.Scale(domain=[0, 1])),
    tooltip=["model", "alt_kume", alt.Tooltip("deger:Q", format=".4f", title="ortalama"), alt.Tooltip("sd:Q", format=".4f", title="std")])
err = base.mark_rule(color="#52514e").encode(x="lo:Q", x2="hi:Q")
st.altair_chart((bars + err).properties(height=100).facet(row=alt.Row("metrik:N", title=None, sort=list(METRICS.values()))),
                width="stretch")
st.caption("Çubuk = seed ortalaması, çizgi = ± 1 standart sapma. mIoU ve Dice, her model için val'de seçilen güven eşiğiyle hesaplanır.")

with st.expander("🔎 Çapraz kontrol: bizim ölçüm kodu ↔ kütüphanenin kendi değeri (val)"):
    st.markdown("İki sütun yakın olmalı. Çok farklıysa sınıf eşlemesinde ya da maske dönüşümünde hata vardır.")
    st.dataframe(res[res["alt_kume"] == "tum_test"][["kosu", "val_mask_map50_95_kutuphane", "val_mask_map50_95_bizim", "esik"]]
                 .round(4), hide_index=True, width="stretch")

# ------------------------------------------------------------------ sınıf bazında
st.subheader("Hastalık bazında")
pc = pd.read_csv(OUT / "per_class.csv").merge(res[["kosu", "model"]].drop_duplicates(), on="kosu")
pc = pc[pc["alt_kume"] == "tum_test"]
metric = st.segmented_control("Ölçüt", ["ap50_95", "iou", "dice", "ap50"], default="ap50_95",
                              format_func={"ap50_95": "Maske AP50-95", "iou": "Piksel IoU", "dice": "Dice", "ap50": "Maske AP50"}.get)
pcm = pc.groupby(["sinif", "model"])[["ap50", "ap50_95", "iou", "dice"]].mean().reset_index()
st.altair_chart(
    alt.Chart(pcm).mark_bar(cornerRadiusEnd=4).encode(
        y=alt.Y("sinif:N", title=None, sort=list(C.CLASSES), axis=alt.Axis(labelLimit=220)),
        yOffset=alt.YOffset("model:N", sort=models),
        x=alt.X(f"{metric or 'ap50_95'}:Q", title="seed ortalaması, tüm test", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("model:N", scale=model_scale(), legend=alt.Legend(orient="top", title=None)),
        tooltip=["sinif", "model"] + [alt.Tooltip(f"{k}:Q", format=".3f") for k in ("ap50", "ap50_95", "iou", "dice")]
    ).properties(height=420), width="stretch")
st.caption("Boş çubuk: o hastalık bu alt kümede hiç yok (ortalamaya girmez).")

# ------------------------------------------------------------------ örnek karışma + Dikkat maddesi
st.subheader("Hangi hastalık hangisiyle karıştı? (hocanın \"Dikkat\" maddesi)")
basitce("Her gerçek leke, onunla en az yarı yarıya örtüşen bir tahminle eşleştirilir (sınıfa bakmadan). Satır = gerçek hastalık, "
        "sütun = modelin verdiği ad. **kaçırıldı:** hiçbir tahminle örtüşmeyen gerçek leke. **fazladan:** gerçekte olmayan bir "
        "yere çizilen tahmin. Alttaki 3×3 tablo hastalıkları **yaprak / meyve / çiçek** gruplarına toplar.")
model = st.segmented_control("Model", models, default=models[0], key="karisma_model")
runs = [s for s in specs if s.model == model and (OUT / "preds" / f"{s.name}_test_ornek_karisma.csv").exists()]
if runs:
    cm = sum(seg.load_instance_confusion(s) for s in runs)
    cmn = cm / np.clip(cm.sum(1, keepdims=True), 1, None)
    rlab, clab = ORDER + ["fazladan"], ORDER + ["kaçırıldı"]
    cdf = pd.DataFrame([{"gerçek": rlab[i], "tahmin": clab[j], "oran": cmn[i, j], "adet": int(cm[i, j])}
                        for i in range(8) for j in range(8) if not (i == 7 and j == 7)])
    heat = alt.Chart(cdf).encode(x=alt.X("tahmin:N", sort=clab, axis=alt.Axis(labelAngle=-40, labelOverlap=False)),
                                 y=alt.Y("gerçek:N", sort=rlab, axis=alt.Axis(labelOverlap=False)))
    st.altair_chart(
        (heat.mark_rect(stroke="white", strokeWidth=2).encode(
            color=alt.Color("oran:Q", scale=alt.Scale(range=["#f7f6f2", "#cde2fb", "#6da7ec", "#256abf", "#0d366b"], domain=[0, 1]),
                            legend=alt.Legend(title="satır oranı")),
            tooltip=["gerçek", "tahmin", "adet", alt.Tooltip("oran:Q", format=".2f")])
         + heat.mark_text(fontSize=11).encode(
            text=alt.Text("oran:Q", format=".2f"),
            color=alt.condition("datum.oran > 0.55", alt.value("white"), alt.value("#0b0b0b"))))
        .properties(height=400), width="stretch")
    m7 = cm[:7, :7]
    st.caption(f"{len(runs)} seed toplamı · eşleşen {int(m7.sum())}, kaçırılan {int(cm[:7, 7].sum())}, fazladan {int(cm[7, :7].sum())} örnek.")
    c1, c2 = st.columns([3, 2])
    c1.dataframe(M.group_confusion(m7).astype(int), width="stretch")
    rate = M.group_confusion_rate(m7)
    if not np.isnan(rate["gruplar_arasi_toplam"]):
        c2.metric("Gruplar arası karışma (eşleşen örneklerde)", f"%{rate['gruplar_arasi_toplam'] * 100:.1f}".replace(".", ","))
        c2.caption(f"yaprak→meyve %{rate['yaprak->meyve'] * 100:.1f} · meyve→yaprak %{rate['meyve->yaprak'] * 100:.1f}".replace(".", ",")
                   .replace("%nan", "—"))
    top = M.top_confusions(m7, 8)
    if len(top):
        st.markdown("**En sık karışmalar**")
        st.dataframe(top, hide_index=True, width="stretch")

# ------------------------------------------------------------------ verimlilik
if (OUT / "efficiency.csv").exists():
    st.subheader("Hız, boyut ve bellek")
    basitce("Aynı GPU'da (T4), fotoğraflar tek tek (batch = 1) verilerek ölçülür. **ileri geçiş:** yalnız ağın hesabı. "
            "**uçtan uca:** dosyayı okumaktan maskeleri çıkarmaya kadar her şey — gerçek kullanımdaki bekleme süresi.")
    e = pd.read_csv(OUT / "efficiency.csv")
    show = e[["model", "parametre", "ms_ileri_gecis", "ms_uctan_uca", "gpu_bellek_mb", "cihaz"]].copy()
    show["parametre"] = (show["parametre"] / 1e6).round(2).astype(str) + " M"
    st.dataframe(show.round(2), hide_index=True, width="stretch")
    acc = res[res["alt_kume"] == "tum_test"].groupby("model")["mask_map50_95"].agg(["mean", "std"]).reset_index()
    sc = acc.merge(e[["model", "ms_uctan_uca", "parametre"]], on="model")
    sc["std"] = sc["std"].fillna(0)
    sc["lo"], sc["hi"] = sc["mean"] - sc["std"], sc["mean"] + sc["std"]
    sc["M parametre"] = (sc["parametre"] / 1e6).round(1)
    b = alt.Chart(sc).encode(x=alt.X("ms_uctan_uca:Q", title="uçtan uca süre (ms/görüntü)", scale=alt.Scale(zero=False)),
                             color=alt.Color("model:N", scale=model_scale(), legend=alt.Legend(orient="top", title=None)))
    st.altair_chart(
        (b.mark_rule(strokeWidth=2).encode(y="lo:Q", y2="hi:Q")
         + b.mark_point(filled=True).encode(y=alt.Y("mean:Q", title="maske mAP50-95 (tüm test)", scale=alt.Scale(zero=False)),
                                            size=alt.Size("M parametre:Q", legend=alt.Legend(title="M parametre")),
                                            tooltip=["model", alt.Tooltip("mean:Q", format=".4f"), "ms_uctan_uca", "M parametre"]))
        .properties(height=300), width="stretch")
    st.caption("Sol üst köşe en iyisi (hızlı ve doğru). Nokta büyüklüğü parametre sayısını gösterir.")

# ------------------------------------------------------------------ en kötü 10
if (OUT / "worst10.csv").exists():
    st.subheader("En kötü 10 test görüntüsü")
    basitce("Her aileden doğrulama (val) ile seçilen en iyi model alınır; ikisinin görüntü başına IoU ortalaması en düşük 10 test "
            "fotoğrafı: **gerçek | YOLO | RF-DETR**. Raporda hata türlerine ayrılır: kaçırılan küçük leke, yanlış sınıf, taşan sınır, "
            "birleşen lekeler, olası etiket hatası.")
    w = pd.read_csv(OUT / "worst10.csv")
    st.dataframe(w.drop(columns=["image_id"]).round(3), hide_index=True, width="stretch")

    summ = json.loads((OUT / "summary.json").read_text()) if (OUT / "summary.json").exists() else {}
    best = {a: seg.parse_run(summ.get(f"secilen_{a}_val", "")) for a in ("yolo", "rfdetr")}
    thr_of = res[res["alt_kume"] == "tum_test"].set_index("kosu")["esik"].to_dict()
    have_preds = all(b is not None and seg.preds_path(b, "test").exists() for b in best.values())
    PAL = np.array([[230, 159, 0], [86, 180, 233], [0, 158, 115], [117, 72, 178], [0, 114, 178], [213, 94, 0],
                    [204, 121, 167]]) / 255

    @st.cache_data(show_spinner=False)
    def preds_by_image(path: str, mtime: float) -> dict:
        d = json.loads(gzip.decompress(Path(path).read_bytes()))
        by: dict = {}
        for r in d["results"]:
            by.setdefault(r["image_id"], []).append(r)
        return by

    def overlay(img: Image.Image, sem: np.ndarray, a: float = .55) -> np.ndarray:
        out = np.asarray(img, float) / 255
        for c in range(7):
            m = sem == c
            out[m] = (1 - a) * out[m] + a * PAL[c]
        return out

    if have_preds:
        st.markdown(" ".join(f"<span style='background:rgb({','.join(str(int(v * 255)) for v in PAL[i])});"
                             f"padding:1px 8px;border-radius:4px;color:white;font-size:0.8em'>{SHORT[c]}</span>"
                             for i, c in enumerate(C.CLASSES)), unsafe_allow_html=True)
        by = {a: preds_by_image(str(seg.preds_path(b, "test")), seg.preds_path(b, "test").stat().st_mtime) for a, b in best.items()}
        imgs = {im["id"]: im for im in seg.load_gt("test")["images"]}
        for _, r in w.iterrows():
            im = imgs[int(r["image_id"])]
            pil = Image.open(seg.COCO_DATA / "test" / im["file_name"]).convert("RGB")
            shape = (im["height"], im["width"])
            cols = st.columns(3)
            cols[0].image(overlay(pil, seg.gt_semantic("test", int(r["image_id"]))), width="stretch",
                          caption=f"{r['stem']} — gerçek: {r['gercek_siniflar']}")
            for col, a, name in ((cols[1], "yolo", "YOLO"), (cols[2], "rfdetr", "RF-DETR")):
                sem = seg.semantic_from_results(by[a].get(int(r["image_id"]), []), shape, thr_of.get(best[a].name, 0.5))
                col.image(overlay(pil, sem), width="stretch", caption=f"{name} ({best[a].model}) · IoU {r[f'iou_{a}']:.2f}")
    else:
        fig = (OUT / "figures" / "seg_en_kotu10_duman.png") if SMOKE else (C.ROOT / "rapor" / "figures" / "seg_en_kotu10.png")
        if fig.exists():
            st.image(str(fig), width="stretch")
        else:
            st.caption("Görseller için ham tahmin dosyaları (`runs/seg/preds/*.json.gz`) gerekir; Drive'dan `runs/seg` indirilip "
                       "`scripts/import_runs.py` ile alınınca burada çizilir.")
