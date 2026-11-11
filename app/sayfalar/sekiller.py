import streamlit as st

from common import ROOT, basitce

st.title("📊 Rapor Şekilleri")
basitce(
    "Rapora girecek grafik ve görseller. Hepsi notebook'lar tarafından otomatik üretilir ve `rapor/figures/` klasörüne kaydedilir; "
    "notebook yeniden çalışınca burası da güncellenir. PDF sürümleri (varsa) rapora vektörel olarak eklenir."
)

FIG = ROOT / "rapor" / "figures"
# Bilinen şekillerin açıklamaları — yeni şekiller listede yoksa dosya adıyla gösterilir
TITLES = {'sinif_dagilimi': 'Sınıf dağılımı', 'ornekler': 'Örnek görüntüler', 'sizinti_esik': 'Sızıntı — eşik eğrisi', 'sizinti_ornek': 'Sızıntı — örnek çiftler', 'cls_egriler': 'Eğitim eğrileri', 'cls_karmasiklik_n': 'Karmaşıklık matrisi — nano', 'cls_karmasiklik_s': 'Karmaşıklık matrisi — small', 'cls_en_kotu10': 'En kötü 10 tahmin', 'roboflow_health_check': 'Roboflow Health Check'}
CAPTIONS = {
    "sinif_dagilimi": ("Aşama 1", "Bölmelere göre sınıf dağılımı (görüntü düzeyi etiket). Sınıf dengesizliği ~7×."),
    "ornekler": ("Aşama 1", "Her sınıftan bir eğitim görüntüsü, uzman poligonları bindirilmiş."),
    "sizinti_esik": ("Aşama 1", "pHash eşiğine göre benzeri eğitimde bulunan test/val oranı; kesikli çizgi seçilen eşik (12)."),
    "sizinti_ornek": ("Aşama 1", "Bölmeler arası yakın-yinelenen örnek çiftler (üst: eğitim, alt: test)."),
    "roboflow_health_check": ("Aşama 1", "Roboflow çapraz kontrolü — Health Check: 2.459 görüntü (41 bölme-içi kopya atlandı), 0 eksik etiket, 5.571 örnek, 7 sınıf."),
    "cls_egriler": ("Aşama 2", "Sınıflandırma eğitim kaybı ve doğrulama doğruluğu (nano ve small, 3 seed)."),
    "cls_karmasiklik_n": ("Aşama 2", "YOLO26n-cls karmaşıklık matrisi (3 seed toplamı, satır normalize)."),
    "cls_karmasiklik_s": ("Aşama 2", "YOLO26s-cls karmaşıklık matrisi (3 seed toplamı, satır normalize)."),
    "cls_en_kotu10": ("Aşama 2", "Val ile seçilen modelin en kötü 10 test tahmini."),
}

pngs = sorted(p for p in FIG.glob("*.png") if "duman" not in p.stem)
if not pngs:
    st.warning("Henüz şekil yok. 01 notebook'unu çalıştırın.")
    st.stop()

stages = sorted({s for s, _ in CAPTIONS.values()} | {CAPTIONS.get(p.stem, ("Diğer", ""))[0] for p in pngs})
sel = st.segmented_control("Aşama", stages, default=stages[0])
for p in [p for p in pngs if CAPTIONS.get(p.stem, ("Diğer", ""))[0] == sel]:
    stage, cap = CAPTIONS.get(p.stem, ("Diğer", p.stem))
    st.subheader(TITLES.get(p.stem, p.stem.replace("_", " ")))
    st.image(str(p), width="stretch")
    pdf = p.with_suffix(".pdf")
    st.caption(f"{cap}  \n`rapor/figures/{p.name}`" + (f" · vektörel: `{pdf.name}`" if pdf.exists() else ""))
    st.download_button("PNG indir", p.read_bytes(), file_name=p.name, mime="image/png", key=p.name)
    st.divider()

missing = [k for k, (s, _) in CAPTIONS.items() if s == sel and not (FIG / f"{k}.png").exists()]
if missing:
    st.info("Bu aşamada henüz üretilmemiş şekiller: " + ", ".join(f"`{m}`" for m in missing))
