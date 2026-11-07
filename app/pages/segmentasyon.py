import streamlit as st

from common import C, basitce

st.title("✂️ Segmentasyon: YOLO-seg ve RF-DETR")
basitce(
    "Bölütlemede (segmentasyon) model yalnızca hastalığın adını söylemez, **hastalıklı bölgenin sınırını piksel piksel "
    "çizer**. İki farklı model ailesini aynı fotoğraflar ve aynı ölçüm koduyla karşılaştıracağız: hızıyla bilinen **YOLO-seg** "
    "ve daha yeni bir mimari olan **RF-DETR**."
)

results = C.RUNS / "seg" / "results.csv"
if not results.exists():
    st.warning("⏳ **Henüz çalıştırılmadı.** Bu sayfa, Colab'daki eğitimler bitip sonuç dosyaları `runs/seg/` klasörüne "
               "kopyalandığında otomatik dolacak. Tahmini zaman: 25 Kasım – 3 Aralık.")
    st.markdown(
        """
**Sonuçlar gelince burada görünecekler**
- Ana karşılaştırma tablosu: mIoU, Dice, maske mAP50 ve mAP50-95, sınıf bazında IoU — **tüm test** ve **temiz test**
- Hız (ms/görüntü), parametre sayısı, GPU bellek kullanımı — aynı T4 GPU'da
- Hız–doğruluk grafiği
- Yan yana görseller: uzman çizimi · YOLO · RF-DETR
- En kötü 10 tahmin ve hata türleri (kaçırılan küçük leke, yanlış sınıf, taşan sınır…)
"""
    )
    st.stop()

st.info("Sonuç dosyaları bulundu — görselleştirmeler 03 notebook'u tamamlanınca eklenecek.")
