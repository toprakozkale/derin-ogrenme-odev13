import streamlit as st

from common import C, basitce

st.title("🏷️ Sınıflandırma (YOLO-cls)")
basitce(
    "Sınıflandırmada model bir fotoğrafa bakıp **tek bir cevap** verir: \"bu fotoğrafta köşeli yaprak lekesi var\" gibi. "
    "İki boyutta model eğiteceğiz: küçük ve hızlı **nano**, biraz daha büyük **small**. Hangisinin daha doğru, hangisinin "
    "daha hızlı olduğunu karşılaştıracağız."
)

results = C.RUNS / "cls" / "results.csv"
if not results.exists():
    st.warning("⏳ **Henüz çalıştırılmadı.** Bu sayfa, Colab'daki eğitim bitip sonuç dosyaları `runs/cls/` klasörüne "
               "kopyalandığında otomatik dolacak. Tahmini zaman: 12–18 Kasım.")
    st.markdown(
        """
**Sonuçlar gelince burada görünecekler**
- Ana tablo: doğruluk, sınıf bazında kesinlik / duyarlılık / F1 (3 tekrarın ortalaması ± sapma) — **tüm test** ve **temiz test** ayrı ayrı
- 7×7 karmaşıklık matrisi: hangi hastalık hangisiyle karıştırıldı
- **Yaprak / meyve / çiçek** 3×3 matrisi: hocanın "Dikkat" maddesi
- Nano ↔ small karşılaştırması (McNemar testi), hız ve parametre sayısı
- En kötü 10 tahmin: fotoğraf, gerçek hastalık, modelin tahmini ve kısa yorum
"""
    )
    st.stop()

st.info("Sonuç dosyaları bulundu — görselleştirmeler 02 notebook'u tamamlanınca eklenecek.")
