import streamlit as st

st.title("📖 Sözlük")
st.caption("Ödevde geçen terimlerin sade açıklamaları. Hocanın sorabileceği kavramlar da burada.")

TERMS = {
    "Veri ve hazırlık": [
        ("Sınıflandırma (classification, CLS)", "Fotoğrafa bakıp tek bir etiket söylemek: \"bu fotoğrafta külleme var\"."),
        ("Bölütleme / segmentasyon (SEG)", "Fotoğraftaki hastalıklı bölgenin sınırını piksel piksel çizmek."),
        ("Örnek bölütleme (instance segmentation)", "Her hastalıklı bölgeyi ayrı ayrı çizmek: aynı yaprakta 3 leke varsa 3 ayrı çizim."),
        ("Poligon", "Bir bölgenin sınırını oluşturan noktalar dizisi (köşeleri birleştirilmiş çizgi)."),
        ("Etiket (annotation / label)", "Uzmanın fotoğrafa eklediği bilgi: hastalığın adı ve bölgenin sınırı."),
        ("LabelMe / YOLO / COCO biçimi", "Etiketleri saklamanın farklı dosya biçimleri. Veri LabelMe ile geldi; YOLO modeli kendi biçimini, RF-DETR COCO'yu istiyor."),
        ("Eğitim / doğrulama / test (train / val / test)", "Ders kitabı / deneme sınavı / final sınavı. Model eğitimden öğrenir, doğrulamayla ayarlanır, testle bir kez notlanır."),
        ("Sızıntı (data leakage)", "Test fotoğraflarının (neredeyse) aynısının eğitimde de bulunması. Notu yapay olarak yükseltir."),
        ("pHash (algısal hash)", "Fotoğrafın 64 bitlik 'parmak izi'. Benzer fotoğrafların parmak izleri de benzer olur."),
        ("Sınıf dengesizliği", "Bazı hastalıklardan çok, bazılarından az fotoğraf olması. Az olanı model zor öğrenir."),
    ],
    "Modeller ve eğitim": [
        ("YOLO (Ultralytics)", "Hızlı ve yaygın bir görüntü modeli ailesi. -cls sınıflandırma, -seg bölütleme yapar. n (nano) küçük, s (small) daha büyük sürüm."),
        ("RF-DETR", "Transformer tabanlı daha yeni bir model (Roboflow, 2025). Sınıflandırma yapmaz; ödevde yalnız bölütlemede kullanılır."),
        ("Ön-eğitimli model", "Milyonlarca genel fotoğrafla önceden eğitilmiş model; biz onu çilek hastalıklarına 'ince ayar' yapıyoruz."),
        ("Epoch (tur)", "Modelin tüm eğitim fotoğraflarını bir kez görmesi."),
        ("Seed (tohum)", "Rastgeleliğin başlangıç sayısı. Aynı seed = aynı sonuç (tekrarlanabilirlik). 3 farklı seed ile sonuçların ne kadar oynadığını ölçüyoruz."),
        ("Erken durdurma (early stopping)", "Doğrulama skoru bir süre iyileşmezse eğitimi durdurmak; ezberlemeyi önler."),
        ("Aşırı öğrenme (overfitting)", "Modelin eğitim fotoğraflarını ezberleyip yeni fotoğraflarda kötü olması."),
        ("GPU / T4", "Modelleri hızlı eğiten ekran kartı. Colab ücretsiz olarak NVIDIA Tesla T4 veriyor."),
    ],
    "Ölçütler": [
        ("Doğruluk (accuracy)", "Doğru tahmin edilen fotoğrafların oranı."),
        ("Kesinlik (precision)", "Model \"X hastalığı\" dediğinde ne sıklıkla haklı?"),
        ("Duyarlılık (recall)", "Gerçekten X olan fotoğrafların ne kadarını yakaladı?"),
        ("F1", "Kesinlik ve duyarlılığın dengeli ortalaması. Dengesiz sınıflarda doğruluktan daha adil."),
        ("Karmaşıklık matrisi (confusion matrix)", "Satırda gerçek hastalık, sütunda tahmin. Köşegen dışı hücreler hangi hastalıkların karıştırıldığını gösterir."),
        ("IoU (Intersection over Union)", "İki bölgenin örtüşmesi: kesişim / birleşim. 1 = birebir aynı, 0 = hiç örtüşmüyor."),
        ("mIoU", "Tüm sınıfların IoU ortalaması."),
        ("Dice", "IoU'ya benzer bir örtüşme ölçüsü: 2 × kesişim / (alan1 + alan2)."),
        ("mAP50 / mAP50-95", "Bölütleme modellerinin standart notu. mAP50: %50 örtüşme yeterli sayılır; mAP50-95: %50'den %95'e kadar giderek zorlaşan eşiklerin ortalaması."),
        ("Bootstrap güven aralığı", "Test kümesinden tekrar tekrar örnekleyerek skorun ne kadar oynayabileceğini tahmin etmek (ör. %95 GA: 0,82–0,86)."),
        ("McNemar testi", "İki modelin aynı test fotoğraflarındaki hatalarını karşılaştırıp farkın tesadüf olup olmadığını söyleyen istatistik testi."),
        ("Çıkarım süresi (ms/görüntü)", "Eğitilmiş modelin tek bir fotoğrafı işleme süresi."),
    ],
}

q = st.text_input("Ara", placeholder="ör. IoU, sızıntı, F1")
for group, items in TERMS.items():
    hits = [(t, d) for t, d in items if not q or q.lower() in (t + d).lower()]
    if not hits:
        continue
    st.subheader(group)
    for t, d in hits:
        st.markdown(f"**{t}** — {d}")
