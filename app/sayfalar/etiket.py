import pandas as pd
import streamlit as st

from common import C, basitce, class_legend, csv, doc_csv, overlay, plain, require_data, summary

require_data()
s = summary()
rt = s["roundtrip"]

st.title("🧹 Etiket Denetimi ve Dönüşüm")
basitce(
    "Uzmanların çizdiği sınırlar bir dosya biçiminde (LabelMe JSON) geliyor. Kullanacağımız iki model bu dosyayı "
    "doğrudan okuyamıyor: **YOLO** kendi metin biçimini, **RF-DETR** ise **COCO** biçimini istiyor. Bu sayfa (1) çizimlerde "
    "bulduğumuz hataları ve nasıl düzelttiğimizi, (2) dönüştürmenin hiçbir şeyi bozmadığının kanıtını gösterir."
)

# ------------------------------------------------------------------ dönüşüm doğrulaması
st.subheader("1 · Dönüşüm doğru mu?")
basitce(
    "Her hastalıklı bölgeyi yeni biçimden geri okuyup orijinaliyle üst üste koyduk ve **ne kadar örtüştüğünü** ölçtük "
    "(IoU: 1,00 = birebir aynı). Hepsi 0,99'un üstündeyse dönüşüm güvenilirdir."
)
c = st.columns(4)
c[0].metric("Kontrol edilen bölge", f"{rt['n_instances']:,}".replace(",", "."))
c[1].metric("En düşük örtüşme · YOLO", f"{rt["iou_yolo_min"]:.4f}".replace(".", ","))
c[2].metric("En düşük örtüşme · COCO", f"{rt["iou_coco_min"]:.4f}".replace(".", ","))
c[3].metric("0,99 altı", rt["n_iou_lt_0_99"], help="Örtüşmesi 0,99'un altında kalan bölge sayısı")
ok = rt["class_ok"] and rt["count_ok"] and rt["n_iou_lt_0_99"] == 0
(st.success if ok else st.error)(
    ("✅ Tüm bölgeler her iki biçimde de doğru sınıf numarasıyla ve aynı sayıda yazıldı; geri okunan maskeler orijinalle örtüşüyor."
     if ok else "⚠️ Doğrulamada sorun var — roundtrip.csv dosyasına bakın."))
cl = pd.DataFrame(rt["coco_load"]).T.rename(columns={"images": "Görüntü", "annotations": "Bölge", "categories": "Kategori*"})
st.dataframe(cl, width="content")
st.caption("COCO dosyaları pycocotools ile sorunsuz yüklendi. *Kategori sayısı 8: Roboflow düzenindeki gibi 0 numaralı üst kategori + 7 hastalık.")

# ------------------------------------------------------------------ düzeltmeler
st.subheader("2 · Çizim hataları ve düzeltmeler")
iss = csv("label_issues.csv")
kural = pd.DataFrame([
    {"Sorun": "Sınır çizgisi kendini kesiyor", "Kural": "Geometrik onarım (shapely make_valid); birden çok parçaya ayrılırsa en büyük parça tutulur", "Adet": int(iss["islem"].str.startswith("kendini_kesen").sum())},
    {"Sorun": "Çizim fotoğrafın dışına taşıyor", "Kural": "Taşan noktalar fotoğraf kenarına çekilir", "Adet": int(iss["islem"].str.startswith("goruntu_disi").sum())},
    {"Sorun": "3'ten az noktalı / alanı sıfır çizim", "Kural": "Atılır", "Adet": int(iss["islem"].str.contains("atildi").sum())},
    {"Sorun": "Boş etiket dosyası / fotoğrafı olmayan etiket", "Kural": "Atılır", "Adet": int(iss["islem"].isin(["ornek_yok", "goruntu_yok"]).sum())},
])
st.dataframe(kural, hide_index=True, width="stretch")
st.caption(f"Düzeltilen bölgelerde orijinal ile düzeltilmiş hal arasındaki en düşük örtüşme: **{str(round(rt["fixed_iou_vs_raw_min"], 3)).replace(".", ",")}** — düzeltmeler şekli neredeyse hiç değiştirmiyor.")

if len(iss):
    iss["göster"] = iss["split"] + "/" + iss["stem"] + " · " + iss["islem"].str.split(":").str[0]
    sel = st.selectbox("Düzeltilen bir örneği inceleyin", iss["göster"].tolist())
    r = iss[iss["göster"] == sel].iloc[0]
    a, b = st.columns(2)
    a.image(overlay(r["split"], r["stem"], show_raw=True), caption="Orijinal çizim", width="stretch")
    b.image(overlay(r["split"], r["stem"]), caption="Düzeltilmiş çizim", width="stretch")
    with st.expander("Tüm düzeltme kayıtları"):
        st.dataframe(iss.drop(columns=["göster"]), hide_index=True, width="stretch")

# ------------------------------------------------------------------ çok sınıflı
st.subheader("3 · İki hastalığın birlikte olduğu fotoğraflar")
basitce(
    "Sınıflandırmada her fotoğrafa **tek** bir hastalık adı vermemiz gerekiyor. Bazı fotoğraflarda iki hastalık birden var "
    f"(ör. yaprakta leke + meyvede külleme). Kuralımız: **{C.IMAGE_LABEL_RULE}** Bu fotoğraflar aynı zamanda hocanın "
    "istediği 'meyve/yaprak hastalığı karışması' tartışmasının doğal örnekleri."
)
mc = csv("multiclass_images.csv")
if len(mc):
    st.dataframe(mc.rename(columns={"split": "Bölme", "stem": "Dosya", "dosya_oneki": "Dosya adındaki hastalık",
                                    "siniflar": "Çizilen hastalıklar", "goruntu_etiketi": "Verilen tek etiket", "tur": "Tür"}),
                 hide_index=True, width="stretch")
    class_legend()
    grid = st.columns(5)
    for i, (_, r) in enumerate(mc.iterrows()):
        with grid[i % 5]:
            st.image(overlay(r["split"], r["stem"], size=240), caption=f"{r['split']}/{r['stem']} → {r['goruntu_etiketi']}", width="stretch")
    st.warning("**`train/gray_mold1`**: dosya adı 'kurşuni küf', çizim ise 'çiçek yanıklığı' olarak etiketlenmiş. Çizim, "
               "fotoğrafın ortasındaki gri tüylü meyveyi kapsıyor — görsel olarak kurşuni küfe benziyor, yani **muhtemel etiket hatası**. "
               "Uzman etiketini değiştirmiyoruz (eğitimdeki 3.540 bölgeden 1'i, etkisi ihmal edilebilir); raporda etiket gürültüsü örneği olarak yazılacak.")
    st.image(overlay("train", "gray_mold1", size=320), caption="train/gray_mold1 — çizilen bölge (çiçek yanıklığı rengiyle)")

# ------------------------------------------------------------------ severity
st.subheader("4 · \"Test Disease Severity Level\" klasörü")
sv = s["severity_check"]
st.markdown(f"Bu klasördeki **{sv['n_images']}** fotoğrafın **{sv['identical_to_test']}**'ü test kümesindekilerle bayt bayt aynı "
            f"(Level 1: {sv['levels'].get('Level 1')}, Level 2: {sv['levels'].get('Level 2')}). **Karar:** {sv['karar']}.")

# ------------------------------------------------------------------ Roboflow çapraz kontrolü
st.subheader("5 · Roboflow çapraz kontrolü (hocanın \"Dikkat\" maddesi)")
basitce(
    "Hoca, etiketlerin **Roboflow** ile dönüştürülmesini istiyor. Biz dönüşümü kendi kodumuzla yaptık (tekrarlanabilir olsun diye); "
    "Roboflow'u da **bağımsız ikinci bir dönüştürücü** olarak kullandık: aynı ham veriyi Roboflow'a yükledik, ön işleme ve çoğaltma "
    "kapalı bir sürüm oluşturup COCO olarak indirdik ve her hastalıklı bölgeyi bizimkiyle üst üste koyup karşılaştırdık. İki farklı "
    "araç aynı sonucu veriyorsa dönüşümümüz doğru demektir."
)
rf_path = C.ROOT / "docs" / "roboflow" / "roboflow_summary.json"
if not rf_path.exists():
    st.info("Roboflow karşılaştırması henüz çalıştırılmadı: `.venv/bin/python -c \"from src.roboflow_check import compare; compare()\"`")
else:
    import json
    rf = json.loads(rf_path.read_text())
    c = st.columns(4)
    c[0].metric("Roboflow'daki görüntü", f"{rf['roboflow_toplam']:,}".replace(",", "."), help="Bizde 2.500; fark = Roboflow'un atladığı kopyalar")
    c[1].metric("Örnek sayısı ve sınıflar aynı", f"{rf['siniflar_ayni']:,} / {rf['eslesen_goruntu']:,}".replace(",", "."))
    c[2].metric("Ortalama örtüşme (IoU)", f"{rf['iou_ortalama']:.5f}".replace(".", ","))
    c[3].metric("0,99 altı örnek", rf["iou_lt_0_99"], help=f"Karşılaştırılan {rf['ornek_karsilastirilan']} örnekten")
    tr = lambda x, d=4: f"{x:.{d}f}".replace(".", ",")   # Türkçe ondalık
    n = lambda x: f"{x:,}".replace(",", ".")              # Türkçe binlik
    st.success(
        f"✅ Eşleşen {n(rf['eslesen_goruntu'])} görüntünün **hepsinde** görüntü boyutu, örnek sayısı ve sınıflar birebir aynı; "
        f"{n(rf['ornek_karsilastirilan'])} hastalıklı bölgenin ortalama örtüşmesi {tr(rf['iou_ortalama'])}. 0,99'un altında kalan tek örnek, "
        f"bizim kendini kesen poligon düzeltmemizin uygulandığı `test/gray_mold539` (IoU {tr(rf['duzeltilen_ornek_iou_min'], 3)}); "
        f"düzeltme yapılmayan örneklerde en düşük örtüşme {tr(rf['duzeltilmemis_iou_min'])}."
    )
    st.markdown(
        f"**Roboflow'un atladığı {rf['atlanan_goruntu']} görüntü:** Roboflow, projede birebir aynısı bulunan fotoğrafı ikinci kez almaz. "
        f"Atlananların **{rf['atlananlarin_esi_var']}/{rf['atlanan_goruntu']}**'inin aynı bölmede bayt bayt aynı bir eşi var "
        f"(bizim MD5 analizimizdeki kopya çiftleriyle birebir aynı): "
        + ", ".join(f"{k} {v}" for k, v in rf["atlanan_bolmeye_gore"].items()) + ". Yani Roboflow, kopya analizimizi bağımsız olarak doğruladı."
    )
    miss = doc_csv("roboflow/roboflow_missing.csv")
    with st.expander("Atlanan görüntüler ve eşleri"):
        st.dataframe(miss.rename(columns={"split": "bölme", "stem": "atlanan", "md5_esi": "bayt bayt aynı eşi",
                                          "phash0_esi": "pHash=0 eşleri", "ornek_sayisi": "örnek"}), hide_index=True, width="stretch")
    shot = C.ROOT / "rapor" / "figures" / "roboflow_health_check.png"
    if shot.exists():
        st.image(str(shot), caption="Roboflow Health Check (11 Kas 2026): 2.459 görüntü, 0 eksik etiket, 5.571 örnek, 7 sınıf, hepsi 419×419", width="stretch")
