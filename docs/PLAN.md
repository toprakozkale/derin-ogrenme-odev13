# Görev 13 — Çilek Hastalığı: Sınıflandırma (YOLO-cls) ve Bölütleme (YOLO-seg vs RF-DETR)

## Context (Neden bu plan?)
Yüksek lisans Derin Öğrenme dersinde vize+final yerine geçen ödev. Öğrenciye Görev 13 atandı:
Strawberry Disease Detection Dataset (Kaggle `usmanafzaal/strawberry-disease-detection-dataset`; Afzaal ve ark., *Sensors* 2021,
JBNU AI Lab), 2.500 görüntü, 7 hastalık sınıfı, poligon (JSON) örnek segmentasyon etiketleri.
Resmi bölme var: train 1.450 / val 307 / test 743 (+ test'in şiddet düzeyine göre gruplanmış kopyası olduğu düşünülen
"Test Disease Severity Level" klasörü, 206+537 — ek veri olarak kullanılmaz, kopya olduğu hash ile doğrulanır).
**Lisans/kaynak (7 Kas 2026'da Kaggle'dan okundu):** Kaggle lisans alanı "Unknown"; açıklamada yazarlar değiştirme ve kullanım
izni veriyor, makaleye atıf istiyor. `Dataset.txt` görüntülerin **çok sayıda dış kaynaktan** (Bugwood.org, üniversite yayım
servisleri, PaDIL, Plantix, alamy.com, bloglar vb.) toplandığını gösteriyor — yani tek bir seradan değil, heterojen web görüntüleri.
- **1. rapor (CLS):** 24 Kasım 2026 23:59 — YOLO-cls, en az iki model boyutu.
- **2. rapor (SEG):** 8 Aralık 2026 23:59 — YOLO-seg ve RF-DETR, aynı bölmeyle karşılaştırma.
- Teslim: PDF rapor (4–6 sayfa) + çalıştırılabilir Colab linki + 10 dk sunum → tekinler.cs@gmail.com
- Puanlama: Veri hazırlama %20 · CLS %20 · SEG karşılaştırma %25 · Hata analizi + "Dikkat" maddesi %20 · Rapor/sunum %15
- "Dikkat": JSON poligonlar YOLO-seg ve COCO'ya çevrilecek; **meyve/yaprak hastalıklarının karışması karmaşıklık matrisiyle incelenecek.**

Hedef: Puanlama tablosundaki her maddeyi karşılayan, **tekrarlanabilir, adil karşılaştırmalı, istatistiksel olarak savunulabilir**
ve akademik yazım kurallarına uygun bir çalışma.

**İş bölümü:** Claude kodu/notebookları, rapor taslağını ve sunum taslağını yazar; öğrenci Colab'da (T4 GPU) çalıştırır,
çıktıları geri getirir. Rapordaki tüm sayılar öğrencinin gerçek çalıştırmalarından gelir — hiçbir sonuç uydurulmaz/tahmin edilmez.

---

## 1. Akademik kurallar (tüm çalışma boyunca)
1. **Tekrarlanabilirlik:** sabit tohum (seed), `pip freeze` ile sürüm kaydı, tüm hiperparametreler YAML/tabloda,
   bölme listeleri (`split_train.txt` vb.) Drive'a kaydedilir ve SHA ile raporlanır. Notebook baştan sona tek seferde çalışır.
2. **Veri sızıntısı yok:** test kümesi yalnızca en sonda, bir kez kullanılır. Model/hiperparametre seçimi yalnız doğrulama (val) ile.
   Resmi train/val/test arasında yinelenen/çok benzer görüntü kontrolü (perceptual hash) yapılır ve raporlanır.
3. **Adil karşılaştırma:** YOLO-seg ve RF-DETR aynı bölme, aynı giriş çözünürlüğü, benzer eğitim bütçesi (epoch/süre),
   aynı GPU ve **aynı değerlendirme kodu** (pycocotools + kendi mIoU/Dice fonksiyonumuz) ile ölçülür.
4. **İstatistiksel güven:** ana deneyler 3 farklı seed ile → ortalama ± std; test metriklerine bootstrap %95 güven aralığı;
   nano vs small için McNemar testi.
5. **Dürüst raporlama:** sınırlılıklar (veri büyüklüğü, heterojen web kaynaklı görüntüler ↔ gerçek sera telefon fotoğrafları arasındaki
   alan kayması, kaynaklar arası olası yinelenenler, etiket gürültüsü) açıkça yazılır.
6. **Kaynakça:** IEEE stili. Asgari: Afzaal 2021 (veri), Ultralytics YOLO, RF-DETR (Robinson ve ark. 2025), DETR (Carion 2020),
   COCO (Lin 2014), Mask R-CNN (He 2017), Dice/IoU tanımı. Veri atfı: Kaggle lisansı "Unknown", yazarların açıklamadaki kullanım izni ve atıf isteği aynen belirtilir;
   görüntülerin dış kaynaklardan derlendiği (`Dataset.txt`) not edilir. (Sensors makalesinin kendisi CC BY 4.0'dır — verinin değil.)
7. **Yapay zekâ beyanı:** okul/hoca politikasına göre raporda kısa bir "Yapay zekâ araçlarının kullanımı" notu (önerilir).

---

## 2. Dosya / klasör yapısı
Yerelde (`/home/user/odev13/`) üretilip Google Drive'a `MyDrive/odev13/` olarak yüklenecek:
```
odev13/
  notebooks/
    01_veri_hazirlama.ipynb     # indirme, inceleme, temizlik, dönüşüm, bölme
    02_siniflandirma.ipynb      # YOLO-cls n/s, metrikler, hata analizi
    03_segmentasyon.ipynb       # YOLO-seg vs RF-DETR, ortak değerlendirme, hata analizi
  src/                          # notebooklarda %run / import edilen ortak fonksiyonlar
    convert.py                  # JSON→YOLO-seg txt, JSON→COCO, görüntü etiketi kuralı, Roboflow çıktısıyla karşılaştırma
    metrics.py                  # mIoU, Dice, bootstrap CI, McNemar, gruplanmış karmaşıklık matrisi
    viz.py                      # en kötü 10 tahmin panelleri, maske bindirme
  data/ (Drive'da, git dışı)   # raw/, cls/, yoloseg/, coco/, roboflow_export/
  runs/ (Drive'da)              # eğitim çıktıları, best.pt, sonuç CSV'leri
  rapor/
    rapor1_cls.docx → PDF      # Word; akademik şablon: 2 sütun/IEEE benzeri, numaralı şekil/tablo,
    rapor2_seg.docx → PDF      #   Word'ün Başvurular→Kaynakça (IEEE stili) ve Resim Yazısı özellikleriyle
    sunum.pptx
  ENV.md                        # sürümler, GPU, seed'ler
```

---

## 3. Aşama 0 — Kurulum (7–9 Kasım)  ·  *öğrenci*
- **Kaggle API anahtarı (Claude'un gerçek veriyle test edebilmesi için):**
  1. kaggle.com'a giriş → sağ üst profil resmi → **Settings** → **API** bölümü → **Create New Token**.
  2. Bilgisayara `kaggle.json` iner; içinde `{"username": "...", "key": "..."}` vardır. Bu bir şifredir, sohbete yapıştırılmaz.
  3. Claude bu oturumda güvenli form açar (`request_credential`, host `www.kaggle.com`, kullanıcı adı + şifre türü):
     *kullanıcı adı* = `username`, *şifre* = `key`. Claude değeri görmez; ağ vekili yalnızca Kaggle isteklerine ekler.
  4. Ayrıca Kaggle'daki veri sayfasında kullanım koşulları gerekiyorsa kabul edilir. Aynı `kaggle.json` Colab'da da kullanılır.
  5. İndirme başka bir alan adına yönlenip engellenirse Claude yalnızca o alan adı için ortam izni ister (onay öğrencide).
- VS Code + Python, Jupyter, Google Colab eklentileri; Colab çekirdeği → T4 GPU; `!nvidia-smi` doğrulaması.
- Kaggle hesabı + veri kullanım koşulları; Drive bağlantısı (gerekirse tarayıcıdaki Colab'dan `drive.mount`).
- **Roboflow hesabı** (ücretsiz plan yeterli) — Aşama 1 adım 8'deki çapraz kontrol için; yükleme/dışa aktarma öğrenci tarafından
  web arayüzünden yapılır, Claude adım adım yönlendirir.
- Kurulum hücresi: `ultralytics`, `rfdetr[train]`, `pycocotools`, `kagglehub`, `imagehash`, `scikit-learn`, `statsmodels`.

## 4. Aşama 1 — Veri hazırlama: `01_veri_hazirlama.ipynb` (8–12 Kasım)  ·  %20
1. **İndirme:** `kagglehub.dataset_download(...)` → Drive'a kopya; dosya sayısı ve klasör yapısı doğrulanır.
2. **Keşifsel analiz (EDA):** bölme başına sınıf dağılımı (grafik), görüntü boyutları, görüntü başına örnek sayısı,
   örnek alanı dağılımı, 7 sınıftan örnek görseller + poligon bindirmesi.
3. **Etiket denetimi:** bozuk/boş JSON, görüntüsü olmayan etiket, kendini kesen/3 noktadan az poligon, görüntü dışına taşan
   koordinat → sayılarıyla raporlanır, kural tablosuyla düzeltilir/atılır.
4. **Görüntü düzeyi etiket kuralı:** görüntüdeki örneklerin sınıfları tek ise o sınıf; birden fazla sınıf varsa
   **toplam poligon alanı en büyük olan sınıf**. Çok sınıflı görüntü sayısı raporlanır (kural raporda açıkça yazılır).
5. **Bölme:** veri setinin resmi train/val/test bölmesi kullanılır (ödev kuralı). Yoksa sınıfa göre tabakalı 70/15/15, seed=42.
   pHash ile bölmeler arası yinelenen kontrolü; bulunursa sızıntı olarak raporlanır ve test'ten çıkarılır (öncesi/sonrası not edilir).
   **Uygulanan karar (11 Kas 2026, kullanıcı onayı):** resmi bölme değiştirilmez; sızan 51 test görüntüsü çıkarılmış "temiz test" (692)
   tüm testin (743) yanında ayrıca raporlanır. Val sızıntısı (%35) sınırlılık olarak yazılır.
6. **Dönüşümler (kodla, tekrarlanabilir):**
   - CLS: `data/cls/{train,val,test}/<sınıf>/img.jpg` (Ultralytics klasör yapısı).
   - YOLO-seg: `images/` + `labels/*.txt` (`sınıf x1 y1 …`, 0–1 normalize) + `data.yaml`.
   - COCO: `{train,valid,test}/_annotations.coco.json` (RF-DETR beklentisi; segmentation=poligon, bbox, area).
7. **Dönüşüm doğrulaması (puanın kritik kısmı):** her formattan maske geri rasterize edilip orijinal poligon maskesiyle IoU
   karşılaştırılır (beklenen ≈1.0); rastgele 12 görüntüde görsel kontrol paneli; `pycocotools` ile COCO dosyası yüklenebilirlik testi.
8. **Roboflow çapraz kontrolü (zorunlu — Görev 13 "Dikkat" maddesi Roboflow'u açıkça anıyor):**
   Ana hat Python betiğimiz olarak kalır (Colab'da baştan sona tekrarlanabilir olsun diye); Roboflow bağımsız ikinci dönüştürücü
   olarak kullanılır ve iki çıktı birbirine karşı doğrulanır.
   1. Roboflow'da *Instance Segmentation* projesi açılır; ham görüntüler + orijinal JSON etiketleri yüklenir (Roboflow bu JSON
      biçimini okuyamazsa bizim ürettiğimiz COCO dosyası yüklenir ve kontrol "Roboflow'un okuduğu = bizim yazdığımız" yönünde yapılır;
      hangi yolun izlendiği raporda yazılır).
   2. Sınıf adları ve 7 sınıfın sayıları Roboflow *Health Check* ekranında bizim EDA tablomuzla karşılaştırılır.
   3. Dışa aktarma: **YOLOv8/YOLO11 Segmentation** ve **COCO Segmentation**; ön işleme ve çoğaltma **kapalı** (resize yok,
      augmentation yok), bölme bizim bölmemizle aynı tutulur → `data/roboflow_export/` altına indirilir.
   4. `convert.py → compare_with_roboflow()`: görüntü eşleştirme (dosya adı), görüntü başına örnek sayısı ve sınıfı,
      örnek bazında poligon IoU'su (bizim maske vs Roboflow maskesi). Rapora: eşleşen örnek oranı, IoU ort./min,
      IoU < 0.99 olan örneklerin listesi ve nedeni (ör. EXIF döndürme, kenar kırpma, poligon sadeleştirme).
   5. **Rapora girecekler:** Roboflow Health Check ve etiketli örnek görüntü ekran görüntüleri (1 şekil, iki panel) +
      karşılaştırma tablosu (1 küçük tablo). Roboflow proje linki ve sürüm numarası Ek'e.
   6. Gizlilik: görüntülerin bir kısmı üçüncü taraf telifli kaynaklardan (ör. alamy.com) geldiği için Roboflow projesi
      **mümkünse özel (private)** açılır. Hesapta yalnız herkese açık seçenek varsa: kontrol ve dışa aktarma bitince proje silinir
      ya da yalnızca küçük bir doğrulama alt kümesi yüklenir; hangisinin yapıldığı raporda yazılır.
      **Durum (7 Kas 2026):** Workspace "Topraks Workspace" (`topraks-workspace-k39ij`), proje "Strawberry Disease ..."
      — Instance Segmentation, **Private**, 0 görüntü. Yükleme Aşama 1'de, veri kontrol edildikten sonra.
      Yüklemeden önce özel projenin deneme süresine bağlı olup olmadığı (Settings → Plan) kontrol edilir.

## 5. Aşama 2 — Sınıflandırma: `02_siniflandirma.ipynb` (12–18 Kasım)  ·  %20
- **Modeller:** `yolo11n-cls` ve `yolo11s-cls` (ImageNet ön-eğitimli; o tarihte güncel Ultralytics sürümü neyse aynı aile, sürüm raporlanır).
- **Ortak ayarlar:** imgsz=224 (ana), epochs=100 + early stopping (patience=20), varsayılan augmentasyon, AdamW/auto, batch=64; 3 seed.
- **Ablasyon (en az biri, bütçeye göre):** (a) imgsz 224 vs 384 — küçük lekeler için çözünürlük etkisi; (b) sınıf dengesizliği
  varsa ağırlıklı örnekleme.
- **Değerlendirme (test, tek sefer, kendi kodumuz + sklearn):**
  doğruluk, sınıf bazında precision/recall/F1, macro & weighted F1, 7×7 karmaşıklık matrisi (ham + normalize),
  makro one-vs-rest ROC-AUC (ek), bootstrap %95 GA, n vs s McNemar testi, çıkarım süresi ve parametre sayısı.
- **"Dikkat" maddesi:** 7×7 matris **yaprak / meyve / çiçek** gruplarına indirgenerek 3×3 matris; gruplar arası karışma oranı;
  beklenen çiftlerin tartışması (Powdery Mildew Leaf↔Fruit, Angular Leafspot↔Leaf Spot, Anthracnose↔Gray Mold).
- **Hata analizi:** gerçek sınıfa en düşük olasılığı veren 10 yanlış tahmin → görsel panel (gerçek/tahmin/güven) + her biri için
  kısa yorum (arka plan, çoklu hastalık, erken evre, ışık vb. kategorize edilerek). İsteğe bağlı Grad-CAM.

## 6. 1. Rapor (18–23 Kasım) → **24 Kasım gönderim**
IEEE/makale düzeni, 4–6 sayfa: Özet · Giriş (problem, motivasyon) · İlgili çalışmalar (Afzaal 2021 vb.) · Veri ve ön işleme
(EDA tablosu, etiket kuralı, bölme, temizlik, Roboflow çapraz kontrolü) · Yöntem (YOLO-cls mimarisi kısaca, hiperparametre tablosu) · Deneyler ve sonuçlar
(ana tablo ort.±std, karmaşıklık matrisleri, ablasyon) · Hata analizi ve meyve/yaprak tartışması · Sınırlılıklar · Sonuç · Kaynakça.
Ek: Colab linki, ENV bilgisi. Şekiller vektörel/yüksek çözünürlüklü, tüm tablo/şekiller metinde atıflı.
Biçim: Word (.docx) — Claude, notebook çıktılarındaki sayı ve şekillerle doldurulmuş taslak .docx üretir; öğrenci okuyup
düzenler, Word'den PDF olarak kaydeder.

## 7. Aşama 3 — Segmentasyon: `03_segmentasyon.ipynb` (25 Kasım – 3 Aralık)  ·  %25
- **Modeller:** `yolo11n-seg`, `yolo11s-seg` (ana: s) ve RF-DETR segmentasyon (paketin güncel küçük/orta Seg modeli, ör. `RFDETRSegMedium`
  ya da Preview; sürüm raporlanır). Hepsi COCO ön-eğitimli, aynı bölme, 7 sınıf, örnek (instance) düzeyinde.
- **Eşleştirme:** benzer giriş çözünürlüğü (YOLO imgsz=640; RF-DETR kendi varsayılanı → farkı raporla), benzer epoch/süre bütçesi,
  val üzerinden en iyi checkpoint.
- **Ortak değerlendirme hattı (adalet için kritik):** her iki modelin test tahminleri COCO "results" JSON'una aktarılır →
  `pycocotools` ile **mask mAP50, mAP50-95** (sınıf bazında AP dahil); tahmin maskeleri sınıf bazında semantik haritaya
  birleştirilip **mIoU, Dice, sınıf bazında IoU** kendi `metrics.py` ile. Ultralytics'in kendi raporladığı değerler ek tablo olarak.
- **Verimlilik:** aynı T4'te batch=1, 20 ısınma + tüm test, ms/görüntü (ön/son işlem dahil ve hariç ayrı), parametre sayısı,
  `torch.cuda.max_memory_allocated` (GPU bellek, ek puan).
- **Hata analizi:** görüntü başına IoU'su en düşük 10 test görüntüsü → GT vs YOLO vs RF-DETR yan yana panel; hata türleri
  (kaçırılan küçük leke, yanlış sınıf, sınır taşması, birleşen örnekler). CLS'deki meyve/yaprak karışmasının SEG'de de sürüp
  sürmediği (sınıf karışma matrisi, IoU≥0.5 eşleşmeleriyle).
- **Bağlam:** Afzaal 2021'in Mask R-CNN sonuçlarıyla (aynı bölme kullanıldıysa) karşılaştırma notu.

## 8. 2. Rapor + Sunum (3–7 Aralık) → **8 Aralık gönderim**
Aynı akademik düzen; ana karşılaştırma tablosu (doğruluk metrikleri + hız + parametre + bellek), hız–doğruluk grafiği,
kalitatif panel, hata analizi, "hangi senaryoda hangi model" önerisi (sera mobil uygulaması bağlamı), sınırlılıklar.
10 dk sunum (~10–12 slayt): problem → veri → yöntem → CLS sonuçları → SEG karşılaştırması → hata analizi → sonuç.
Sunum provası için Claude olası hoca sorularıyla soru-cevap çalışması yapar.

---

## 9. Takvim özeti
| Tarih | İş | Kim |
|---|---|---|
| 7–9 Kas | Kurulum, Drive yapısı | Öğrenci (Claude destek) |
| 8–12 Kas | 01 notebook: EDA, denetim, dönüşüm + doğrulama | Claude yazar, öğrenci çalıştırır |
| 10–12 Kas | Roboflow yükleme + dışa aktarma, `compare_with_roboflow` karşılaştırması, ekran görüntüleri | Öğrenci yükler, Claude karşılaştırma kodunu yazar |
| 12–18 Kas | 02 notebook: n/s × 3 seed, ablasyon, metrikler, hata analizi | 〃 |
| 18–23 Kas | 1. rapor yazımı, öğrenci kontrolü | Claude taslak, öğrenci son hal |
| **24 Kas** | **1. rapor + Colab linki gönderimi** | Öğrenci |
| 25 Kas–3 Ara | 03 notebook: YOLO-seg, RF-DETR, ortak değerlendirme | Claude yazar, öğrenci çalıştırır |
| 3–7 Ara | 2. rapor + sunum + prova | Claude taslak, öğrenci |
| **8 Ara** | **2. rapor gönderimi** | Öğrenci |

Tampon: her aşamada 1–2 gün; Colab GPU kotası biterse seed sayısı 3→2'ye indirilir (raporda belirtilir).

## 10. Doğrulama (her aşamada nasıl kontrol edeceğiz)
- **Kod:** Claude, veri gelmeden önce sahte (sentetik) JSON/görüntülerle dönüşüm ve metrik fonksiyonlarını bu ortamda CPU'da test eder
  (ör. bilinen iki maske için IoU/Dice elle hesaplanan değere eşit mi; poligon→YOLO→maske geri dönüşümünde IoU≈1).
  Kaggle anahtarı eklenince `01_veri_hazirlama` notebook'u gerçek veriyle burada (CPU) baştan sona çalıştırılır;
  eğitim hücreleri burada yalnızca 1 epoch'luk küçük alt kümeyle duman testinden geçer, gerçek eğitim Colab'da.
- **Veri:** dönüşüm sonrası sayım tutarlılığı (görüntü/örnek sayıları her formatta aynı), görsel kontrol panelleri;
  Roboflow dışa aktarımıyla örnek bazında IoU karşılaştırması (bizim betik ↔ Roboflow).
- **Eğitim:** Colab'da önce 1 epoch'luk "duman testi", sonra tam eğitim; eğitim/val eğrileri aşırı öğrenme kontrolü.
- **Notebook tekrarlanabilirliği:** teslimden önce "Runtime → Restart and run all" ile baştan sona temiz çalıştırma.
- **Teslim kontrol listesi:** PDF'teki her sayı notebook çıktısında var mı; Colab linki gizli pencerede açılıyor mu;
  PDF'te zorunlu ölçütlerin hepsi (puanlama tablosu) işaretli mi; Roboflow ekran görüntüsü + karşılaştırma tablosu raporda mı.
