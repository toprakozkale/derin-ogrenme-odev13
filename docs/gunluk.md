# İlerleme Günlüğü — Ödev 13

Bu dosya ödev boyunca yapılan her önemli işi, alınan kararı ve doğrulamayı tarih sırasıyla kaydeder.
Panelde **İlerleme Günlüğü** sayfasında gösterilir. Buradaki her sayı gerçek bir çalıştırma çıktısından alınmıştır
(kaynağı parantez içinde). Yeni kayıtlar en üste eklenir.

---

## 12 Kasım 2026 (öğleden sonra) — Mac mini'ye devir hazırlığı

- Kullanıcı segmentasyon aşamasını **başka bilgisayardan (Mac mini)** yürütmeye karar verdi.
- CLS sonuçlarının küçük metin dosyaları (~2 MB: `results.csv`, `per_class.csv`, `preds/`, `train/*/results.csv` vb.) artık **git'te**;
  yeni bilgisayarda Drive'dan indirmeden panel ve şekiller çalışır. Ağırlıklar (`*.pt`) Drive'da kaldı.
  `select_by_val` ağırlık yerine DONE işaretiyle de çalışacak şekilde güncellendi.
- `CLAUDE.md`: §0 devir notu, yeni **§6a Aşama 3 planı** (hazır veri, yapılacaklar, karar bekleyenler), Colab'dan öğrenilenler,
  çalışma kuralları (raporlar en sona). `docs/PLAN.md` ve #10: SEG modelleri yolo11 → **YOLO26** (CLS ile tutarlı).

---

## 12 Kasım 2026 — Sınıflandırma eğitimleri bitti, sonuçlar panelde

### Yapılanlar
- **Colab'da 9 eğitimin hepsi hatasız bitti** (11 Kas, T4; toplam yaklaşık 2 saat). Hiçbiri 100 epoch'a ulaşmadı, hepsi erken durdurmayla durdu:

| Koşu | Epoch (durduğu / en iyi) | Süre |
|---|---|---|
| nano 224 — seed 0 / 1 / 2 | 49/29 · 59/39 · 60/40 | 0,190 · 0,221 · 0,227 sa |
| small 224 — seed 0 / 1 / 2 | 32/12 · 35/15 · 44/24 | 0,135 · 0,136 · 0,182 sa |
| nano 384 (ablasyon) — seed 0 / 1 / 2 | 36/16 · 46/26 · 45/25 | 0,251 · 0,315 · 0,305 sa |

- Sonuçlar Drive'dan indirilip `scripts/import_runs.py` ile `runs/cls/`'e alındı (171 dosya).
- **Yeni betik `scripts/make_cls_figures.py`:** notebook şekilleri Drive'daki `rapor/figures`'a yazdığı için, aynı şekiller
  kaydedilmiş test tahminlerinden yerelde yeniden çizildi (yeniden eğitim/tahmin yok). Betik önce **tutarlılık denetimi** yapar:
  9 koşu × 2 alt küme doğruluğu tahmin dosyalarından yeniden hesaplandı → `results.csv` ile birebir aynı; val ile seçilen
  koşu ve en kötü 10 listesi Colab'dakiyle aynı.
- Panelin Sınıflandırma sayfası gerçek sonuçlarla kontrol edildi (#8); karmaşıklık matrisi varsayılanı val ile seçilen model ailesi (nano) yapıldı.

### İlk sonuçlar (test; 3 seed ortalaması ± std — kaynak `runs/cls/results.csv`)
| Model | Tüm test (743) doğruluk | Temiz test (692) doğruluk | Tüm test macro F1 |
|---|---|---|---|
| YOLO26n-cls, 224 | 0,915 ± 0,015 | 0,910 ± 0,016 | 0,867 ± 0,036 |
| YOLO26s-cls, 224 | 0,904 ± 0,005 | 0,898 ± 0,005 | 0,855 ± 0,020 |
| YOLO26n-cls, 384 (ablasyon) | 0,934 ± 0,008 | 0,930 ± 0,009 | 0,898 ± 0,005 |

- Val ile seçilen koşu: `ana_yolo26n-cls_im224_s1` (val 0,990 — val sızıntısı nedeniyle iyimser). Bu koşunun testi 0,931 / temiz 0,928.
- Temiz test tüm testten yalnız ~0,5 puan düşük → test sızıntısının nota etkisi küçük.
- Small, nano'dan iyi değil: McNemar seed 0 ve 2'de p = 1,0; seed 1'de p = 0,003 (nano lehine). Parametre 5,45 M ↔ 1,54 M,
  T4'te model süresi 4,6 ↔ 3,9 ms/görüntü (batch 1; ön/son işlem dahil 10,8 ↔ 9,1 ms).
- En zor sınıflar meyve hastalıkları: **Powdery Mildew Fruit** (F1 nano 0,57; çoğu Gray Mold sanılıyor) ve **Anthracnose** (F1 0,77).
  Blossom Blight ve PM Leaf neredeyse kusursuz. 384 çözünürlük en çok PM Fruit'e yarıyor (F1 0,70).
- **Dikkat maddesi:** gruplar arası karışma küçük — nano %2,2, small %2,4; neredeyse tamamı **yaprak → meyve** (%3,3 / %3,7).
  Karışmaların çoğu grup **içinde** (meyve ↔ meyve: PM Fruit/Anthracnose → Gray Mold; yaprak ↔ yaprak: Leaf Spot → Angular Leafspot).
- **En kötü 10'da not:** `powdery_mildew_fruit137` ile `164` test içinde neredeyse aynı fotoğraf (pHash 0), `anthracnose_fruit_rot94`
  ile `100` aynı fotoğrafın kırpılmış hâli (pHash 16) → 10 hata aslında 8 farklı sahne. Raporda belirtilecek (#7).

---

## 11 Kasım 2026 — Aşama 2 başladı: sınıflandırma notebook'u, Colab doğrulaması

### Yapılanlar
- **Sınıflandırma kodu yazıldı** (`src/cls.py`, `src/metrics.py`, `notebooks/02_siniflandirma.ipynb`).
  - Model ailesi **YOLO26** seçildi: Ultralytics 8.4.210'daki en güncel aile (plan "o tarihteki güncel aile" diyordu).
    Modeller `yolo26n-cls` (1.540.071 parametre) ve `yolo26s-cls`.
  - Deney düzeni: imgsz 224, en fazla 100 epoch, 20 epoch iyileşme yoksa erken durdurma, batch 64, optimizer `auto` (AdamW),
    **seed 0/1/2** → 6 ana koşu. Ablasyon: nano model, imgsz **384**, 3 seed (küçük lekeler için çözünürlüğün etkisi).
  - Koşular kesilirse **kaldığı yerden devam** eder, biten koşular atlanır (Colab bağlantısı koparsa emek kaybolmaz).
  - Değerlendirme kendi kodumuzla: doğruluk, sınıf bazında P/R/F1, macro/weighted F1, ROC-AUC, bootstrap %95 GA,
    McNemar (n ↔ s), 7×7 karmaşıklık + **yaprak/meyve/çiçek 3×3** matrisi, ms/görüntü, parametre, en kötü 10 tahmin.
    Hepsi **tüm test (743)** ve **temiz test (692)** için ayrı ayrı. Model seçimi yalnız val ile.
- **Metrik testleri:** elle hesaplanmış değerlere karşı 5 yeni test → toplam **13 test geçiyor** (`pytest`).
- **Yerel duman testi:** 02 notebook'u Mac'te (CPU) 1 epoch ve verinin %5'iyle baştan sona hatasız çalıştı.
  Bu çıktılar `runs/cls_duman/` klasörüne yazılır ve **asıl sonuçlara, panele veya rapora karışmaz**.
- **Colab kurulumu:** notebook'ların ilk hücresi Drive'ı bağlar, proje kodunu GitHub'dan `MyDrive/odev13`'e klonlar/günceller
  (private depo için `GITHUB_TOKEN`, salt-okuma). Veri hız için Colab'ın yerel diskine (`/content/odev13_data`), kod ve sonuçlar Drive'a.
- **Colab Secrets** eklendi: `GITHUB_TOKEN`, `KAGGLE_USERNAME`, `KAGGLE_KEY`.
- **01 notebook'u Colab'da çalıştı (44 sn) ve sonuçlar yerelle birebir aynı:**

| Ölçü | Colab (11 Kas) | Yerel Mac (7 Kas) |
|---|---|---|
| Örnek (poligon) sayısı | 5.707 | 5.707 |
| En düşük IoU — YOLO-seg / COCO | 0,9947 / 0,9936 | 0,9947 / 0,9936 |
| IoU < 0,99 olan örnek | 0 | 0 |
| Sızan test / temiz test / sızan val | 51 / 692 / 108 | 51 / 692 / 108 |
| Bölme listeleri SHA-256 (train, val, test, test_clean) | 4/4 aynı | — |

### Karşılaşılan sorunlar ve çözümleri
- *"2 · Tüm hat" hücresi Colab'da çöktü:* veri proje klasörünün dışına alınınca `data.yaml` yolunu göreli yazan satır hata
  veriyordu → düzeltildi (`74e83d9`), Colab düzeni yerelde taklit edilerek doğrulandı.
- *Düzeltmeden sonra da aynı hata:* Colab oturumu eski kodu hafızada tutuyordu → kurulum hücresi artık eski `src` modüllerini
  temizliyor ve çalışan kod sürümünü yazdırıyor (`c39e408`). Kod değişince Colab'da **Oturumu yeniden başlat** yeterli.
- *`SecretNotFoundError: KAGGLE_USERNAME`:* gizli anahtarın adı yanlış yazılmıştı → doğru adla yeniden eklendi.
- *Notebook çekirdeği:* notebook'larda yerel çekirdek adı ("odev13") yazılıydı, Colab tanımıyordu → standart "Python 3" yapıldı.

### Karar: hesaplama ortamı
- Tüm eğitimler **ücretsiz Colab'da, T4 GPU** ile yapılacak. Colab Pro (9,99 $/ay) ve Kaggle Notebooks (2×T4, arka planda çalışma)
  değerlendirildi; zaman sıkıntısı olmadığı ve hocanın teslim şartı "Colab linki" olduğu için gerek görülmedi.
- Sonuç: tüm modellerin eğitimi ve hız ölçümü aynı GPU'da (T4) → hız karşılaştırması tutarlı. Segmentasyon eğitimleri birkaç güne
  yayılabilir; kota sorun olursa seed sayısı 3→2 (raporda belirtilir) ya da Kaggle yedek olarak düşünülür.

### Karar: sızıntı yaklaşımı onaylandı (issue #1)
- Kullanıcı "Yol 1"i onayladı: veri setinin **resmi bölmesi değiştirilmez** (ödev kuralı); test sonuçları hem **tüm test (743)** hem
  **temiz test (692)** üzerinde ayrı raporlanır, aradaki fark sızıntının etkisi olarak yorumlanır. Doğrulama kümesindeki sızıntı (%35)
  raporda sınırlılık olarak yazılır; eğitim sırasında görülen val doğruluğu (%98–99) iyimser kabul edilir ve sonuç olarak kullanılmaz.
- Neden "Yol 2" (benzerleri gruplayıp yeniden bölmek) değil: ödevin "resmi bölme varsa onu kullan" kuralını çiğner ve
  Afzaal ve ark. (2021) ile karşılaştırmayı bozar.

### Roboflow çapraz kontrolü (issue #2) — hocanın "Dikkat" maddesi
- Ham Kaggle verisi (bizim düzeltmelerimiz **uygulanmadan**, yalnız JSON içindeki base64 görüntü kopyası çıkarılarak;
  `scripts/prepare_roboflow_upload.py`) Roboflow'daki **private** projeye yüklendi: valid → train → test.
- Roboflow planı: Core (private veri, ayda 10 kredi, kart yok; kredi biterse ücret almaz, işlemi durdurur).
- **Health Check:** 2.459 görüntü, **0 eksik etiket**, 5.571 örnek, 7 sınıf, hepsi 419×419 (`rapor/figures/roboflow_health_check.png`).
- Sürüm v1: **ön işleme yok** (Auto-Orient ve Resize 432×432 kaldırıldı), çoğaltma yok → COCO Segmentation dışa aktarıldı.
  Roboflow'un kategori düzeni bizimkiyle aynı (0 = üst kategori, sınıflar 1–7 alfabetik).
- **Karşılaştırma** (`src/roboflow_check.py`, sonuç `docs/roboflow/`):

| Ölçü | Sonuç |
|---|---|
| Roboflow'daki görüntü | 2.459 (train 1.437 / valid 307 / test 715) |
| Roboflow'un atladığı | **41** — hepsinin aynı bölmede **bayt bayt aynı** eşi var (train 13, test 28) = bizim MD5 kopya listemizle birebir |
| Eşleşen görüntüde boyut / örnek sayısı / sınıflar aynı | 2.459 / 2.459 / 2.459 |
| Karşılaştırılan örnek | 5.571 |
| Ortalama IoU (bizim ↔ Roboflow) | **0,99995** |
| IoU < 0,99 | **1** — `test/gray_mold539` (0,979): bizim kendini kesen poligon düzeltmemizin uygulandığı örnek |
| Düzeltme yapılmayan örneklerde en düşük IoU | 0,9936 |

- **Düzeltme:** 7 Kasım kaydında "bayt düzeyinde aynı 41 dosyanın hepsi train içinde" yazılmıştı; doğrusu **train 13, test 28**
  (hep aynı bölme içinde). Roboflow karşılaştırması sırasında fark edildi; günlük, veri bulguları ve panel düzeltildi.
  Test içindeki 28 kopya sızıntı değildir ama test notunda iki kez sayılır → raporda belirtilecek.
- Roboflow'dan yanlışlıkla YOLOv8-**OBB** (döndürülmüş kutu) biçimi de indirildi; maske karşılaştırmasına uygun olmadığı için
  kullanılmadı. Poligon içeriği COCO karşılaştırmasıyla doğrulandığından ayrı YOLO karşılaştırmasına gerek görülmedi.

### Panel ve kayıt düzeni
- Panele **İlerleme Günlüğü** (bu dosya + git geçmişi) ve **Rapor Şekilleri** (`rapor/figures/` galerisi, PNG indirme) sayfaları eklendi.
- **Sınıflandırma sayfası** tamamlandı: koşuların durumu (bitti / yarım / başlamadı, en iyi val doğruluğu), ana tablo (ortalama ± std,
  tüm test + temiz test), doğruluk/macro F1 grafiği, hastalık bazında F1, 7×7 karmaşıklık ısı haritası + yaprak/meyve/çiçek tablosu,
  en sık karışmalar, McNemar, bootstrap, verimlilik, eğitim eğrileri ve en kötü 10 galerisi. Sonuç dosyaları `runs/cls/`'a gelince
  kendiliğinden dolar; geçici duman verisiyle denendi (o sayılar hiçbir yere kaydedilmedi).
- Colab'daki sonuçları yerele almak için `scripts/import_runs.py` (Drive'dan inen zip → `runs/`).
- Panel klasörü `app/pages` → `app/sayfalar` (Streamlit'in eski otomatik menüsü doğrudan bağlantılarda devreye giriyordu).
- `CLAUDE.md` yeni bilgisayarda devam edilebilecek şekilde yenilendi: "kaldığımız yer", panel sayfaları, Colab eğitim akışı.

### Kapanan işler
#5 (02 notebook'u + duman testi), #3 (01'i Colab'da çalıştırma), #2 (Roboflow çapraz kontrolü), #1 (sızıntı kararı).

---

## 7 Kasım 2026 — Aşama 0 (kurulum) ve Aşama 1 (veri hazırlama)

### Kurulum
- Hocanın görev PDF'i ve kullanıcının planı `docs/` altına alındı; Roboflow çapraz kontrolü plana **zorunlu** adım olarak eklendi.
- Kaggle API anahtarı (`~/.kaggle/kaggle.json`, izin 600), Colab T4 kontrolü (Tesla T4 15 GB, CUDA 13.0, Python 3.13.16),
  Mac'te Python 3.13 sanal ortamı, Roboflow hesabı (**Private** instance segmentation projesi).

### Veri incelemesi (`docs/veri_bulgulari.md`)
- Kaggle zip 559,7 MB, SHA-256 `e728dd56…de525`. Etiketler **LabelMe JSON**, tüm görüntüler **419×419**.
- Resmi bölme: **train 1.450 / val 307 / test 743** görüntü; 3.540 / 751 / 1.416 poligon.
- **Lisans:** Kaggle'da "Unknown"; yazarlar kullanım izni veriyor ve atıf istiyor. `Dataset.txt`'ye göre görüntüler 40'tan fazla
  web kaynağından (Bugwood, PaDIL, alamy.com, bloglar…) derlenmiş → tek sera değil, heterojen. (CC BY 4.0 makalenin lisansı.)
- **"Test Disease Severity Level" klasörü** test/'in birebir kopyası (743/743, bayt bayt) → kullanılmaz.
- **Sınıf dengesizliği:** eğitimde Leaf Spot 382, Anthracnose Fruit Rot 52 görüntü (~7×).

### Etiket denetimi ve düzeltme
- Kendini kesen poligon **12** → `make_valid` + en büyük parça; görüntü dışına taşan **4** → kırpma. Düzeltilen poligonların
  orijinalle en düşük örtüşmesi 0,979. Boş/bozuk etiket yok.
- **9 görüntüde iki hastalık birden** (yaprak lekesi + meyve küllemesi/antraknoz) → görüntü düzeyi etiket kuralı:
  *toplam poligon alanı en büyük sınıf*. Bu görüntüler "meyve/yaprak karışması" tartışmasının doğal örnekleri.
- `train/gray_mold1`: "Blossom Blight" etiketli ama görsel olarak kurşuni küf → muhtemel etiket hatası, **değiştirilmedi**.

### Sızıntı analizi
- Bayt düzeyinde aynı dosya: 41 çift — **train içinde 13, test içinde 28** (hep aynı bölmenin kendi içinde; bölmeler arası yok).
  *(Düzeltme, 11 Kas: ilk kayıtta yanlışlıkla "hepsi train içinde" yazılmıştı; Roboflow karşılaştırması sırasında fark edildi.)*
- pHash yakın-yinelenen eşiği görsel kalibrasyonla **12** (`docs/phash_calibration.csv`: ≤12'de incelenen 16/16 çift aynı sahne,
  14'te 6/16 yanlış eşleşme). Sonuç: **test 51/743 (%6,9)** ve **val 108/307 (%35,2)** görüntünün benzeri eğitimde.
  → **Temiz test kümesi: 692 görüntü.** Ölçülen sızıntı bir alt sınır.
- Karar ("Yol 1"): resmi bölme korunur, test metrikleri tüm test + temiz test için ayrı raporlanır (11 Kas'ta onaylandı, #1).

### Dönüşüm ve doğrulama
- Üç biçim: Ultralytics sınıflandırma klasörleri, YOLO-seg, COCO (Roboflow düzeni: 0 = üst kategori, sınıflar 1..7).
- 5.707 örneğin her biri YOLO ve COCO'dan geri okunup kaynakla karşılaştırıldı: **en düşük IoU 0,9947 / 0,9936, 0,99 altı 0**,
  sınıf ve sayı tutarlı; COCO dosyaları pycocotools ile yüklendi.

### Altyapı
- Streamlit takip paneli (`app/`), sentetik verili 8 test, `01_veri_hazirlama.ipynb` (yerelde hatasız, 4 rapor şekli).
- Özel GitHub deposu, `CLAUDE.md` (proje hafızası + kurulum), 15 issue ve 2 kilometre taşı (24 Kas / 8 Ara).
