# İlerleme Günlüğü — Ödev 13

Bu dosya ödev boyunca yapılan her önemli işi, alınan kararı ve doğrulamayı tarih sırasıyla kaydeder.
Panelde **İlerleme Günlüğü** sayfasında gösterilir. Buradaki her sayı gerçek bir çalıştırma çıktısından alınmıştır
(kaynağı parantez içinde). Yeni kayıtlar en üste eklenir.

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
#5 (02 notebook'u + duman testi), #3 (01'i Colab'da çalıştırma).

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
- Bayt düzeyinde aynı dosya: 41 grup, hepsi train içinde (bölmeler arası yok).
- pHash yakın-yinelenen eşiği görsel kalibrasyonla **12** (`docs/phash_calibration.csv`: ≤12'de incelenen 16/16 çift aynı sahne,
  14'te 6/16 yanlış eşleşme). Sonuç: **test 51/743 (%6,9)** ve **val 108/307 (%35,2)** görüntünün benzeri eğitimde.
  → **Temiz test kümesi: 692 görüntü.** Ölçülen sızıntı bir alt sınır.
- Karar ("Yol 1"): resmi bölme korunur, test metrikleri tüm test + temiz test için ayrı raporlanır (onay bekliyor: #1).

### Dönüşüm ve doğrulama
- Üç biçim: Ultralytics sınıflandırma klasörleri, YOLO-seg, COCO (Roboflow düzeni: 0 = üst kategori, sınıflar 1..7).
- 5.707 örneğin her biri YOLO ve COCO'dan geri okunup kaynakla karşılaştırıldı: **en düşük IoU 0,9947 / 0,9936, 0,99 altı 0**,
  sınıf ve sayı tutarlı; COCO dosyaları pycocotools ile yüklendi.

### Altyapı
- Streamlit takip paneli (`app/`), sentetik verili 8 test, `01_veri_hazirlama.ipynb` (yerelde hatasız, 4 rapor şekli).
- Özel GitHub deposu, `CLAUDE.md` (proje hafızası + kurulum), 15 issue ve 2 kilometre taşı (24 Kas / 8 Ara).
