# İlerleme Günlüğü — Ödev 13

Bu dosya ödev boyunca yapılan her önemli işi, alınan kararı ve doğrulamayı tarih sırasıyla kaydeder.
Panelde **İlerleme Günlüğü** sayfasında gösterilir. Buradaki her sayı gerçek bir çalıştırma çıktısından alınmıştır
(kaynağı parantez içinde). Yeni kayıtlar en üste eklenir.

---

## 22 Kasım 2026 (öğleden sonra) — SEG eğitimleri bitti (9/9), sonuçlar içe alındı, panel gerçek veriyle denetlendi (#11, #13)

Kaynak: `runs/seg/*` (Colab'dan Drive üzerinden indirildi, `scripts/import_runs.py`; zip 633 MB, 231 dosya), `DONE` işaretleri (UTC → TR +3).

- **Son koşu** `ana_rfdetr-seg-small_im480_s2` 13:35'te erken durdurmayla bitti (toplam 46 epoch); değerlendirme hücreleri 13:47'de
  hatasız tamamlandı. Kullanıcı tarafında elektrik kesintisi oldu, Colab sunucudaki çalışma etkilenmedi.
- **Koşu özeti** (epoch / kütüphanenin en iyi val maske mAP50-95): YOLO26n s0/s1/s2 100 · 93 · 100 (0,731 · 0,748 · 0,745) ·
  YOLO26s 100 · 100 · 98 (0,760 · 0,767 · 0,748) · RF-DETR 40 · 48 · 46 (0,789 · 0,792 · 0,786). YOLO koşularının çoğu 100 epoch
  sınırına dayandı.
- **Çapraz kontrol (val maske mAP50-95, kütüphane ↔ bizim kod):** RF-DETR 0,789↔0,796 · 0,792↔0,800 · 0,786↔0,793;
  YOLO26n 0,731↔0,702 · 0,748↔0,701 · 0,745↔0,731; YOLO26s 0,760↔0,748 · 0,767↔0,752 · 0,748↔0,731. Fark RF-DETR'da ≤ 0,008,
  YOLO'da 0,01–0,05 (bizim kod hep biraz düşük) → sınıf eşlemesi doğru; YOLO farkı muhtemelen maske çözünürlüğü / iç
  değerlendirme ayrıntılarından (raporda not edilecek).
- **Test sonuçları (3 seed ort. ± std, tüm test 743; `runs/seg/ozet.csv`):**

| Model | mask mAP50 | mask mAP50-95 | mIoU | Dice | Param. | uçtan uca ms (T4, batch 1) | GPU bellek |
|---|---|---|---|---|---|---|---|
| YOLO26n-seg | 0,802 ± 0,020 | 0,600 ± 0,010 | 0,682 ± 0,020 | 0,805 ± 0,016 | 3,09 M | 18,1 | 39 MB |
| YOLO26s-seg | 0,829 ± 0,014 | 0,623 ± 0,010 | 0,710 ± 0,018 | 0,828 ± 0,013 | 11,44 M | 18,9 | 92 MB |
| RF-DETR Seg Small | 0,872 ± 0,002 | 0,691 ± 0,003 | 0,770 ± 0,012 | 0,869 ± 0,008 | 33,63 M | 60,0 | 246 MB |

  Temiz testte (692) skorlar aynı ya da biraz yüksek (ör. RF-DETR mAP50-95 0,696). Hız/bellek her modelin seed 0 koşusundan.
- **Meyve/yaprak karışması (SEG, IoU ≥ 0,5 eşleşen örnekler):** gruplar arası toplam YOLO26n %0,19, YOLO26s %0,28, RF-DETR %0,03 —
  CLS'dekinden (%2,1–2,4) çok düşük. Asıl hata türü sınıf karışması değil **kaçırılan lekeler** (ör. YOLO26n s0 358, RF-DETR s2 193 kaçırılan örnek).
- **Val ile seçilen:** `ana_yolo26s-seg_im480_s1` ve `ana_rfdetr-seg-small_im480_s1`. En kötü 10 test görüntüsünün hepsinde iki
  modelin de görüntü IoU'su 0 (çoğu Angular Leafspot / Leaf Spot ve Gray Mold) — yorum #12'de.
- **Ortam (SEG, 22 Kas `env.json`):** Python 3.13.16, torch 2.14.1+cu130, **ultralytics 8.4.216** (16 Kas'taki oturumlarda 8.4.215),
  rfdetr 1.12.1, supervision 0.32.0, pycocotools 2.0.11, T4. Colab her oturumda en güncel paketi kurduğundan sürümler koşular
  arasında küçükçe değişti → raporda sınırlılık notu.
- **Panel (#13):** Segmentasyon sayfası gerçek sonuçlarla açıldı — 9/9 koşu, eğrileri, ana sonuç grafiği, sınıf bazında, karışma
  matrisi, hız–doğruluk grafiği ve en kötü 10 (tahmin maskeleri `preds/*.json.gz`'den) hatasız.
- **Eksik:** Colab'ın çizdiği `rapor/figures/seg_*` şekilleri Drive'daki `rapor/figures`'ta kaldı (indirilen yalnız `runs/seg`).

---

## 18–22 Kasım 2026 — SEG eğitimleri 8/9, 1. rapor taslağı, hocanın Colab linki kararı (#11, #9, #7, #4)

Kaynak: Colab çıktısı ve Drive dosya/etkinlik saatleri (Claude in Chrome ile okundu), `runs/cls/*`, `data/derived/summary.json`.
Saatler Türkiye saati.

**Colab SEG eğitimleri (#11)**
| Tarih | Olan |
|---|---|
| 18 Kas | RF-DETR s1 bir epoch ilerledi, oturum ~13:50'de durdu; ardından ücretsiz GPU kotası doldu ("GPU arka ucuna bağlanılamıyor"). |
| 19 Kas | İlk `Tümünü çalıştır` Drive izni geç onaylandığı için `mount failed` verdi; yeniden çalıştırınca bağlandı. **RF-DETR s1 48. epoch'ta erken durdu** (kütüphanenin en iyi val maske mAP50-95'i 0,792, EMA; o oturumda 131,7 dk). **YOLO n s2** ~23:07'de bitti. |
| 20 Kas | **YOLO s s2** ~00:23'te bitti. RF-DETR s2 başladı, ~00:58'de (yaklaşık 8. epoch) kota yüzünden durdu. |
| 21 Kas | 19:10'da RF-DETR s2 kaldığı yerden sürdü; 21:17'de durdu (40 epoch tamam). Durma nedeni ekranda görülmedi (büyük ihtimalle kota). |
| 22 Kas | ~13:00'te yeniden başlatıldı (T4 hemen bağlandı). Kullanıcı tarafında elektrik kesildi ama Colab sunucudaki çalışma durmadı; 13:20'de epoch 42. |

- Colab ortamında 19 Kas'ta **supervision 0.32.0** kuruldu (16 Kas'ta 0.31.7) — ENV.md'ye SEG sürümleri yazılırken not edilecek.
- Seed sayısı kararı fiilen **3 seed** oldu (hepsi koştu).
- Not: Colab'ın "Bağlanıyor" göstergesi bazen yalnız görüntü hatası; asıl kontrol Drive'daki `metrics.csv` / `results.csv` saatleri.

**1. rapor taslağı (#9, #7)** — kullanıcı istedi (21 Kas), biçim kararları CLAUDE.md §3'te
- `scripts/build_rapor1.py` → `rapor/rapor1_cls.docx` + Word'den `rapor1_cls.pdf` (**4 sayfa**, iki sütun IEEE, Türkçe, 5 tablo,
  2 şekil, 12 kaynak, yapay zekâ beyanı, ek). Bütün sayılar betikte `runs/cls` ve `data/derived` dosyalarından okunur
  (elle sayı yok); Türkçe sayı ekleri (0,867'den, %46'sı) `ek()` ile üretilir.
- En kötü 10 tahmin (#7) dört türe ayrılarak yorumlandı: meyve hastalıklarının görsel benzerliği (8/10 meyve, 7'si Gray Mold
  tahmini), olgunlaşmamış meyvede antraknoz, yaprak lekeleri arası karışma, alışılmadık kadraj (tek gruplar arası hata).
- PDF düzeni Word çıktısından görüntüye çevrilerek denetlendi; tablo sütunları sabit genişliğe alındı, dar sütunu bozan uzun
  dosya adları okunur adlarla değiştirildi.

**Hocanın Colab linki (#4) — karar (kullanıcı, 22 Kas): teslim günü depo herkese açık yapılacak**
- Defterlerin kurulum hücresi artık gizli anahtar yoksa **misafir kipinde** çalışıyor: Drive'a dokunmaz, depoyu `/content/odev13`'e
  klonlar, veriyi Kaggle'dan anahtarsız indirir. Anahtarsız indirme yerelde denendi (534 MB, 30 sn). Sahip kipi (Secrets varken) değişmedi.
  01'in çıktıları korunarak yalnız ilgili hücreleri güncellendi; 02/03 betikle yeniden üretildi. Testler 18/18.
- Misafir kipi gerçek Colab'da henüz denenmedi (depo private) → public yapılınca başka hesapla gizli pencerede denenecek.
- Public öncesi git geçmişi gizli anahtar desenleri için tarandı: temiz. Depoda veri yok; yalnız rapor şekillerinde birkaç örnek görüntü var.

---

## 16–17 Kasım 2026 — Colab SEG eğitimleri: 9 koşudan 5'i bitti, oturum gece koptu (#11)

Kaynak: Colab çıktısı (Claude in Chrome ile okundu) ve Drive'daki dosya saatleri. Saatler Türkiye saati (Colab günlüğü UTC yazar).

| Koşu | Durum | Epoch | Süre (o oturum) | En iyi val maske mAP50-95 (kütüphanenin kendi ölçümü) |
|---|---|---|---|---|
| `ana_yolo26n-seg_im480_s0` | ✅ bitti | 100/100 (erken durmadı) | 64,5 dk | 0,7311 |
| `ana_yolo26s-seg_im480_s0` | ✅ bitti | 100/100 (erken durmadı) | 74,9 dk | 0,7603 |
| `ana_rfdetr-seg-small_im480_s0` | ✅ bitti (17 Kas 00:17) | 40'ta erken durdu (20 tur iyileşme yok) | 122,1 dk (16 Kas 22:15'te 12 Kas'ta yarıda kalan yerden devam etti) | 0,789 (EMA) |
| `ana_yolo26n-seg_im480_s1` | ✅ bitti | — | ~64 dk (00:17–01:21) | — |
| `ana_yolo26s-seg_im480_s1` | ✅ bitti | — | ~77 dk (01:21–02:38) | — |
| `ana_rfdetr-seg-small_im480_s1` | ⏸ yarıda | ~13 (son `last.ckpt` 03:32) | 02:38–03:34 | (ara değer, sayılmaz) |
| seed 2 (3 koşu) | bekliyor | | | |

- **Süreler:** bir seed'in üç modeli ≈ 4,5–5 saat (YOLO n ~65 dk, s ~75 dk, RF-DETR ~4,5 dk/epoch → erken durursa ~2–3 sa).
  Kalan iş ≈ 7–8 saat (RF-DETR s1'in kalanı + seed 2).
- **Oturum koptu:** Colab çalışma zamanı ~03:34'te bağlantısını kaybetti (oturum 16 Kas 19:02'de başlamıştı, ~8,5 sa).
  Sebep (boşta kalma mı, GPU kotası mı) görülemedi. Dosyalar sağlam; `Tümünü çalıştır` biten 5 koşuyu atlar, RF-DETR s1 `last.ckpt`'den sürer.
- **Colab defterinde hücre karışıklığı:** Kullanıcının Colab kopyasında eğitim hücresi (`for spec in RUNS: seg.train(...)`) önce
  silinmişti → `Tümünü çalıştır` eğitimi atlayıp değerlendirmeye geçti ve 8. bölümde (`select_by_val` → boş liste) durdu.
  Hücre yeniden eklendi ama önce defterin sonuna, sonra **en üste (kurulum hücresinden önce)** düştü. Bu oturumda çalıştı (kurulum
  zaten yapılmıştı) ama yeni oturumda `NameError` verir. **Çözüm:** yeni oturumda defteri GitHub'daki temiz hâliyle yeniden yüklemek
  (`Dosya → Not defteri yükle → notebooks/03_segmentasyon.ipynb`) ya da hücreyi "3 · Eğitim" başlığının altına taşımak.
- **Ara değerlendirme (sonuç değil):** Eğitimsiz koşulan `Tümünü çalıştır` YOLO s0 koşularını ortak kodla değerlendirdi. Çapraz kontrol
  val mAP50-95 kütüphane ↔ bizim: nano 0,7311 ↔ 0,7019, small 0,7603 ↔ 0,7476 — yakın (iki kod farklı enterpolasyon/maske çözünürlüğü
  kullanıyor; sınıf eşleme hatası olsaydı fark çok büyük olurdu). Drive'daki `runs/seg/results.csv` vb. bu ara hâli tutuyor;
  tüm koşular bitince değerlendirme hücreleri yeniden çalışıp üzerine yazacak. Test sayıları rapora **girmez**.
- **Not:** İki YOLO s0 koşusu da 100 epoch sınırına dayandı (erken durdurma tetiklenmedi) → eğitim eğrileri raporda yorumlanmalı
  (val mAP hâlâ hafif yükseliyor olabilir; bütçe eşitliği için sınır değiştirilmedi).
- **Açık karar:** 3 seed mi 2 seed mi — Claude 3 seed önerdi (8 Aralık'a zaman var, bir oturum daha yeter); kullanıcı henüz yanıtlamadı.
- Çalışma başka bilgisayara (MacBook) devredilmek üzere CLAUDE.md §0 ve §6a güncellendi.

## 12 Kasım 2026 (gece) — Colab SEG eğitimi başladı, panelin Segmentasyon sayfası yazıldı (#11, #13)

- **Colab (#11):** Drive'daki kod kopyası, git'te izlenmeyen 6 eski CLS şekli (`rapor/figures/cls_*`) yüzünden güncellenemiyordu
  (kullanıcı eski yüklenen defteri çalıştırdığı için yeni kurulum hücresi devreye girmedi). Kullanıcının onayıyla bu 6 dosya
  Google Drive web arayüzünden `odev13/_git_yedek/`'e **taşındı** (silinmedi; aynıları git'te). Sonra kod güncellendi ve eğitim başladı.
- İlk ölçüm (YOLO26n-seg, seed 0, T4): epoch ≈ 38 sn (eğitim ~34 + val ~4), GPU belleği 5–7 GB; val maske mAP50-95
  epoch 1'de 0,238 → epoch 8'de 0,537. Veri sayıları Colab'da da aynı (örnek 3.540 / 751 / 1.416, görüntü 1.450 / 307 / 743).
  Diğer modellerin süreleri henüz ölçülmedi.
- **Panel Segmentasyon sayfası (#13):** koşu durumu, eğitim eğrileri (yarım koşular da), ana tablo + grafik (mAP50-95, mAP50,
  mIoU, Dice; tüm/temiz test), çapraz kontrol tablosu, hastalık bazında AP/IoU/Dice, örnek karışma matrisi (kaçırılan/fazladan
  dahil) + yaprak/meyve/çiçek, hız–doğruluk grafiği, en kötü 10 (gerçek | YOLO | RF-DETR maskeleri, `preds/*.json.gz`'den).
  `ODEV13_SEG_DIR=runs/seg_duman` ile duman verisiyle tarayıcıda sınandı (kayıp grafiğindeki kesme işaretli alan adı hatası
  düzeltildi). `src/seg.py`'ye `semantic_from_results`, `parse_run` eklendi. Issue #10 ve #15 kapatıldı.

---

## 12 Kasım 2026 (akşam) — Mac mini kuruldu, 03 segmentasyon notebook'u yazıldı ve duman testinden geçti (#15, #10)

### Mac mini kurulumu (#15)
- Depo `~/Desktop/derin-ogrenme-odev13`'e klonlandı (CLAUDE.md §5'teki addan farklı). Python 3.13.18, `.venv`, Jupyter çekirdeği,
  Kaggle anahtarı (AirDrop). `prepare_data.py`: bölme 1.450 / 307 / 743, temiz test 692 — kayıtlarla aynı. CLS tahmin dosyalarındaki
  743 dosya adı yeni test listesiyle birebir eşleşti. Testler geçti, panel açıldı.
- Panel düzeltmesi: Sınıflandırma sayfası ağırlık (`best.pt`) Drive'da olduğu için "0/9 koşu bitti" diyordu → bitmişlik artık
  DONE işaretinden okunuyor ("9/9"); ağırlığın nerede olduğu ayrı sütunda.

### Kararlar (kullanıcı, 12 Kas)
- **Ortak giriş çözünürlüğü 480** (YOLO ve RF-DETR). Görüntüler 419×419; 480, iki paketin kuralına (YOLO: 32'nin katı,
  RF-DETR Seg Small: 24'ün katı) birlikte uyan ve gerçek boyuta en yakın değer. Plandaki "YOLO 640 / RF-DETR kendi varsayılanı" yerine.
- **RF-DETR Seg Small** (Medium değil): bütün RF-DETR Seg boyları aynı DINOv2 omurgasını kullanıyor (~33–35 M parametre);
  Small daha az katman/sorguyla ücretsiz T4'e daha uygun. 7 sınıf, 480'de 33,6 M parametre (YOLO26n-seg ~3,1 M, s ~11,4 M; kesin
  sayılar verimlilik tablosundan gelecek).
- Eşit bütçe: en çok 100 epoch, 20 epoch iyileşme yoksa erken durdurma, etkin batch 16 (RF-DETR 4 × 4 birikim), seed 0/1/2.
  Sıra seed'e göre (önce üç modelin seed 0'ı) → kota yetmezse yine tam bir karşılaştırma kalır. Seed sayısı ilk Colab süreleri görülünce kesinleşecek.

### RF-DETR 1.12.0 incelemesi (kaynak kodu + küçük denemelerle; tahminle değil)
- Roboflow COCO yükleyicisi kategori 1..7'yi 0..6'ya çeviriyor (üst kategori 0 "diseases" atlanıyor); tahmindeki `class_name`
  aynı sırada. Düşük eşikte bir de `__background__` (nesne yok) yuvası dönüyor → bizim kod atıyor.
- **Çapraz kontrol:** aynı küçük val alt kümesinde bizim kodla mask mAP50-95 = 0,132, kütüphanenin kendi değeri 0,130 → eşleme doğru.
  Notebook bu kontrolü tam val'de her koşu için tekrarlıyor.
- Kaldığı yerden devam (`last.ckpt`) çalışıyor; ama **ayar değişirse (ör. çözünürlük) yüklenemiyor** — denendi.
- Dosya boyutları: `last.ckpt` ~535 MB, her `.pth` ~134 MB; ara arşiv varsayılanı her 10 epoch'ta bir ~535 MB → kapatıldı
  (`checkpoint_interval=1000`). Koşu bitince yalnız `checkpoint_best_total.pth` + günlükler kalıyor (Drive'da yer için).

### Düzeltme: COCO örnek kimliği 0
- `pycocotools` COCOeval eşleşmeyi örnek kimliğiyle kaydedip **0'ı "eşleşmedi" sayıyor** → kimliği 0 olan gerçek örnek hiç doğru
  bulunamazdı (her bölmede 1 örnek). Testle yakalandı. `src/convert.py` artık kimlikleri 1'den başlatıyor (COCO kuralı);
  `coco_mask_map` de kendi içinde yeniden numaralıyor. Veri yeniden üretildi: geri-dönüş doğrulaması aynı (IoU < 0,99 yok), temiz test 692.
  CLS sonuçlarını etkilemez (CLS COCO dosyası kullanmıyor).

### Kod (#10)
- `src/metrics.py`: ortak SEG ölçütleri — RLE dönüşümü, piksel sınıf haritası (çakışmada yüksek skor kazanır), piksel karışma →
  sınıf IoU/Dice, mIoU, mDice, hastalık↔arka plan IoU; görüntü başına IoU; IoU ≥ 0,5 açgözlü örnek eşleştirme → 8×8 karışma
  (kaçırılan/fazladan dahil); pycocotools mask mAP50/50-95 + sınıf AP. `tests/test_seg_metrics.py`: elle hesaplanmış 5 test → toplam **18 test**.
- `src/seg.py`: eğitim (iki aile, kaldığı yerden devam, RF-DETR temizliği), tahmin → COCO results (`.json.gz`), val'de güven eşiği
  seçimi (piksel metrikleri için mIoU'yu en yükselten eşik; test bakılmaz), tüm test + temiz test değerlendirmesi, en kötü 10,
  verimlilik (parametre, yalnız ağ ve uçtan uca ms/görüntü, tepe GPU belleği).
- `scripts/build_notebook_03.py` → `notebooks/03_segmentasyon.ipynb`. **Yerel duman testi** (CPU, 1 epoch, küçük alt küme): bütün
  hücreler hatasız, ~50 sn. Çıktılar `runs/seg_duman/` (git dışı) — sayılar anlamsız, sonuç değildir.
- `.gitignore`: `runs/seg/` için küçük sonuç dosyaları git'te; ağırlık (`*.pt`, `*.pth`, `*.ckpt`) ve ham tahmin (`*.json.gz`) Drive'da.

### Colab'daki ilk deneme: kod güncellenemedi (düzeltildi)
- Kullanıcı 03'ü Colab'da çalıştırdı; kurulum hücresi `ImportError: cannot import name 'seg'` verdi. Sebep: Drive'daki kod kopyası
  güncellenememiş (`git pull` başarısız, kod sürümü 69a00da · 11.11). Drive'da Colab'ın 02'de çizdiği `rapor/figures/cls_*` şekilleri
  git'te izlenmiyordu; sonradan aynı adlarla git'e eklendikleri için git üzerlerine yazmayı reddetti. Eski kurulum hücresi hatayı
  yalnız uyarı olarak basıp eski kodla devam ediyordu.
- Düzeltme (`scripts/_nb_common.py`): çakışan dosyalar silinmeden `_git_yedek/<tarih>/`'e taşınıp pull yeniden denenir; pull yine
  başarısızsa hücre açık bir hatayla **durur** (eski kodla devam etmez). Git mesajları dil bağımsız okunsun diye `LC_ALL=C`.
  Sahte depo ile yerelde sınandı. Yalnız 03 yeniden üretildi (01/02'nin Colab çıktıları korunsun diye onlara dokunulmadı).

### Sıradaki
- #11: Colab'da 03 notebook'u (T4). İlk oturumdan sonra epoch süreleri ölçülüp seed sayısı ve takvim kesinleşecek.

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
