# CLAUDE.md — Ödev 13: Çilek Hastalığı (CLS + SEG)

Bu dosya Claude'un her oturumda otomatik okuduğu proje hafızasıdır. Claude'un makineye özel hafızası (`~/.claude/...`)
başka bilgisayara taşınmaz; **bu proje hakkında bilinmesi gereken her şey burada** ve `docs/` altındadır
(ayrıntılı tarihçe: `docs/gunluk.md`). Her önemli karar ya da aşama sonunda güncellenir.

---

## 0. Kaldığımız yer — yeni oturumda önce bunu oku
**Son güncelleme: 22 Kasım 2026 (öğle).** Teslimler: 1. rapor (CLS) **24 Kasım 23:59** · 2. rapor (SEG) **8 Aralık 23:59**.

> **Colab SEG eğitimleri (#11): 9 koşudan 8'i bitti**, son koşu `ana_rfdetr-seg-small_im480_s2` 22 Kas ~13:00'te 40. epoch'tan
> sürdürüldü (13:20'de epoch 42; en iyi EMA dün ~epoch 24 → erken durdurma yakın). Colab Claude'un Chrome sekme grubundaki sekmede
> çalışıyor — **kapatma**. Bitince değerlendirme hücreleri kendiliğinden çalışır → §6a adım 5–7 (çapraz kontrol, Drive'dan `runs/seg`
> indir, `import_runs.py`) → #12 → 2. rapor. Seed sayısı fiilen **3** (hepsi koştu). Ayrıntı: `docs/gunluk.md` (18–22 Kas).
> **1. rapor taslağı hazır** (aşağıda); kullanıcının okuması bekleniyor. Teslim günü depo public yapılacak (#4).

- 📝 **1. rapor taslağı (21 Kas akşamı):** `scripts/build_rapor1.py` → `rapor/rapor1_cls.docx` (Türkçe, iki sütun IEEE, A4;
  bütün sayılar betikte `runs/cls` + `data/derived` dosyalarından okunur, Türkçe sayı ekleri `ek()` ile). Metni değiştirmek için
  betiği düzenle ve yeniden çalıştır (Word'de elle düzeltilen sürüm betikle ezilir — kullanıcı elle düzeltmeye başlarsa betiği bırak).
  Ekte Colab (02, 01) ve depo linkleri var (yer tutucu kalmadı). PDF (22 Kas): Word ile 4 sayfa, düzen denetlendi.
  **Word'le PDF alma:** önce eski PDF'i sil (yoksa "üzerine yaz?" penceresinde takılır), dosyayı `open -a "Microsoft Word" x.docx`
  ile aç (AppleScript `open` dosya erişim izni penceresinde takılıyor), sonra AppleScript `save as active document file name … file format format PDF`.
- ✅ Aşama 0 (kurulum) ve Aşama 1 (veri hazırlama) **tamamen** bitti; 01 notebook'u hem Mac'te hem Colab'da çalıştı, sonuçlar birebir aynı.
- ✅ Roboflow çapraz kontrolü bitti (#2): 5.571 örnek, ort. IoU 0,99995; Roboflow'un atladığı 41 görüntü = bölme içi bayt kopyaları (train 13, test 28).
- ✅ Sızıntı yaklaşımı **onaylandı** (11 Kas, #1): resmi bölme + tüm test ve temiz test ayrı raporlanır.
- ✅ **Aşama 2 (CLS) eğitimleri bitti (#6, #8):** 9 koşu Colab T4'te ~2 saatte hatasız (11 Kas). Sonuçların küçük dosyaları
  **git'te** (`runs/cls/*.csv`, `preds/`, `train/*/results.csv`) → yeni Mac'te `git pull` yeter, panel dolu açılır.
  Ağırlıklar (`*.pt`, ~120 MB) yalnız Drive'da (`MyDrive/odev13/runs/cls`). Şekiller `rapor/figures/cls_*` (git'te).
  Test doğruluğu (3 seed ort. ± std): nano 0,915 ± 0,015 · small 0,904 ± 0,005 · nano-384 0,934 ± 0,008; temiz test ~0,5 puan düşük.
  Val ile seçilen: `ana_yolo26n-cls_im224_s1`. Gruplar arası (yaprak/meyve/çiçek) karışma ~%2. Ayrıntı: `docs/gunluk.md` (12 Kas).
- ✅ **Aşama 3 hazırlığı (12 Kas, #10):** kararlar ortak çözünürlük **480** + **RF-DETR Seg Small** (kullanıcı); `src/seg.py`,
  ortak SEG metrikleri (`src/metrics.py`, 18 test), `notebooks/03_segmentasyon.ipynb`; duman testi hatasız (~50 sn, `runs/seg_duman/`).
  RF-DETR sınıf eşlemesi çapraz kontrolle doğrulandı. COCO örnek kimliği 0 hatası düzeltildi (ayrıntı `docs/gunluk.md`, 12 Kas akşam).
- 🔄 **#11 Colab SEG eğitimleri (16–22 Kas):** 8/9 bitti (RF-DETR s2 sürüyor). Ölçülen süreler: YOLO n ~65 dk, s ~75 dk
  (s0'lar 100 epoch sınırına dayandı), RF-DETR ~4,5 dk/epoch (s0 epoch 40'ta, s1 48'de erken durdu). Kütüphanenin val maske
  mAP50-95'i (son sonuç değil): n 0,731 · s 0,760 · RF-DETR 0,789 (s0) / 0,792 (s1). Çapraz kontrol (s0): n 0,702 ↔ 0,731, s 0,748 ↔ 0,760.
  Ücretsiz Colab GPU kotası son günlerde ~2–2,5 saatte bitiyor ("GPU arka ucuna bağlanılamıyor"); kalan yerden sürdürmek yetiyor.
  Kullanıcı tarafında elektrik kesildiğinde (22 Kas) Colab sunucudaki çalışma durmadı.
- ✅ #13 panel Segmentasyon sayfası yazıldı (duman verisiyle sınandı); gerçek sonuçlar içe alınınca kontrol edilecek.
- ⏭️ **Sıradaki iş: #11'i bitir → sonuçları içe al (§6a adım 6) → #12 karşılaştırma/hata analizi → #13 gerçek veriyle kontrol.**
- 📝 #7 en kötü 10 yorumu 1. rapora yazıldı (dört hata türü). Raporlara 21 Kas'ta kullanıcı isteğiyle başlandı; sunum henüz yok.
- ✅ #4 kararı (22 Kas): teslim günü depo public; defterler gizli anahtarsız (misafir kipi) çalışacak şekilde güncellendi.

## 1. Ödev özeti
- Yüksek lisans Derin Öğrenme dersi; vize+final yerine geçen ödev. Kullanıcıya **Görev 13 — Çilek bitkisinde hastalık** atandı.
- Hocanın görev metni: `docs/Derin_Ogrenme_Odev_Gorevleri.pdf` · Ayrıntılı plan: `docs/PLAN.md`
- **Veri:** Kaggle `usmanafzaal/strawberry-disease-detection-dataset` (Afzaal ve ark., *Sensors* 2021). 2.500 görüntü (419×419),
  7 hastalık sınıfı, LabelMe JSON poligon etiketleri. Resmi bölme: train 1.450 / val 307 / test 743.
- **CLS:** yalnız Ultralytics YOLO-cls, en az iki boyut (n, s). **SEG:** YOLO-seg ↔ RF-DETR, aynı bölme ve aynı test.
- **Hocanın "Dikkat" maddesi:** JSON poligonlar YOLO-seg ve COCO'ya çevrilecek (Roboflow anılıyor);
  **meyve/yaprak hastalıklarının karışma oranı karmaşıklık matrisiyle** incelenecek.
- **Teslimler** (PDF rapor 4–6 sayfa + çalıştırılabilir Colab linki + 10 dk sunum → tekinler.cs@gmail.com):
  - 1. rapor (CLS): **24 Kasım 2026 23:59**
  - 2. rapor (SEG): **8 Aralık 2026 23:59**
- **Puanlama:** veri hazırlama %20 · CLS %20 · SEG karşılaştırma %25 · hata analizi + Dikkat %20 · rapor/sunum %15.
- **Zorunlu ölçütler:** CLS → doğruluk, sınıf bazında P/R/F1, karmaşıklık matrisi. SEG → mIoU, Dice, mask mAP50 ve mAP50-95
  (+ sınıf bazında IoU). Karşılaştırma → aynı testte skor, ms/görüntü, parametre sayısı (+ GPU bellek). En kötü 10 tahmin + yorum.

## 2. Çalışma kuralları (kullanıcıyla anlaşılmış)
- **Kullanıcı derin öğrenme konusunda uzman değil.** Her şeyi sade Türkçeyle, adım adım ve benzetmelerle anlat; teknik terimi
  ilk geçtiği yerde açıkla. Tarayıcıda/Colab'da ne yapacağını menü adlarıyla (Colab arayüzü Türkçe) tarif et.
- **İş bölümü:** Claude kodu, notebook'ları, rapor (.docx → PDF) ve sunum taslaklarını yazar; kullanıcı eğitimleri Colab'da (T4) çalıştırır.
- **Hiçbir sonuç uydurulmaz veya tahmin edilmez.** Rapordaki her sayı gerçek bir çalıştırma çıktısından gelir; yoksa yer tutucu bırakılır.
  Duman testi sayıları asla sonuç gibi sunulmaz.
- Tekrarlanabilirlik: sabit seed (bölme 42, eğitim 0/1/2), sürümler `ENV.md` + `requirements-local.txt`, bölme listeleri SHA-256 ile.
- Test kümesi yalnız en sonda bir kez kullanılır; model seçimi val ile.
- Adil karşılaştırma: YOLO-seg ve RF-DETR aynı bölme, benzer çözünürlük/bütçe, aynı T4, **aynı değerlendirme kodu**.
- Kaggle anahtarı, şifreler, token'lar asla koda veya git'e girmez; sohbette görülürse tekrar edilmez ve kullanılmaz.
- **Dal açılmaz, doğrudan `main`'de çalışılır** (kullanıcı tek çalışıyor, GitHub'ı yedek için kullanıyor; Colab da `main`'i çeker).
- **~1 dakikadan uzun komutlardan önce** (büyük kurulum, indirme) ne yapacağını ve yaklaşık süresini söyle; çıktıyı gizleyen sessiz
  kurulum yapma (12 Kas'ta sessiz `pip install` donmuş sanıldı).
- Colab'ı kontrol etmek için önce Drive'daki koşu klasörlerine bak (`runs/seg/train/<koşu>/`: `last.ckpt`, `metrics.csv`,
  `results.csv`, `DONE` değiştirilme saatleri) — notebook'u ikinci sekmede açmaktan daha güvenli; gerekirse notebook'u aç, oku, hemen kapat.
- Kullanıcı bir aşama bitince "GitHub'a gönder" der; commit + push sonrası `docs/gunluk.md`, `docs/durum.yaml` ve bu dosya güncel olmalı.
  Aşama sonunda kullanıcı "GitHub'a ve panele detaylı kaydet" ister: günlük kaydı + durum + panel + şekiller + bu dosya.
- **Raporlar ve sunum en sona** (kullanıcı kararı, 11 Kas). Kullanıcı istemeden rapor taslağına başlama.
- Kullanıcıya bir şeyi "yaptım" demeden önce doğrula; yanlış bir iddia fark edilirse açıkça düzelt ve günlüğe düzeltme notu düş
  (örnek: 41 bayt kopyası önce yanlışlıkla "bölmeler arası" sanılmıştı).
- Colab'daki çıktıları okumak gerekirse: **Claude in Chrome** eklentisi (MacBook'ta ve Mac mini'de kurulu; yeni makinede §5 adım 12).
  Colab notebook'unu yeni sekmede açıp `get_page_text` / JS ile okumak mümkün. **Okuduktan sonra sekmeyi hemen kapat** — aynı
  notebook iki sekmede açıkken Colab'ın otomatik kaydı çakışma uyarısı veriyor. Kullanıcının sekmesindeki hücrelere dokunma.

## 3. Alınmış kararlar
| Konu | Karar |
|---|---|
| Bölme | Resmi bölme korunur (ödev kuralı). **Sızıntı:** test metrikleri hem tüm test (743) hem **temiz test** (692) üzerinde raporlanır ("Yol 1"). ✅ **Kullanıcı onayladı (11 Kas 2026, #1).** Val sızıntısı (%35) sınırlılık olarak yazılır; val skorları iyimser kabul edilir. |
| Yakın-yinelenen eşiği | pHash Hamming ≤ **12** (görsel kalibrasyon: `docs/phash_calibration.csv`). Ölçülen sızıntı bir alt sınır. Test %6,9, val %35,2. |
| Görüntü düzeyi etiket | Tek sınıf → o sınıf; çok sınıflı → toplam poligon alanı en büyük sınıf (`src/config.py: IMAGE_LABEL_RULE`). 9 çok sınıflı görüntü. |
| Poligon düzeltme | Kendini kesen → `shapely.make_valid` + en büyük parça (12 adet); görüntü dışı → kırpma (4 adet); <3 nokta/sıfır alan → at (0). |
| `train/gray_mold1` | "Blossom Blight" etiketli ama görsel olarak kurşuni küf → muhtemel etiket hatası; **değiştirilmedi**, raporda not. |
| "Test Disease Severity Level" klasörü | test/'in birebir kopyası (743/743) → kullanılmaz. |
| Bölme içi bayt kopyaları | 41 çift: **train 13, test 28** (bölmeler arası yok). Resmi bölme korunur; test içindeki 28 kopya test notunda iki kez sayılır → raporda not. (İlk kayıtta yanlışlıkla "hepsi train" yazılmıştı, düzeltildi.) |
| COCO kategori kimliği | Roboflow düzeni: 0 = üst kategori "diseases", sınıflar 1..7 (= YOLO kimliği + 1). |
| Sınıf sırası | Alfabetik, `src/config.py: CLASSES`. Gruplar: yaprak (Angular Leafspot, Leaf Spot, PM Leaf), meyve (Anthracnose, Gray Mold, PM Fruit), çiçek (Blossom Blight). |
| Lisans | Kaggle'da "Unknown"; yazarlar kullanım izni veriyor, atıf istiyor. Görüntüler çok sayıda web kaynağından (alamy.com dahil) → **Roboflow projesi private**. CC BY 4.0 makalenin lisansı, verinin değil. |
| Roboflow | Ana dönüşüm Python; Roboflow bağımsız çapraz kontrol — **yapıldı** (11 Kas). Workspace `topraks-workspace-k39ij`, proje `strawberry-disease-segmentation-20mib` (Private, Core plan), sürüm v1 = ön işlemesiz COCO. Sonuç `docs/roboflow/`; ham dışa aktarım `data/roboflow_export/` (git'te yok). |
| Model ailesi | **YOLO26** (Ultralytics 8.4.210'daki en güncel aile; plan "o tarihteki güncel aile" diyordu): CLS `yolo26n-cls` (1,54 M parametre), `yolo26s-cls` (5,45 M); SEG için `yolo26n-seg`, `yolo26s-seg` (PLAN.md ve #10 eskiden yolo11 diyordu → YOLO26'ya güncellendi, CLS ile tutarlı). |
| CLS deney düzeni | imgsz 224, 100 epoch + patience 20, batch 64, optimizer auto, seed 0/1/2 → 6 ana koşu; ablasyon: nano imgsz 384 × 3 seed. Model seçimi val doğruluğuyla; en kötü 10 tahmin val ile seçilen koşudan. Karmaşıklık matrisi 3 seed toplamı. |
| CLS sonuçları (12 Kas) | Nano small'dan kötü değil (McNemar: 2 seed'de p = 1,0, 1 seed'de nano lehine p = 0,003); çözünürlük 384 en çok PM Fruit'e yarıyor. En zor: PM Fruit (≈%46'sı Gray Mold sanılıyor), Anthracnose. En kötü 10'da 2 test içi yakın kopya çifti (`powdery_mildew_fruit137/164` pHash 0, `anthracnose_fruit_rot94/100` pHash 16) → 8 farklı sahne. |
| Duman testi | Yerelde GPU yokken `SMOKE` modu (1 epoch, verinin %5'i) → çıktılar `runs/cls_duman/`; asıl sonuçlara/panele/rapora **asla** karışmaz. SEG için de aynı düzen (`runs/seg_duman/`). |
| Hesaplama ortamı (11 Kas kararı) | **Tüm eğitimler ücretsiz Colab'da, T4 GPU** — Colab Pro alınmadı (kullanıcının zaman sıkıntısı yok), Kaggle Notebooks kullanılmıyor (hocanın PDF'i teslimde yalnız "Colab linki" istiyor; Kaggle'dan söz etmiyor). Kaggle yalnız Colab kotası uzun süre engellerse yedek. Segmentasyon gerekirse 3→2 seed. |
| Colab düzeni | Kod + sonuçlar Drive'da (`MyDrive/odev13`), veri Colab'ın yerel diskinde (`/content/odev13_data`, `ODEV13_DATA`). Ücretsiz plan; sekme açık kalmalı. Modeller **T4**'te koşulur (adil hız ölçümü). |
| SEG deney düzeni (12 Kas, kullanıcı) | **Ortak çözünürlük 480** (YOLO 32'nin, RF-DETR Seg Small 24'ün katı ister; 419'a en yakın ortak değer). Modeller `yolo26n-seg`, `yolo26s-seg`, **`RFDETRSegSmall`** (rfdetr 1.12.0; 7 sınıfta 480'de 33,6 M parametre — tüm RF-DETR Seg boyları aynı DINOv2 omurgası, ~33–35 M). 100 epoch + erken durdurma 20, etkin batch 16 (RF-DETR 4×4), seed 0/1/2, sıra seed'e göre. Ayarlar `src/seg.py` başında. |
| SEG değerlendirme (12 Kas) | Tahminler COCO results'a (RLE) → mask mAP (pycocotools, `CONF_MIN` 0,001, en çok 100 tahmin). Piksel metrikleri (mIoU, Dice) için güven eşiği **val'de mIoU'yu en yükselten** eşik (model başına; `esikler.csv`). mIoU/mDice = 7 hastalık sınıfı ortalaması (arka plan hariç). Örnek karışma: sınıftan bağımsız açgözlü IoU ≥ 0,5 eşleşme. En kötü 10: val ile seçilen en iyi YOLO + en iyi RF-DETR'ın görüntü IoU ortalaması. |
| RF-DETR ayrıntıları (12 Kas) | Roboflow yükleyicisi kategori 1..7 → 0..6; `__background__` yuvası atılır. Bizim val mAP ↔ kütüphanenin val mAP'si çapraz kontrolü notebook'ta (yakın olmalı). Kaldığı yerden devam ayar değişmezse çalışır. Ara arşiv kapalı (`checkpoint_interval=1000`); bitince yalnız `checkpoint_best_total.pth` kalır (`last.ckpt` ~535 MB silinir). |
| COCO örnek kimliği (12 Kas) | 1'den başlar (`convert.py`). pycocotools COCOeval kimliği 0 olan örneği hiç eşleştiremiyor; `coco_mask_map` ayrıca kendi içinde yeniden numaralar. |
| Sonuçlar git'te mi? (12 Kas) | Eğitim sonuçlarının **küçük metin dosyaları git'te** (`.gitignore` istisnaları: `runs/cls/*.csv|json`, `preds/*.csv`, `train/*/{results.csv,args.yaml,DONE}`), ağırlık ve eğitim görselleri yalnız Drive'da. İki Mac arasında Drive indirmesi gerekmesin diye. SEG için aynı istisnalar `runs/seg/` için eklendi (+ RF-DETR `metrics.csv`, `training_config.json`; `*.pth`, `*.ckpt`, `*.json.gz` git dışı). |
| Hocanın Colab linki (#4, 22 Kas, kullanıcı) | **Depo teslim günü herkese açık (public) yapılacak.** Rapordaki linkler: `https://colab.research.google.com/github/toprakozkale/derin-ogrenme-odev13/blob/main/notebooks/02_siniflandirma.ipynb` (+ 01) ve depo adresi. Defterlerin kurulum hücresi gizli anahtar yoksa **misafir kipinde** çalışır (Drive'a dokunmaz, `/content/odev13`'e klonlar, Kaggle verisi anahtarsız iner — 22 Kas'ta yerelde anahtarsız indirme denendi: 534 MB, 30 sn). Public yapmadan önce git geçmişi gizli anahtar için tarandı (temiz). Public yapınca misafir kipini gizli pencerede başka hesapla dene. |
| 1. rapor biçimi (21 Kas, kullanıcı) | **Türkçe**, **iki sütun IEEE düzeni**, Word (.docx) → PDF, 4–6 sayfa; sonunda kısa **"Yapay zekâ araçlarının kullanımı" beyanı**. Kimlik (kullanıcı verdi, 21 Kas): Toprak Özkale · öğrenci no **250401006** · **Iğdır Üniversitesi, Lisansüstü Eğitim Enstitüsü** · ders **"Derin Öğrenme-Sinir Ağları"** · hoca **Gültekin Işık** (unvanı bilinmiyor → raporda unvansız; kullanıcı ekleyebilir, tahmin etme). Bölümler: Özet · Giriş · Veri ve ön işleme · Yöntem · Sonuçlar · Hata analizi + Dikkat · Sınırlılıklar/Sonuç · Kaynakça · Ek. |

## 4. Depo yapısı
```
src/            config.py (yollar, sınıflar, eşikler; ODEV13_ROOT / ODEV13_DATA ile değiştirilebilir)
                dataset.py (LabelMe okuma/düzeltme/istatistik) · convert.py (CLS / YOLO-seg / COCO + geri-dönüş doğrulaması)
                dedup.py (MD5/pHash, sızıntı) · pipeline.py (Aşama 1 hattı + indirme) · roboflow_check.py (Roboflow COCO ↔ bizim dönüşüm)
                metrics.py (P/R/F1, karmaşıklık + yaprak/meyve/çiçek grupları, bootstrap GA, McNemar;
                            SEG: RLE, piksel IoU/Dice/mIoU, örnek eşleştirme + karışma, pycocotools mask mAP)
                cls.py (Aşama 2: eğitim — kaldığı yerden devam —, tahmin, değerlendirme, verimlilik, en kötü 10)
                seg.py (Aşama 3: YOLO26-seg + RF-DETR eğitimi, COCO results tahmini, val'de eşik, ortak değerlendirme,
                        en kötü 10, verimlilik; çıktılar runs/seg, duman testi runs/seg_duman)
scripts/        prepare_data.py (Aşama 1'i koşar) · build_notebook_01.py / _02.py / _03.py (notebook'ları üretir)
                _nb_common.py (notebook'ların ortak Colab kurulum hücresi) · import_runs.py (Drive'dan inen sonuç zip'ini runs/'a alır)
                make_cls_figures.py (runs/cls'ten rapor şekilleri + tutarlılık denetimi)
                prepare_roboflow_upload.py (ham veriden data/roboflow_upload/{train,valid,test})
notebooks/      01_veri_hazirlama.ipynb (çıktılı; Mac + Colab'da çalıştı) · 02_siniflandirma.ipynb (Colab'da çalıştı, 11 Kas)
                03_segmentasyon.ipynb (12 Kas; yerel duman testi geçti, Colab'da çalıştırılacak — #11)
app/            Streamlit paneli: Ana_Sayfa.py + sayfalar/ (klasör adı bilerek "pages" değil — Streamlit'in eski otomatik menüsü devreye girmesin)
tests/          pytest: veri hattı (8) + CLS metrikleri (5) + SEG metrikleri (5) = 18 test
docs/           PLAN.md · gunluk.md (ilerleme günlüğü, panelde gösterilir) · veri_bulgulari.md · durum.yaml (panelin ilerleme kaynağı)
                phash_calibration.csv · roboflow/ (çapraz kontrol özeti, atlananlar, en düşük IoU'lar) · hocanın PDF'i
rapor/figures/  rapora girecek şekiller (notebook'lar üretir; panelde "Rapor Şekilleri")
data/           GIT'TE YOK — raw/, processed/ (cls, yoloseg, coco), derived/ (özet, CSV'ler) → scripts/prepare_data.py yeniden üretir
runs/           Eğitim çıktıları (Colab → Drive → scripts/import_runs.py → buraya). Yalnız küçük sonuç dosyaları git'te, *.pt Drive'da
weights/        GIT'TE YOK — ön-eğitimli ağırlıklar (otomatik iner)
```
- Notebook'lar elle değil `scripts/build_notebook_*.py` ile üretilir (01'in çıktıları korunmak isteniyorsa yeniden üretip yerelde çalıştır).
- Notebook çekirdeği `python3` olmalı (Colab "odev13" adını tanımaz); yerelde nbconvert'e `--ExecutePreprocessor.kernel_name=odev13` verilir.

### Panel sayfaları (`.venv/bin/streamlit run app/Ana_Sayfa.py` → http://localhost:8501)
| Sayfa | Kaynak |
|---|---|
| Genel Bakış | `docs/durum.yaml` + `data/derived/summary.json` |
| İlerleme Günlüğü | `docs/gunluk.md` + `git log` |
| Rapor Şekilleri | `rapor/figures/*.png` (adlar/açıklamalar `app/sayfalar/sekiller.py` içinde) |
| Veri Seti · Etiket Denetimi · Sızıntı | `data/derived/` |
| Sınıflandırma | `runs/cls/` (results.csv, per_class.csv, preds/, mcnemar.csv, efficiency.csv, bootstrap.csv, worst10.csv, train/*/results.csv). Geliştirmede başka klasör: `ODEV13_CLS_DIR=...` (adında "duman" varsa sayfada uyarı çıkar) |
| Segmentasyon | `runs/seg/` (results.csv, per_class.csv, preds/*_test_ornek_karisma.csv, efficiency.csv, worst10.csv, summary.json, train/*/{results,metrics}.csv; en kötü 10 görselleri için `preds/*_test.json.gz` — yoksa `rapor/figures/seg_en_kotu10.png`). Geliştirmede: `ODEV13_SEG_DIR=runs/seg_duman` (sayfada duman uyarısı) |
| Sözlük | sabit metin |
Panel dosyaları değişiklik zamanına göre önbelleği yeniler: veri yeniden üretilince sayfayı yenilemek yeter.

## 5. Yeni bir Mac'te kurulum (ör. Mac mini) — sırayla
1. **Homebrew** yoksa: https://brew.sh adresindeki tek satırlık komut.
2. **Python 3.13 + araçlar** (Colab ile aynı ana sürüm): `brew install python@3.13 gh git`
3. **GitHub girişi:** `gh auth login` → GitHub.com → HTTPS → tarayıcıyla giriş.
4. **Depoyu indir:** `gh repo clone toprakozkale/derin-ogrenme-odev13 ~/Desktop/derin_ogrenme_yl_odev` ve `cd ~/Desktop/derin_ogrenme_yl_odev`
5. **Sanal ortam** (PyTorch + Ultralytics dahil, ~1 GB):
   `/opt/homebrew/opt/python@3.13/bin/python3.13 -m venv .venv && .venv/bin/pip install -r requirements-local.txt`
6. **Jupyter çekirdeği (VS Code için):** `.venv/bin/python -m ipykernel install --user --name odev13 --display-name "Python 3.13 (odev13)"`
7. **Kaggle anahtarı:** eski Mac'teki `~/.kaggle/kaggle.json` dosyasını **AirDrop** ile gönder (asla git'e/sohbete koyma), sonra:
   `mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json`
   (Kaybolursa kaggle.com → profil → Settings → API → Create New Token; eskisi geçersiz olur ve Colab Secrets'taki `KAGGLE_KEY` de güncellenmeli.)
   Not: Kaggle kullanıcı adı **`toprakzkale`** ("o" yok), GitHub'daki `toprakozkale`'den farklı.
8. **Veriyi indir ve hazırla (~1 dk):** `.venv/bin/python scripts/prepare_data.py` → `data/` oluşur (temiz test 692 çıkmalı).
9. **Testler:** `.venv/bin/python -m pytest -q tests` → **18 passed**.
10. **Panel:** `.venv/bin/streamlit run app/Ana_Sayfa.py` → http://localhost:8501
11. **CLS sonuçları git'le gelir** (ağırlıklar hariç). Ağırlık gerekirse (yeniden tahmin vb.): Drive'da `odev13/runs/cls` → İndir →
    `.venv/bin/python scripts/import_runs.py ~/Downloads/cls-*.zip`. SEG sonuçları da aynı yolla (`seg-*.zip`).
12. (İsteğe bağlı) **Claude in Chrome** eklentisi: https://chromewebstore.google.com/detail/fcoeoabgfenejglbffodgkkbkcdhcgfn
    → Chrome'da Claude yan panelinde aynı hesapla giriş. Claude Colab sekmesini doğrudan okuyabilir.
13. Mac uyumasın (uzun Colab eğitimlerinde): `caffeinate -dimsu` (Terminal'de açık kalsın, Ctrl+C ile kapanır; kapak kapatılmaz).

Colab Secrets ve Roboflow hesaba bağlıdır; yeni Mac'te yeniden kurmak gerekmez.
**İki Mac arasında çalışma düzeni:** başlarken `git pull`, bitirirken commit + `git push`. `data/` her makinede komutla yeniden üretilir;
`runs/` içindeki sonuç tabloları git'le gelir, ağırlıklar Drive'dan `import_runs.py` ile alınır.
Claude'un makineye özel hafızası taşınmaz: Mac mini'deki Claude yalnız bu dosyayı, `docs/` ve `git log`'u bilir — eksik bağlam buraya yazılır.
RF-DETR (`rfdetr[train]` 1.12.0) `requirements-local.txt`'te; ön-eğitimli ağırlığı ilk kullanımda `~/.roboflow/models`'a iner (~135 MB).

## 6. Hatırlatıcılar — dış servisler
**Kaggle**
- Anahtar: `~/.kaggle/kaggle.json` (izin 600). Zip SHA-256: `e728dd56eaab4e11e64d46da77b06eb377beb264c53d0a32c1eaf44d051de525` (sürüm 1).

**Google Colab — genel**
- Colab Secrets (🔑, hesaba bağlı, kurulu): `GITHUB_TOKEN`, `KAGGLE_USERNAME`, `KAGGLE_KEY` — her birinde "Not defteri erişimi" açık.
  Adlar birebir böyle olmalı (bir kez `KAGGLE_USERNAME` yanlış yazıldı → `SecretNotFoundError`).
- `GITHUB_TOKEN`: fine-grained, yalnız `derin-ogrenme-odev13`, Contents **Read-only**, 31 Ara 2026'ya kadar. Token yalnız istek
  başlığında kullanılır, Drive'daki `.git/config`'e yazılmaz.
- Notebook'ların ilk hücresi Drive'ı bağlar, `MyDrive/odev13` yoksa depoyu klonlar, varsa `git pull` yapar, **çalışan kod sürümünü yazdırır**
  ve eski `src` modüllerini hafızadan siler. Kod GitHub'da değiştiyse Colab'da **Çalışma zamanı → Oturumu yeniden başlat → Tümünü çalıştır**.
- Notebook'u Colab'a almak: **Dosya → Not defteri yükle** → `notebooks/0X_....ipynb`. Çalışma zamanı türü **Python 3** olmalı.
- 01 notebook'u GPU istemez → CPU çalışma zamanında çalıştır; T4 yalnız 02/03 eğitimleri için.
- Drive'daki kod kopyasında git'te izlenmeyen bir dosya GitHub'daki aynı adlı dosyanın yolunu keserse `git pull` başarısız olur
  (12 Kas: `rapor/figures/cls_*`). 03'ün kurulum hücresi bunları `_git_yedek/`'e taşıyıp yeniden dener, olmazsa durur; 01/02'nin eski hücresi yalnız uyarır.
- Colab ortamı (11 Kas 2026): Tesla T4 15 GB, sürücü 580.82.07, CUDA 13.0, Python 3.13 (Eylül 2026 imajı).
- Ücretsiz plan: sekme/Mac açık kalmalı, oturum ≤ ~12 saat, arada "Hâlâ orada mısınız?" çıkabilir. Pro 9,99 $/ay, Pro+ 49,99 $/ay.
- Hocanın da çalıştırabileceği Colab linki kararı açık — issue #4 (öneri: teslim günü depoyu public yapmak).

**Google Colab — eğitim akışı (Aşama 2, issue #6)**
1. Colab → Dosya → Not defteri yükle → `notebooks/02_siniflandirma.ipynb`.
2. Çalışma zamanı türü: **Python 3 + T4 GPU** → Kaydet. Mac'te `caffeinate -dimsu`.
3. **Tümünü çalıştır.** Veri her oturumda ~1 dk'da yeniden üretilir; eğitim koşuları `MyDrive/odev13/runs/cls/train/` altına yazılır.
4. Bağlantı koparsa: Oturumu yeniden başlat → Tümünü çalıştır. Biten koşular atlanır, yarım kalan `last.pt`'den devam eder.
5. Bitince Drive'da `odev13/runs/cls` → sağ tık → İndir → yerelde `scripts/import_runs.py` → panelin Sınıflandırma sayfası dolar.
   Şekiller Drive'daki `rapor/figures`'a yazılır; indirilmediyse `.venv/bin/python scripts/make_cls_figures.py` aynı şekilleri
   kaydedilmiş tahminlerden yeniden çizer (ve `results.csv` ile tutarlılığı denetler).
6. `ENV.md`'ye Colab'daki torch/ultralytics sürümlerini (`runs/cls/env.json`) yaz; `docs/gunluk.md`'ye kayıt ekle.

**Google Colab — Aşama 2'den öğrenilenler**
- 9 CLS koşusu ~2 saat sürdü, ücretsiz kotaya tek oturumda sığdı. Hiçbir koşu 100 epoch'a ulaşmadı (erken durdurma 32–60. epoch'ta).
- Colab şekilleri Drive'daki `rapor/figures`'a yazar; Drive'dan yalnız `runs/` indirilirse şekiller yerelde betikle yeniden çizilir.
- Drive klasörü indirince zip adı `cls-<tarih>-1-001.zip` gibi; büyükse birden çok parça gelir — hepsi `import_runs.py`'a verilir.

**Roboflow**
- https://app.roboflow.com/topraks-workspace-k39ij — proje **Private** kalmalı. **Upgrade / Buy credits'e basma, kart girme.**
- Plan: Core (private, ayda 10 kredi, dönem her ayın 1'i). Sürüm oluşturma kredi harcar. Kredi biterse ücret alınmaz, işlem durur.
- Yüklemede Roboflow projede aynısı olan görüntüyü atlar (bizde 41 bölme içi kopya). Sürümde Auto-Orient ve Resize **kaldırılmalı**.
- Doğru dışa aktarım biçimi: **COCO Segmentation** (YOLOv8-OBB poligon değil, döndürülmüş kutudur — kullanma).

**Teslim**
- E-posta: tekinler.cs@gmail.com · PDF rapor + Colab linki (gizli pencerede açılıyor mu kontrol et) + sunum.
- Raporda kısa "Yapay zekâ araçlarının kullanımı" beyanı önerildi.

## 6a. Aşama 3 — segmentasyon (durum: Colab eğitimleri sürüyor, 5/9 bitti — 17 Kas)
Ayrıntılı plan: `docs/PLAN.md` §7 · issue'lar #10–#13 · puan ağırlığı %25 (en büyük kalem) + hata analizi %20'nin bir kısmı.
Kararlar ve değerlendirme kuralları: §3 ("SEG deney düzeni", "SEG değerlendirme", "RF-DETR ayrıntıları").

**Yapılanlar (12 Kas, #10):** `src/seg.py`, ortak SEG metrikleri (`src/metrics.py` + `tests/test_seg_metrics.py`),
`scripts/build_notebook_03.py` → `notebooks/03_segmentasyon.ipynb`, `.gitignore` istisnaları. Duman testi (CPU, 1 epoch, küçük alt
küme) bütün hücrelerde hatasız. RF-DETR sınıf eşlemesi çapraz kontrolle doğrulandı (bizim 0,132 ↔ kütüphane 0,130, aynı görüntüler).

**Kaldığı yerden devam (17 Kas sabahı itibarıyla; herhangi bir bilgisayardan, yalnız tarayıcı yeter)**
Durum: biten 5 koşu `ana_yolo26n-seg_im480_s0`, `ana_yolo26s-seg_im480_s0`, `ana_rfdetr-seg-small_im480_s0`, `ana_yolo26n-seg_im480_s1`,
`ana_yolo26s-seg_im480_s1` (her birinde `DONE`). Yarıda: `ana_rfdetr-seg-small_im480_s1` (`last.ckpt` 03:32, ~epoch 13). Başlamadı: seed 2'nin üçü.
1. Bilgisayarda depo varsa `git pull` (Colab için gerekmez; Colab kodu kendisi GitHub'dan çeker).
2. **Defteri temiz yükle:** Colab'daki eski kopyada eğitim hücresi en üste (kurulum hücresinin üstüne) kaymış; öyle çalıştırılırsa
   `NameError` verir. En kolayı: Colab → **Dosya → Not defteri yükle** → depodaki `notebooks/03_segmentasyon.ipynb` (bilgisayarda depo
   yoksa GitHub'da dosyayı açıp "Download raw file" ile indir). Eski kopyayı kullanacaksan önce hücreyi taşı: hücreye tıkla →
   **Düzenle → Hücreyi veya seçimi kes** → "3 · Eğitim" altındaki açıklama metnine tıkla → **Düzenle → Yapıştır** ("Süre özeti"nin üstü).
3. **Çalışma zamanı → Çalışma zamanı türünü değiştir → T4 GPU** → Bağlan → **Tümünü çalıştır**. Gizli anahtar izin pencerelerine "Erişim izni ver".
   Biten koşular "atlanıyor" der; RF-DETR s1 `last.ckpt`'den sürer. "GPU kullanım sınırı" uyarısı çıkarsa kota dolmuştur → birkaç saat/1 gün bekle.
4. Bilgisayar uyumasın (Mac: `caffeinate -dimsu`; Windows: Güç ayarlarında uyku "Hiçbir zaman"), Colab sekmesi açık kalsın.
5. Kontrol: Drive → `odev13/runs/seg/train/<koşu>/` dosya saatleri (`last.ckpt`/`metrics.csv` RF-DETR, `results.csv` YOLO) ilerliyor mu.
6. 9/9 bitince değerlendirme hücreleri otomatik çalışır (test değerlendirmesi, karışma, verimlilik, en kötü 10, `summary.json`).
   Sonra aşağıdaki akışın 5–7. adımları. Colab sürümleri (16 Kas `env.json`): Python 3.13.16, torch 2.14.1+cu130, ultralytics 8.4.215,
   **rfdetr 1.12.1** (yerelde 1.12.0), supervision 0.31.7, pycocotools 2.0.11. Drive 17 Kas: 4,4 / 15 GB dolu.

**Google Colab — Aşama 3 eğitim akışı (#11)**
1. Colab → Dosya → Not defteri yükle → `notebooks/03_segmentasyon.ipynb`. Çalışma zamanı türü **Python 3 + T4 GPU**. Mac'te `caffeinate -dimsu`.
2. **Tümünü çalıştır.** İlk hücre `ultralytics` ve `rfdetr[train]` kurar (Colab'daki torch'u değiştirip değiştirmediğine `env.json`'da bak).
3. Eğitim sırası: seed 0'ın üç modeli → seed 1 → seed 2. Kota/bağlantı biterse **Oturumu yeniden başlat → Tümünü çalıştır**:
   biten koşular atlanır, yarım kalan `last.pt` / `last.ckpt`'den sürer. **Arada ayar (özellikle çözünürlük) değiştirme.**
4. İlk oturumdan sonra "Süre özeti" hücresinden epoch sürelerine bak → kullanıcıyla seed sayısını kesinleştir
   (2 seed'e inmek için Colab'da ilk hücreden önce `os.environ["ODEV13_SEG_SEEDS"]="0,1"`; karar §3'e ve günlüğe yazılır).
5. Değerlendirme hücresinde **çapraz kontrol tablosuna bak**: `val_mask_map50_95_bizim` ≈ `..._kutuphane` olmalı; çok farklıysa dur, incele.
6. Bitince Drive'da `odev13/runs/seg` → İndir → `.venv/bin/python scripts/import_runs.py ~/Downloads/seg-*.zip`.
   Şekiller Drive'daki `rapor/figures/seg_*`; ham tahminler (`preds/*.json.gz`) ve ağırlıklar yalnız Drive'da kalır.
7. `ENV.md`'ye Colab sürümlerini (`runs/seg/env.json`) yaz; `docs/gunluk.md`'ye süreler + sonuçlar.
- **Drive yeri:** RF-DETR koşusu eğitim sırasında ~800 MB tutar (`last.ckpt` ~535 MB), bitince ~134 MB'a iner; YOLO koşuları küçük.
  Ücretsiz Drive 15 GB — eğitimden önce boş yer kontrol edilmeli.

**Kalanlar:** #11 Colab eğitimleri → #12 sonuçların yorumu (mAP/mIoU/Dice karşılaştırması, hız/bellek, en kötü 10 hata türleri,
meyve/yaprak karışması SEG'de) → #13 panel Segmentasyon sayfası **yazıldı** (`app/sayfalar/segmentasyon.py`); gerçek sonuçlar
alınınca kontrol edilecek → #14 rapor. Bağlam notu: Afzaal 2021'in Mask R-CNN sonuçlarıyla karşılaştırma (aynı bölme mi, kontrol et).
Takvim: 2. rapor 8 Aralık; eğitimler en geç ~3 Aralık'ta bitmeli. Arada 24 Kasım'da 1. rapor var.

## 7. Durum ve eksikler
Güncel ilerleme: panelin Genel Bakış sayfası (`docs/durum.yaml`) ve İlerleme Günlüğü (`docs/gunluk.md`). Açık işler GitHub issue:
`gh issue list` · depo https://github.com/toprakozkale/derin-ogrenme-odev13 (private) · kilometre taşları "1. rapor (CLS)" ve "2. rapor (SEG)".
Bir iş bitince: ilgili issue'yu kapat, `docs/durum.yaml` + `docs/gunluk.md` + bu dosyanın §0'ını güncelle.

| # | İş | Durum |
|---|---|---|
| 1 | [Karar] Sızıntı: resmi bölme + temiz test onayı | ✅ onaylandı, kapandı (11 Kas) |
| 2 | Roboflow çapraz kontrolü | ✅ kapandı (11 Kas) |
| 3 | 01 notebook'u Colab'da çalıştır | ✅ kapandı (11 Kas) |
| 4 | [Karar] Hocanın çalıştırabileceği Colab linki (private depo) | ✅ karar (22 Kas): teslim günü public; defterler misafir kipine hazır — public yapma işi teslim günü |
| 5 | 02 notebook'u + yerel duman testi | ✅ kapandı (11 Kas) |
| 6 | Colab: CLS eğitimleri | ✅ kapandı (11 Kas bitti, 12 Kas içe alındı) |
| 7 | CLS sonuç analizi + en kötü 10 yorumları | 1. rapor taslağında yazıldı (22 Kas) |
| 8 | Panel CLS sayfası | ✅ kapandı (12 Kas, gerçek sonuçla kontrol edildi) |
| 9 | 1. rapor — 24 Kasım | taslak hazır (22 Kas, 4 sayfa PDF) — kullanıcı okuyacak |
| 10 | 03 segmentasyon notebook'u + yerel duman testi | ✅ kapandı (12 Kas) |
| 11 | Colab: SEG eğitimleri (YOLO26-seg n/s, RF-DETR Seg Small) | **sürüyor** — 8/9 bitti (22 Kas); RF-DETR s2 son koşu |
| 12 | SEG sonuçları: karşılaştırma + hata analizi | 11'den sonra |
| 13 | Panel: Segmentasyon sayfası | sayfa hazır (12 Kas; duman verisiyle sınandı) — gerçek sonuçlarla kontrol #11'den sonra |
| 14 | 2. rapor + sunum — 8 Aralık | Aşama 3 sonrası |
| 15 | Mac mini kurulumu | ✅ kapandı (12 Kas): klon `~/Desktop/derin-ogrenme-odev13` (§5'teki addan farklı), Python 3.13.18, veri hazır (temiz test 692), testler geçti, panel açıldı. Disk ~16 GB boş — dikkat |

## 8. Sık komutlar
```bash
git pull                                          # başka Mac'te yapılanları al
.venv/bin/python scripts/prepare_data.py          # Aşama 1'i baştan üret (data/processed, data/derived)
.venv/bin/python -m pytest -q tests               # testler (18)
.venv/bin/streamlit run app/Ana_Sayfa.py          # panel → http://localhost:8501
.venv/bin/python scripts/import_runs.py ~/Downloads/cls-*.zip   # Colab sonuçlarını runs/'a al
.venv/bin/python scripts/make_cls_figures.py      # CLS şekillerini runs/cls'ten yeniden çiz + tutarlılık denetimi
.venv/bin/python scripts/build_notebook_02.py     # 02 notebook'unu yeniden üret (01 için _01, 03 için _03)
cd notebooks && ODEV13_SMOKE=1 ../.venv/bin/jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=odev13 02_siniflandirma.ipynb --output-dir /tmp   # 02 duman testi
cd notebooks && ODEV13_SMOKE=1 ../.venv/bin/jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=odev13 --ExecutePreprocessor.timeout=1800 03_segmentasyon.ipynb --output-dir /tmp   # 03 duman testi (~1 dk; runs/seg_duman)
```
