# CLAUDE.md — Ödev 13: Çilek Hastalığı (CLS + SEG)

Bu dosya Claude'un her oturumda otomatik okuduğu proje hafızasıdır. Claude'un makineye özel hafızası (`~/.claude/...`)
başka bilgisayara taşınmaz; **bu proje hakkında bilinmesi gereken her şey burada** ve `docs/` altındadır
(ayrıntılı tarihçe: `docs/gunluk.md`). Her önemli karar ya da aşama sonunda güncellenir.

---

## 0. Kaldığımız yer — yeni oturumda önce bunu oku
**Son güncelleme: 11 Kasım 2026 (öğleden önce).** 1. rapora (CLS) son teslim **24 Kasım 23:59**.

- ✅ Aşama 0 (kurulum) ve Aşama 1 (veri hazırlama) **tamamen** bitti; 01 notebook'u hem Mac'te hem Colab'da çalıştı, sonuçlar birebir aynı.
- ✅ Roboflow çapraz kontrolü bitti (#2): 5.571 örnek, ort. IoU 0,99995; Roboflow'un atladığı 41 görüntü = bölme içi bayt kopyaları (train 13, test 28).
- ✅ Aşama 2 kodu hazır: `02_siniflandirma` notebook'u yazıldı, yerel duman testi hatasız. Panelin Sınıflandırma sayfası sonuç
  dosyaları gelince kendiliğinden doluyor (duman verisiyle denendi).
- ⏳ **Colab'da sınıflandırma eğitimi başladı (11 Kas ~09:20, issue #6)** — T4, torch 2.14.1+cu130, Ultralytics 8.4.210. Bitince §6 adım 5–6.
  Süre henüz ölçülmedi; kullanıcı ilk epoch satırlarını gönderince hesaplanacak.
- ✅ Sızıntı yaklaşımı **onaylandı** (11 Kas, #1): resmi bölme + tüm test ve temiz test ayrı raporlanır.
- ⏳ Kullanıcının kararı bekleyen: hocanın çalıştırabileceği Colab linki (#4; öneri: teslim günü depoyu public yapmak).
- Yeni bilgisayarda başlıyorsan: §5'teki kurulumu yap, sonra `git pull` ve paneli aç.

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
- Kullanıcı bir aşama bitince "GitHub'a gönder" der; commit + push sonrası `docs/gunluk.md`, `docs/durum.yaml` ve bu dosya güncel olmalı.
- Colab'daki çıktıları okumak gerekirse: kullanıcının Chrome'unda **Claude in Chrome** eklentisi var (bu Mac'te kurulu ve bağlı);
  Colab notebook'unu yeni sekmede açıp `get_page_text` ile okumak mümkün. Tablolar/grafikler için ekran görüntüsü gerekir.

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
| Model ailesi | **YOLO26** (Ultralytics 8.4.210'daki en güncel aile; plan "o tarihteki güncel aile" diyordu): CLS `yolo26n-cls` (1,54 M parametre), `yolo26s-cls`; SEG için `yolo26n-seg`, `yolo26s-seg` mevcut. |
| CLS deney düzeni | imgsz 224, 100 epoch + patience 20, batch 64, optimizer auto, seed 0/1/2 → 6 ana koşu; ablasyon: nano imgsz 384 × 3 seed. Model seçimi val doğruluğuyla; en kötü 10 tahmin val ile seçilen koşudan. Karmaşıklık matrisi 3 seed toplamı. |
| Duman testi | Yerelde GPU yokken `SMOKE` modu (1 epoch, verinin %5'i) → çıktılar `runs/cls_duman/`; asıl sonuçlara/panele/rapora **asla** karışmaz. |
| Colab düzeni | Kod + sonuçlar Drive'da (`MyDrive/odev13`), veri Colab'ın yerel diskinde (`/content/odev13_data`, `ODEV13_DATA`). Ücretsiz plan; sekme açık kalmalı. Modeller **T4**'te koşulur (adil hız ölçümü). |

## 4. Depo yapısı
```
src/            config.py (yollar, sınıflar, eşikler; ODEV13_ROOT / ODEV13_DATA ile değiştirilebilir)
                dataset.py (LabelMe okuma/düzeltme/istatistik) · convert.py (CLS / YOLO-seg / COCO + geri-dönüş doğrulaması)
                dedup.py (MD5/pHash, sızıntı) · pipeline.py (Aşama 1 hattı + indirme) · roboflow_check.py (Roboflow COCO ↔ bizim dönüşüm)
                metrics.py (P/R/F1, karmaşıklık + yaprak/meyve/çiçek grupları, bootstrap GA, McNemar)
                cls.py (Aşama 2: eğitim — kaldığı yerden devam —, tahmin, değerlendirme, verimlilik, en kötü 10)
scripts/        prepare_data.py (Aşama 1'i koşar) · build_notebook_01.py / _02.py (notebook'ları üretir)
                _nb_common.py (notebook'ların ortak Colab kurulum hücresi) · import_runs.py (Drive'dan inen sonuç zip'ini runs/'a alır)
                prepare_roboflow_upload.py (ham veriden data/roboflow_upload/{train,valid,test})
notebooks/      01_veri_hazirlama.ipynb (çıktılı; Mac + Colab'da çalıştı) · 02_siniflandirma.ipynb (Colab'da eğitilecek) · 03 yazılacak
app/            Streamlit paneli: Ana_Sayfa.py + sayfalar/ (klasör adı bilerek "pages" değil — Streamlit'in eski otomatik menüsü devreye girmesin)
tests/          pytest: veri hattı (8) + metrikler (5) = 13 test
docs/           PLAN.md · gunluk.md (ilerleme günlüğü, panelde gösterilir) · veri_bulgulari.md · durum.yaml (panelin ilerleme kaynağı)
                phash_calibration.csv · roboflow/ (çapraz kontrol özeti, atlananlar, en düşük IoU'lar) · hocanın PDF'i
rapor/figures/  rapora girecek şekiller (notebook'lar üretir; panelde "Rapor Şekilleri")
data/           GIT'TE YOK — raw/, processed/ (cls, yoloseg, coco), derived/ (özet, CSV'ler) → scripts/prepare_data.py yeniden üretir
runs/           GIT'TE YOK — eğitim çıktıları (Colab → Drive → scripts/import_runs.py → buraya)
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
| Segmentasyon | henüz yer tutucu (Aşama 3) |
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
9. **Testler:** `.venv/bin/python -m pytest -q tests` → **13 passed**.
10. **Panel:** `.venv/bin/streamlit run app/Ana_Sayfa.py` → http://localhost:8501
11. **Colab sonuçları varsa:** Drive'da `odev13/runs/cls` klasörünü indir → `.venv/bin/python scripts/import_runs.py ~/Downloads/cls-*.zip`
12. (İsteğe bağlı) **Claude in Chrome** eklentisi: https://chromewebstore.google.com/detail/fcoeoabgfenejglbffodgkkbkcdhcgfn
    → Chrome'da Claude yan panelinde aynı hesapla giriş. Claude Colab sekmesini doğrudan okuyabilir.
13. Mac uyumasın (uzun Colab eğitimlerinde): `caffeinate -dimsu` (Terminal'de açık kalsın, Ctrl+C ile kapanır; kapak kapatılmaz).

Colab Secrets ve Roboflow hesaba bağlıdır; yeni Mac'te yeniden kurmak gerekmez.
**İki Mac arasında çalışma düzeni:** başlarken `git pull`, bitirirken commit + `git push`. `data/` her makinede komutla yeniden üretilir;
`runs/` git'te değildir → Drive'dan `import_runs.py` ile alınır.

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
- Colab ortamı (11 Kas 2026): Tesla T4 15 GB, sürücü 580.82.07, CUDA 13.0, Python 3.13 (Eylül 2026 imajı).
- Ücretsiz plan: sekme/Mac açık kalmalı, oturum ≤ ~12 saat, arada "Hâlâ orada mısınız?" çıkabilir. Pro 9,99 $/ay, Pro+ 49,99 $/ay.
- Hocanın da çalıştırabileceği Colab linki kararı açık — issue #4 (öneri: teslim günü depoyu public yapmak).

**Google Colab — eğitim akışı (Aşama 2, issue #6)**
1. Colab → Dosya → Not defteri yükle → `notebooks/02_siniflandirma.ipynb`.
2. Çalışma zamanı türü: **Python 3 + T4 GPU** → Kaydet. Mac'te `caffeinate -dimsu`.
3. **Tümünü çalıştır.** Veri her oturumda ~1 dk'da yeniden üretilir; eğitim koşuları `MyDrive/odev13/runs/cls/train/` altına yazılır.
4. Bağlantı koparsa: Oturumu yeniden başlat → Tümünü çalıştır. Biten koşular atlanır, yarım kalan `last.pt`'den devam eder.
5. Bitince Drive'da `odev13/runs/cls` → sağ tık → İndir → yerelde `scripts/import_runs.py` → panelin Sınıflandırma sayfası dolar.
6. `ENV.md`'ye Colab'daki torch/ultralytics sürümlerini (`runs/cls/env.json`) yaz; `docs/gunluk.md`'ye kayıt ekle.

**Roboflow**
- https://app.roboflow.com/topraks-workspace-k39ij — proje **Private** kalmalı. **Upgrade / Buy credits'e basma, kart girme.**
- Plan: Core (private, ayda 10 kredi, dönem her ayın 1'i). Sürüm oluşturma kredi harcar. Kredi biterse ücret alınmaz, işlem durur.
- Yüklemede Roboflow projede aynısı olan görüntüyü atlar (bizde 41 bölme içi kopya). Sürümde Auto-Orient ve Resize **kaldırılmalı**.
- Doğru dışa aktarım biçimi: **COCO Segmentation** (YOLOv8-OBB poligon değil, döndürülmüş kutudur — kullanma).

**Teslim**
- E-posta: tekinler.cs@gmail.com · PDF rapor + Colab linki (gizli pencerede açılıyor mu kontrol et) + sunum.
- Raporda kısa "Yapay zekâ araçlarının kullanımı" beyanı önerildi.

## 7. Durum ve eksikler
Güncel ilerleme: panelin Genel Bakış sayfası (`docs/durum.yaml`) ve İlerleme Günlüğü (`docs/gunluk.md`). Açık işler GitHub issue:
`gh issue list` · depo https://github.com/toprakozkale/derin-ogrenme-odev13 (private) · kilometre taşları "1. rapor (CLS)" ve "2. rapor (SEG)".
Bir iş bitince: ilgili issue'yu kapat, `docs/durum.yaml` + `docs/gunluk.md` + bu dosyanın §0'ını güncelle.

| # | İş | Durum |
|---|---|---|
| 1 | [Karar] Sızıntı: resmi bölme + temiz test onayı | ✅ onaylandı, kapandı (11 Kas) |
| 2 | Roboflow çapraz kontrolü | ✅ kapandı (11 Kas) |
| 3 | 01 notebook'u Colab'da çalıştır | ✅ kapandı (11 Kas) |
| 4 | [Karar] Hocanın çalıştırabileceği Colab linki (private depo) | açık — kullanıcı |
| 5 | 02 notebook'u + yerel duman testi | ✅ kapandı (11 Kas) |
| 6 | Colab: CLS eğitimleri | **sürüyor** (11 Kas'ta başlatıldı) |
| 7 | CLS sonuç analizi + en kötü 10 yorumları | eğitimden sonra |
| 8 | Panel CLS sayfası | kod hazır ve duman verisiyle denendi; gerçek sonuçla kontrol edilip kapanacak |
| 9 | 1. rapor — 24 Kasım | eğitimden sonra |
| 10–13 | 03 notebook · Colab SEG eğitimleri · SEG analizi · panel SEG sayfası | Aşama 3 |
| 14 | 2. rapor + sunum — 8 Aralık | Aşama 3 sonrası |
| 15 | Mac mini kurulumu | açık |

## 8. Sık komutlar
```bash
git pull                                          # başka Mac'te yapılanları al
.venv/bin/python scripts/prepare_data.py          # Aşama 1'i baştan üret (data/processed, data/derived)
.venv/bin/python -m pytest -q tests               # testler (13)
.venv/bin/streamlit run app/Ana_Sayfa.py          # panel → http://localhost:8501
.venv/bin/python scripts/import_runs.py ~/Downloads/cls-*.zip   # Colab sonuçlarını runs/'a al
.venv/bin/python scripts/build_notebook_02.py     # 02 notebook'unu yeniden üret (01 için build_notebook_01.py)
cd notebooks && ODEV13_SMOKE=1 ../.venv/bin/jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=odev13 02_siniflandirma.ipynb --output-dir /tmp   # 02 duman testi
```
