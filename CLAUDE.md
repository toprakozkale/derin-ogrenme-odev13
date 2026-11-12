# CLAUDE.md — Ödev 13: Çilek Hastalığı (CLS + SEG)

Bu dosya Claude'un her oturumda otomatik okuduğu proje hafızasıdır. Claude'un makineye özel hafızası (`~/.claude/...`)
başka bilgisayara taşınmaz; **bu proje hakkında bilinmesi gereken her şey burada** ve `docs/` altındadır
(ayrıntılı tarihçe: `docs/gunluk.md`). Her önemli karar ya da aşama sonunda güncellenir.

---

## 0. Kaldığımız yer — yeni oturumda önce bunu oku
**Son güncelleme: 12 Kasım 2026 (öğleden sonra).** Teslimler: 1. rapor (CLS) **24 Kasım 23:59** · 2. rapor (SEG) **8 Aralık 23:59**.

> **Kullanıcı 12 Kas'ta: "Segmentasyonu başka bilgisayardan (Mac mini) yapacağız."** Bu dosya oraya devir için güncellendi.
> Mac mini'de ilk iş: §5 kurulumu (hiç kurulmadıysa) → `git pull` → `pytest` 13 passed → paneli aç → **§6a'daki Aşama 3 planıyla başla (#10).**

- ✅ Aşama 0 (kurulum) ve Aşama 1 (veri hazırlama) **tamamen** bitti; 01 notebook'u hem Mac'te hem Colab'da çalıştı, sonuçlar birebir aynı.
- ✅ Roboflow çapraz kontrolü bitti (#2): 5.571 örnek, ort. IoU 0,99995; Roboflow'un atladığı 41 görüntü = bölme içi bayt kopyaları (train 13, test 28).
- ✅ Sızıntı yaklaşımı **onaylandı** (11 Kas, #1): resmi bölme + tüm test ve temiz test ayrı raporlanır.
- ✅ **Aşama 2 (CLS) eğitimleri bitti (#6, #8):** 9 koşu Colab T4'te ~2 saatte hatasız (11 Kas). Sonuçların küçük dosyaları
  **git'te** (`runs/cls/*.csv`, `preds/`, `train/*/results.csv`) → yeni Mac'te `git pull` yeter, panel dolu açılır.
  Ağırlıklar (`*.pt`, ~120 MB) yalnız Drive'da (`MyDrive/odev13/runs/cls`). Şekiller `rapor/figures/cls_*` (git'te).
  Test doğruluğu (3 seed ort. ± std): nano 0,915 ± 0,015 · small 0,904 ± 0,005 · nano-384 0,934 ± 0,008; temiz test ~0,5 puan düşük.
  Val ile seçilen: `ana_yolo26n-cls_im224_s1`. Gruplar arası (yaprak/meyve/çiçek) karışma ~%2. Ayrıntı: `docs/gunluk.md` (12 Kas).
- ⏭️ **Sıradaki iş: Aşama 3 — segmentasyon (#10 → #11 → #12 → #13).** Plan ve açık tasarım kararları §6a'da.
- ⏸️ #7 (CLS sonuç yorumu, en kötü 10 açıklaması) ve raporlar (#9, #14) + sunum: **kullanıcı "hepsini en son yaparız" dedi** —
  kendiliğinden başlama; ama 1. rapor 24 Kas'ta teslim, takvimi kullanıcıya hatırlat.
- ⏳ Kullanıcının kararı bekleyen: hocanın çalıştırabileceği Colab linki (#4; öneri: teslim günü depoyu public yapmak).

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
  Aşama sonunda kullanıcı "GitHub'a ve panele detaylı kaydet" ister: günlük kaydı + durum + panel + şekiller + bu dosya.
- **Raporlar ve sunum en sona** (kullanıcı kararı, 11 Kas). Kullanıcı istemeden rapor taslağına başlama.
- Kullanıcıya bir şeyi "yaptım" demeden önce doğrula; yanlış bir iddia fark edilirse açıkça düzelt ve günlüğe düzeltme notu düş
  (örnek: 41 bayt kopyası önce yanlışlıkla "bölmeler arası" sanılmıştı).
- Colab'daki çıktıları okumak gerekirse: **Claude in Chrome** eklentisi (MacBook'ta kurulu; Mac mini'de §5 adım 12 ile kurulmalı).
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
| Sonuçlar git'te mi? (12 Kas) | Eğitim sonuçlarının **küçük metin dosyaları git'te** (`.gitignore` istisnaları: `runs/cls/*.csv|json`, `preds/*.csv`, `train/*/{results.csv,args.yaml,DONE}`), ağırlık ve eğitim görselleri yalnız Drive'da. İki Mac arasında Drive indirmesi gerekmesin diye. SEG için aynı istisnalar `runs/seg/` için eklenecek. |

## 4. Depo yapısı
```
src/            config.py (yollar, sınıflar, eşikler; ODEV13_ROOT / ODEV13_DATA ile değiştirilebilir)
                dataset.py (LabelMe okuma/düzeltme/istatistik) · convert.py (CLS / YOLO-seg / COCO + geri-dönüş doğrulaması)
                dedup.py (MD5/pHash, sızıntı) · pipeline.py (Aşama 1 hattı + indirme) · roboflow_check.py (Roboflow COCO ↔ bizim dönüşüm)
                metrics.py (P/R/F1, karmaşıklık + yaprak/meyve/çiçek grupları, bootstrap GA, McNemar)
                cls.py (Aşama 2: eğitim — kaldığı yerden devam —, tahmin, değerlendirme, verimlilik, en kötü 10)
scripts/        prepare_data.py (Aşama 1'i koşar) · build_notebook_01.py / _02.py (notebook'ları üretir)
                _nb_common.py (notebook'ların ortak Colab kurulum hücresi) · import_runs.py (Drive'dan inen sonuç zip'ini runs/'a alır)
                make_cls_figures.py (runs/cls'ten rapor şekilleri + tutarlılık denetimi)
                prepare_roboflow_upload.py (ham veriden data/roboflow_upload/{train,valid,test})
notebooks/      01_veri_hazirlama.ipynb (çıktılı; Mac + Colab'da çalıştı) · 02_siniflandirma.ipynb (Colab'da çalıştı, 11 Kas) · 03 yazılacak (#10)
app/            Streamlit paneli: Ana_Sayfa.py + sayfalar/ (klasör adı bilerek "pages" değil — Streamlit'in eski otomatik menüsü devreye girmesin)
tests/          pytest: veri hattı (8) + metrikler (5) = 13 test
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
11. **CLS sonuçları git'le gelir** (ağırlıklar hariç). Ağırlık gerekirse (yeniden tahmin vb.): Drive'da `odev13/runs/cls` → İndir →
    `.venv/bin/python scripts/import_runs.py ~/Downloads/cls-*.zip`. SEG sonuçları da aynı yolla (`seg-*.zip`).
12. (İsteğe bağlı) **Claude in Chrome** eklentisi: https://chromewebstore.google.com/detail/fcoeoabgfenejglbffodgkkbkcdhcgfn
    → Chrome'da Claude yan panelinde aynı hesapla giriş. Claude Colab sekmesini doğrudan okuyabilir.
13. Mac uyumasın (uzun Colab eğitimlerinde): `caffeinate -dimsu` (Terminal'de açık kalsın, Ctrl+C ile kapanır; kapak kapatılmaz).

Colab Secrets ve Roboflow hesaba bağlıdır; yeni Mac'te yeniden kurmak gerekmez.
**İki Mac arasında çalışma düzeni:** başlarken `git pull`, bitirirken commit + `git push`. `data/` her makinede komutla yeniden üretilir;
`runs/` içindeki sonuç tabloları git'le gelir, ağırlıklar Drive'dan `import_runs.py` ile alınır.
Claude'un makineye özel hafızası taşınmaz: Mac mini'deki Claude yalnız bu dosyayı, `docs/` ve `git log`'u bilir — eksik bağlam buraya yazılır.
RF-DETR yerelde henüz kurulu değil: 03 yazılırken `.venv/bin/pip install "rfdetr[train]"` (sürüm `requirements-local.txt` ve `ENV.md`'ye işlenir).

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

## 6a. Aşama 3 — segmentasyon planı (Mac mini'de buradan başla)
Ayrıntılı plan: `docs/PLAN.md` §7 · issue'lar #10–#13 · puan ağırlığı %25 (en büyük kalem) + hata analizi %20'nin bir kısmı.

**Hazır olanlar (Aşama 1'den):**
- YOLO-seg verisi: `data/processed/yoloseg/` (`data.yaml`, `images/{train,val,test}`, `labels/...`; `data.yaml` yolu makineye göre yeniden üretilir).
- COCO verisi: `data/processed/coco/{train,valid,test}/_annotations.coco.json` — RF-DETR'in beklediği Roboflow düzeni;
  kategori 0 = üst kategori, sınıflar 1..7 (= YOLO kimliği + 1). Geri-dönüş IoU'su ≥ 0,99 doğrulandı, Roboflow'la da karşılaştırıldı.
- Temiz test listesi: `data/derived/splits/test_clean.txt` (692). Gruplar: `src/config.py: GROUP_OF`.
- Örnek alınacak kod: `src/cls.py` (RunSpec adlandırma, `last.pt`'den devam, DONE işareti, SMOKE modu, `efficiency`, `select_by_val`),
  `scripts/build_notebook_02.py` + `_nb_common.py` (Colab kurulum hücresi), `app/sayfalar/siniflandirma.py` (panel sayfası düzeni).

**Yapılacaklar (sırayla):**
1. `src/seg.py` + `scripts/build_notebook_03.py` → `notebooks/03_segmentasyon.ipynb` (#10). Modeller: `yolo26n-seg`, `yolo26s-seg`, RF-DETR Seg
   (`pip install "rfdetr[train]"`; paketin o günkü güncel küçük/orta Seg modeli — adı ve sürümü kurulunca kontrol edilip `ENV.md`'ye yazılır,
   **tahminle yazılmaz**).
2. **Ortak değerlendirme kodu** (adaletin kalbi; `src/metrics.py`'ye eklenip testle sınanmalı): her iki modelin test tahminleri →
   COCO results JSON → `pycocotools` mask **mAP50 / mAP50-95** (+ sınıf bazında AP); maskeler sınıf bazında semantik haritaya →
   **mIoU, Dice, sınıf bazında IoU**; IoU ≥ 0,5 eşleşmeleriyle **sınıf karışma matrisi + yaprak/meyve/çiçek** (Dikkat maddesi SEG'de de).
   Hepsi tüm test (743) + temiz test (692). Ultralytics/RF-DETR'in kendi raporladığı değerler yalnız ek tablo.
3. Verimlilik: aynı T4, batch 1, 20 ısınma, ms/görüntü (ön/son işlem dahil/hariç), parametre, `torch.cuda.max_memory_allocated`.
4. Hata analizi: görüntü başına IoU'su en düşük 10 test görüntüsü → GT | YOLO | RF-DETR yan yana; hata türleri.
5. Yerel duman testi (CPU, küçük alt küme, `runs/seg_duman/`) → kullanıcı Colab'da çalıştırır (#11) → `import_runs.py` → panel SEG sayfası (#13).
6. `.gitignore`'a `runs/seg/` için CLS'deki gibi küçük-dosya istisnaları ekle.

**03'ü yazmadan önce karar verilecekler (kullanıcıya sade dille sor / öner):**
- Giriş çözünürlüğü: görüntüler 419×419. Plan YOLO için 640 diyor (büyütme); RF-DETR'in çözünürlüğü paketin kuralına bağlı.
  İkisini mümkün olduğunca yakın tut, farkı raporla.
- Seed sayısı ve epoch bütçesi: önce duman testiyle T4'te epoch süresi ölçülür; toplam GPU süresi kabaca 10–17 saat tahmin edilmişti
  (ölçülmedi). Ücretsiz Colab'a göre birkaç güne yayılır; gerekirse 3 → 2 seed (§3). Eğitim parçaları 2–3 saatlik olmalı, kaldığı yerden devam etmeli.
- Takvim: 2. rapor 8 Aralık; eğitimler en geç ~3 Aralık'ta bitmeli. Arada 24 Kasım'da 1. rapor teslimi var.

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
| 6 | Colab: CLS eğitimleri | ✅ kapandı (11 Kas bitti, 12 Kas içe alındı) |
| 7 | CLS sonuç analizi + en kötü 10 yorumları | sayılar/şekiller hazır; yorum raporla birlikte en sona |
| 8 | Panel CLS sayfası | ✅ kapandı (12 Kas, gerçek sonuçla kontrol edildi) |
| 9 | 1. rapor — 24 Kasım | en sona (kullanıcı isteği) |
| 10 | 03 segmentasyon notebook'u + yerel duman testi | **sıradaki** (Mac mini'de; plan §6a) |
| 11 | Colab: SEG eğitimleri (YOLO26-seg n/s, RF-DETR) | 10'dan sonra |
| 12 | SEG sonuçları: karşılaştırma + hata analizi | 11'den sonra |
| 13 | Panel: Segmentasyon sayfası | 11'den sonra |
| 14 | 2. rapor + sunum — 8 Aralık | Aşama 3 sonrası |
| 15 | Mac mini kurulumu | **şimdi** (12 Kas) — §5 |

## 8. Sık komutlar
```bash
git pull                                          # başka Mac'te yapılanları al
.venv/bin/python scripts/prepare_data.py          # Aşama 1'i baştan üret (data/processed, data/derived)
.venv/bin/python -m pytest -q tests               # testler (13)
.venv/bin/streamlit run app/Ana_Sayfa.py          # panel → http://localhost:8501
.venv/bin/python scripts/import_runs.py ~/Downloads/cls-*.zip   # Colab sonuçlarını runs/'a al
.venv/bin/python scripts/make_cls_figures.py      # CLS şekillerini runs/cls'ten yeniden çiz + tutarlılık denetimi
.venv/bin/python scripts/build_notebook_02.py     # 02 notebook'unu yeniden üret (01 için build_notebook_01.py; 03 yazılınca _03)
cd notebooks && ODEV13_SMOKE=1 ../.venv/bin/jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=odev13 02_siniflandirma.ipynb --output-dir /tmp   # 02 duman testi
```
