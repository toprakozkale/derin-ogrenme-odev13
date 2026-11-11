# CLAUDE.md — Ödev 13: Çilek Hastalığı (CLS + SEG)

Bu dosya Claude'un her oturumda otomatik okuduğu proje hafızasıdır. Claude'un makineye özel hafızası (`~/.claude/...`)
başka bilgisayara taşınmaz; **bu proje hakkında bilinmesi gereken her şey burada** ve `docs/` altındadır.
Her önemli karar ya da aşama sonunda güncellenir.

---

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
  ilk geçtiği yerde açıkla. Tarayıcıda/Colab'da ne yapacağını menü adlarıyla (Türkçe arayüz) tarif et.
- **İş bölümü:** Claude kodu, notebook'ları, rapor (.docx → PDF) ve sunum taslaklarını yazar; kullanıcı eğitimleri Colab'da (T4) çalıştırır.
- **Hiçbir sonuç uydurulmaz veya tahmin edilmez.** Rapordaki her sayı gerçek bir çalıştırma çıktısından gelir; yoksa yer tutucu bırakılır.
- Tekrarlanabilirlik: sabit seed (bölme 42, eğitim 0/1/2), sürümler `ENV.md` + `requirements-local.txt`, bölme listeleri SHA-256 ile.
- Test kümesi yalnız en sonda bir kez kullanılır; model seçimi val ile.
- Adil karşılaştırma: YOLO-seg ve RF-DETR aynı bölme, benzer çözünürlük/bütçe, aynı T4, **aynı değerlendirme kodu**.
- Kaggle anahtarı, şifreler, token'lar asla sohbete, koda veya git'e girmez.
- Commit/push yalnız kullanıcı isteyince.

## 3. Alınmış kararlar
| Konu | Karar |
|---|---|
| Bölme | Resmi bölme korunur (ödev kuralı). **Sızıntı:** test metrikleri hem tüm test (743) hem **temiz test** (692) üzerinde raporlanır ("Yol 1"). ⚠️ Kullanıcının açık onayı bekleniyor — bkz. GitHub issue. |
| Yakın-yinelenen eşiği | pHash Hamming ≤ **12** (görsel kalibrasyon: `docs/phash_calibration.csv`). Ölçülen sızıntı bir alt sınır. Val sızıntısı %35. |
| Görüntü düzeyi etiket | Tek sınıf → o sınıf; çok sınıflı → toplam poligon alanı en büyük sınıf (`src/config.py: IMAGE_LABEL_RULE`). 9 çok sınıflı görüntü. |
| Poligon düzeltme | Kendini kesen → `shapely.make_valid` + en büyük parça (12 adet); görüntü dışı → kırpma (4 adet); <3 nokta/sıfır alan → at (0). |
| `train/gray_mold1` | "Blossom Blight" etiketli ama görsel olarak kurşuni küf → muhtemel etiket hatası; **değiştirilmedi**, raporda not. |
| "Test Disease Severity Level" klasörü | test/'in birebir kopyası (743/743) → kullanılmaz. |
| COCO kategori kimliği | Roboflow düzeni: 0 = üst kategori "diseases", sınıflar 1..7 (= YOLO kimliği + 1). |
| Sınıf sırası | Alfabetik, `src/config.py: CLASSES`. Gruplar: yaprak (Angular Leafspot, Leaf Spot, PM Leaf), meyve (Anthracnose, Gray Mold, PM Fruit), çiçek (Blossom Blight). |
| Lisans | Kaggle'da "Unknown"; yazarlar kullanım izni veriyor, atıf istiyor. Görüntüler çok sayıda web kaynağından (alamy.com dahil) → **Roboflow projesi private**. CC BY 4.0 makalenin lisansı, verinin değil. |
| Roboflow | Ana dönüşüm Python; Roboflow bağımsız çapraz kontrol (zorunlu, `docs/PLAN.md` Aşama 1 adım 8). Workspace `topraks-workspace-k39ij`, Instance Segmentation projesi (Private, henüz boş). |
| Model ailesi | **YOLO26** (Ultralytics 8.4.210'daki en güncel aile; plan "o tarihteki güncel aile" diyordu): CLS `yolo26n-cls`, `yolo26s-cls`; SEG için `yolo26n-seg`, `yolo26s-seg` mevcut. |
| CLS deney düzeni | imgsz 224, 100 epoch + patience 20, batch 64, optimizer auto, seed 0/1/2 → 6 ana koşu; ablasyon: nano imgsz 384 × 3 seed. Model seçimi val doğruluğuyla; en kötü 10 tahmin val ile seçilen koşudan. Karmaşıklık matrisi 3 seed toplamı. |
| Duman testi | Yerelde GPU yokken `SMOKE` modu (1 epoch, verinin %5'i) → çıktılar `runs/cls_duman/`; asıl sonuçlara/panele/rapora **asla** karışmaz. |
| Colab | Ücretsiz plan yeterli; sekme açık kalmalı. Eğitimler her epoch Drive'a checkpoint yazacak ve 2–3 saatlik parçalara bölünecek. Ücretli planda da modeller **T4**'te koşulacak (adil hız ölçümü). |

## 4. Depo yapısı
```
src/            config.py (yollar, sınıflar, eşikler) · dataset.py (LabelMe okuma/düzeltme/istatistik)
                convert.py (CLS / YOLO-seg / COCO + geri-dönüş doğrulaması) · dedup.py (MD5/pHash, sızıntı) · pipeline.py (Aşama 1 hattı)
                metrics.py (P/R/F1, karmaşıklık + yaprak/meyve/çiçek grupları, bootstrap GA, McNemar) · cls.py (Aşama 2 eğitim/tahmin/değerlendirme)
scripts/        prepare_data.py · build_notebook_01.py · build_notebook_02.py · _nb_common.py (notebook'ların ortak Colab kurulum hücresi)
notebooks/      01_veri_hazirlama.ipynb (yerelde çalıştı, çıktılı) · 02_siniflandirma.ipynb (duman testi geçti, Colab'da eğitilecek) · 03 yazılacak
weights/        GIT'TE YOK — ön-eğitimli ağırlıklar (otomatik iner)
app/            Streamlit paneli (Ana_Sayfa.py + pages/) — data/derived/ ve docs/durum.yaml'ı okur
tests/          sentetik verili pytest testleri (8 test)
docs/           PLAN.md · veri_bulgulari.md · durum.yaml (panelin ilerleme kaynağı) · phash_calibration.csv · hocanın PDF'i
rapor/figures/  rapora girecek şekiller (01 notebook'u üretir)
data/           GIT'TE YOK — raw/ (Kaggle), processed/ (cls, yoloseg, coco), derived/ (özet, CSV'ler) → scripts/prepare_data.py yeniden üretir
runs/           GIT'TE YOK — eğitim çıktıları (Colab → Drive → buraya)
```
Notebook'lar elle değil `scripts/build_notebook_*.py` ile üretilir. Panel ilerleme durumunu `docs/durum.yaml`'dan okur — aşama bitince güncelle.

## 5. Yeni bir Mac'te kurulum (ör. Mac mini) — sırayla
1. **Homebrew** yoksa: https://brew.sh adresindeki tek satırlık komut.
2. **Python 3.13** (Colab ile aynı ana sürüm; Mac'in varsayılan Python'u çok yeni olabilir):
   `brew install python@3.13 gh git`
3. **GitHub girişi:** `gh auth login` → GitHub.com → HTTPS → tarayıcıyla giriş.
4. **Depoyu indir:** `gh repo clone toprakozkale/derin-ogrenme-odev13 ~/Desktop/derin_ogrenme_yl_odev` ve `cd ~/Desktop/derin_ogrenme_yl_odev`
5. **Sanal ortam:**
   `/opt/homebrew/opt/python@3.13/bin/python3.13 -m venv .venv && .venv/bin/pip install -r requirements-local.txt`
6. **Jupyter çekirdeği (VS Code için):** `.venv/bin/python -m ipykernel install --user --name odev13 --display-name "Python 3.13 (odev13)"`
7. **Kaggle anahtarı:** eski Mac'teki `~/.kaggle/kaggle.json` dosyasını **AirDrop** ile gönder (asla git'e/sohbete koyma), sonra:
   `mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json`
   (Kaybolursa kaggle.com → profil → Settings → API → Create New Token ile yenisi alınır; eskisi geçersiz olur.)
8. **Veriyi indir ve hazırla (~1 dk):** `.venv/bin/python scripts/prepare_data.py` → `data/` oluşur.
9. **Testler:** `.venv/bin/python -m pytest -q tests` → 8 passed beklenir.
10. **Panel:** `.venv/bin/streamlit run app/Ana_Sayfa.py` → http://localhost:8501
11. Mac uyumasın (uzun Colab eğitimlerinde): `caffeinate -dimsu` (Terminal'de açık kalsın, Ctrl+C ile kapanır; kapak kapatılmaz).

**İki Mac arasında çalışma düzeni:** başlarken `git pull`, bitirirken commit + `git push`. `data/` her makinede komutla yeniden üretilir.

## 6. Hatırlatıcılar — dış servisler
**Kaggle**
- Anahtar: `~/.kaggle/kaggle.json` (izin 600). Colab'da: sol menü 🔑 **Secrets** → `KAGGLE_USERNAME` ve `KAGGLE_KEY` ekle, "Notebook access" aç.
- Zip SHA-256: `e728dd56eaab4e11e64d46da77b06eb377beb264c53d0a32c1eaf44d051de525` (sürüm 1).

**Google Colab**
- Çalışma zamanı → Çalışma zamanı türünü değiştir → **T4 GPU** → Kaydet. Kontrol: `!nvidia-smi`.
- Colab ortamı (7 Kas 2026): Tesla T4 15 GB, sürücü 580.82.07, CUDA 13.0, **Python 3.13.16** (Eylül 2026 imajı).
- Ücretsiz plan: sekme/Mac açık kalmalı, oturum ≤ ~12 saat, arada "Hâlâ orada mısınız?" çıkabilir. Pro 9,99 $/ay, Pro+ 49,99 $/ay (arka planda çalışma).
- Proje kodu Colab'a nasıl gelir? Notebook'ların ilk hücresi (`scripts/_nb_common.py`) Drive'ı bağlar; `MyDrive/odev13` yoksa depoyu
  GitHub'dan **klonlar**, varsa `git pull` yapar. Depo private → Colab 🔑 Secrets'a **`GITHUB_TOKEN`** eklenmeli: github.com →
  Settings → Developer settings → Fine-grained tokens → yalnız `derin-ogrenme-odev13` deposu, Contents: **Read-only**.
  Token yalnız istek başlığında kullanılır, Drive'daki .git/config'e yazılmaz. Bu kod yolu henüz gerçek Colab'da denenmedi (issue #3).
- Colab Secrets toplam: `KAGGLE_USERNAME`, `KAGGLE_KEY`, `GITHUB_TOKEN` (her biri için "Notebook access" açık).
- Colab'da **kod + sonuçlar (`runs/`) Drive'da** (`MyDrive/odev13`, kalıcı), **veri Colab'ın yerel diskinde** (`/content/odev13_data`,
  `ODEV13_DATA`): Drive'a binlerce küçük dosya yazmak çok yavaş. Veri her yeni oturumda notebook tarafından ~1–2 dk'da yeniden üretilir.
- 01 notebook'u GPU istemez → GPU kotasını harcamamak için onu CPU çalışma zamanında çalıştır; T4 yalnız 02/03 eğitimleri için.
- Hocanın da çalıştırabileceği Colab linki kararı açık — bkz. issue #4.
- Sonuçlar Drive'a yazılır → yerelde `runs/` altına kopyalanınca panel dolar.

**Roboflow**
- https://app.roboflow.com/topraks-workspace-k39ij — proje **Private** kalmalı. **Upgrade'e basma, kart girme.**
- Yüklemeden önce Settings → Plan'a bak (private proje deneme süresine bağlı mı?).
- Dışa aktarırken ön işleme ve çoğaltma **kapalı**, bölme bizimkiyle aynı.

**Teslim**
- E-posta: tekinler.cs@gmail.com · PDF rapor + Colab linki (gizli pencerede açılıyor mu kontrol et) + sunum.
- Raporda kısa "Yapay zekâ araçlarının kullanımı" beyanı önerildi.

## 7. Durum ve eksikler
Güncel ilerleme: panelin Genel Bakış sayfası (`docs/durum.yaml`). Açık işlerin her biri GitHub issue olarak duruyor:
`gh issue list` · depo https://github.com/toprakozkale/derin-ogrenme-odev13 (private) · kilometre taşları "1. rapor (CLS)" ve "2. rapor (SEG)".
Bir iş bitince ilgili issue'yu kapat ve `docs/durum.yaml`'ı güncelle.

| # | İş | Etiket |
|---|---|---|
| 1 | [Karar] Sızıntı: resmi bölme + temiz test onayı | karar |
| 2 | Roboflow çapraz kontrolü | aşama-1 |
| 3 | 01 notebook'u Colab'da çalıştır | aşama-1, colab |
| 4 | [Karar] Hocanın çalıştırabileceği Colab linki (private depo) | karar, colab |
| 5–8 | 02 notebook · Colab CLS eğitimleri · CLS analizi · panel CLS sayfası | aşama-2 |
| 9 | 1. rapor — 24 Kasım | rapor |
| 10–13 | 03 notebook · Colab SEG eğitimleri · SEG analizi · panel SEG sayfası | aşama-3 |
| 14 | 2. rapor + sunum — 8 Aralık | rapor |
| 15 | Mac mini kurulumu | kurulum |

Son durum (11 Kasım 2026): Aşama 0 ve Aşama 1'in kodu bitti. Aşama 2: `02_siniflandirma` notebook'u yazıldı, yerel duman testi
hatasız (issue #5 kapandı). Sırada: Colab Secrets (KAGGLE + GITHUB_TOKEN) → 01'i Colab'da dene (#3) → 02 ile gerçek eğitim (#6).
Hâlâ açık: Roboflow (#2), sızıntı onayı (#1), Colab-link kararı (#4), panel CLS sayfası (#8; sonuç biçimi artık belli: `runs/cls/*.csv`).

## 8. Sık komutlar
```bash
.venv/bin/python scripts/prepare_data.py          # Aşama 1'i baştan üret (data/processed, data/derived)
.venv/bin/python scripts/build_notebook_01.py     # 01 notebook'unu yeniden üret (02 için build_notebook_02.py)
.venv/bin/python -m pytest -q tests               # testler (13)
cd notebooks && ODEV13_SMOKE=1 ../.venv/bin/jupyter nbconvert --to notebook --execute 02_siniflandirma.ipynb --output-dir /tmp   # 02 duman testi
.venv/bin/streamlit run app/Ana_Sayfa.py          # panel → http://localhost:8501
```
