"""1. rapor (CLS) taslağını üretir: rapor/rapor1_cls.docx

Türkçe, iki sütunlu IEEE düzeni (A4). Rapordaki HER SAYI bu betikte doğrudan
runs/cls/*.csv, runs/cls/preds/*.csv, data/derived/summary.json ve docs/roboflow/roboflow_summary.json
dosyalarından okunur — elle sayı yazılmaz. Teslime kadar henüz belli olmayan bilgiler
(ör. Colab linki) sarı vurgulu yer tutucu olarak bırakılır.

Çalıştırma:  .venv/bin/python scripts/build_rapor1.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _rapor_ortak import *  # noqa: F401,F403,E402

CLS = ROOT / "runs" / "cls"
FIG = ROOT / "rapor" / "figures"
OUT = ROOT / "rapor" / "rapor1_cls.docx"
TESLIM = "24 Kasım 2026"

# ---------------------------------------------------------------- veriler
S = json.load(open(ROOT / "data/derived/summary.json"))
RF = json.load(open(ROOT / "docs/roboflow/roboflow_summary.json"))
CS = json.load(open(CLS / "summary.json"))
ENV = CS["env"]
res = pd.read_csv(CLS / "results.csv")
pc = pd.read_csv(CLS / "per_class.csv")
eff = pd.read_csv(CLS / "efficiency.csv")
mcn = pd.read_csv(CLS / "mcnemar.csv")
boot = pd.read_csv(CLS / "bootstrap.csv")
w10 = pd.read_csv(CLS / "worst10.csv")

GRUPLAR = {  # (deney, boyut, imgsz) → okunur ad
    "n224": ("ana", "n", 224, "YOLO26n-cls"),
    "s224": ("ana", "s", 224, "YOLO26s-cls"),
    "n384": ("abl384", "n", 384, "YOLO26n-cls"),
}


def kosular(key):
    deney, boyut, imgsz, _ = GRUPLAR[key]
    return sorted(res[(res.deney == deney) & (res.boyut == boyut) & (res.imgsz == imgsz)].kosu.unique())


def agg(key, alt, col):
    deney, boyut, imgsz, _ = GRUPLAR[key]
    v = res[(res.deney == deney) & (res.boyut == boyut) & (res.imgsz == imgsz) & (res.alt_kume == alt)][col]
    assert len(v) == 3, (key, alt, len(v))
    return float(v.mean()), float(v.std(ddof=1))


def preds(kosu):
    return pd.read_csv(CLS / "preds" / f"{kosu}_test.csv")


def cm_toplam(key):
    cm = np.zeros((7, 7))
    for k in kosular(key):
        p = preds(k)
        cm += M.confusion(p.y_true.to_numpy(), p.y_pred.to_numpy())
    return cm


CM = {k: cm_toplam(k) for k in GRUPLAR}
CI = {c: i for i, c in enumerate(C.CLASSES)}


def oran(key, gercek, tahmin):
    cm = CM[key]
    return cm[CI[gercek], CI[tahmin]] / cm[CI[gercek]].sum()


def pcm(key, sinif, col, alt="tum_test"):
    ks = kosular(key)
    v = pc[(pc.kosu.isin(ks)) & (pc.alt_kume == alt) & (pc.sinif == sinif)][col]
    assert len(v) == 3
    return float(v.mean())


def egitim_bilgisi(kosu):
    r = pd.read_csv(CLS / "train" / kosu / "results.csv")
    r.columns = [c.strip() for c in r.columns]
    b = int(r["metrics/accuracy_top1"].idxmax())
    return dict(epoch=len(r), en_iyi=int(r.epoch[b]), sure_dk=float(r.time.iloc[-1]) / 60,
                lr_max=float(r["lr/pg0"].max()))


EG = {k: egitim_bilgisi(k) for k in res.kosu.unique()}
SECILEN = CS["secilen_kosu_val"]
sec_tum = res[(res.kosu == SECILEN) & (res.alt_kume == "tum_test")].iloc[0]
sec_tmz = res[(res.kosu == SECILEN) & (res.alt_kume == "temiz_test")].iloc[0]
sec_boot = boot[(boot["koşu"] == SECILEN) & (boot["alt_küme"] == "tum_test")].iloc[0]
EFF = {r.kosu: r for r in eff.itertuples()}
EFF_K = {"n224": "ana_yolo26n-cls_im224_s0", "s224": "ana_yolo26s-cls_im224_s0", "n384": "abl384_yolo26n-cls_im384_s0"}

st = S["stats"]
dup = S["duplicates"]
rt = S["roundtrip"]
n_multi = sum(st[s]["n_multiclass_images"] for s in ("train", "val", "test"))

# ================================================================ BAŞLIK (tek sütun)
p = para([("Çilek Bitkisinde Hastalık Tanıma: YOLO26 Sınıflandırma Modelleriyle "
           "Model Boyutu, Çözünürlük ve Meyve/Yaprak Karışmasının İncelenmesi", {})],
         align=WD_ALIGN_PARAGRAPH.CENTER, girinti=False, size=20, after=8)
p = para([(YAZAR, {})], align=WD_ALIGN_PARAGRAPH.CENTER, girinti=False, size=11, after=1)
p = para([(f"Öğrenci No: {OGRENCI_NO} · {KURUM}", {"italic": True})],
         align=WD_ALIGN_PARAGRAPH.CENTER, girinti=False, size=9, after=1)
p = para([(f"{DERS} dersi · Dersin sorumlusu: {HOCA} · 1. Rapor (Sınıflandırma) · {TESLIM}", {})],
         align=WD_ALIGN_PARAGRAPH.CENTER, girinti=False, size=9, after=10)

yeni_bolum(2)

# ================================================================ ÖZET
nA, nS = agg("n224", "tum_test", "accuracy"), agg("s224", "tum_test", "accuracy")
n3 = agg("n384", "tum_test", "accuracy")
nAt, nSt, n3t = (agg(k, "temiz_test", "accuracy") for k in ("n224", "s224", "n384"))
nF, sF, n3F = (agg(k, "tum_test", "macro_f1") for k in ("n224", "s224", "n384"))
gk = pd.read_csv(CLS / "grup_karisma.csv", index_col=0)
gk_n = float(gk.loc["gruplar_arasi_toplam", "yolo26n-cls"])
gk_s = float(gk.loc["gruplar_arasi_toplam", "yolo26s-cls"])
pmf_gm = oran("n224", "Powdery Mildew Fruit", "Gray Mold")

p = para([("Özet", {"bold": True, "italic": True}),
          ("—Bu çalışmada çilek bitkisindeki yedi hastalık, Afzaal ve ark. [1] veri kümesinin resmî bölmesi "
           f"(eğitim {n(st['train']['n_images'])}, doğrulama {n(st['val']['n_images'])}, test {n(st['test']['n_images'])} görüntü) "
           "kullanılarak görüntü düzeyinde sınıflandırılmıştır. LabelMe çokgen etiketleri görüntü etiketine çevrilmiş, "
           "bölmeler arasındaki yakın-yinelenen görüntüler algısal özetle (pHash) tespit edilerek test sonuçları hem tüm test "
           f"kümesinde hem de sızıntıdan arındırılmış “temiz test” kümesinde ({n(dup['test_clean'])} görüntü) raporlanmıştır. "
           "Ultralytics YOLO26 ailesinin nano ve small sınıflandırma modelleri, üçer farklı tohum (seed) ile Colab T4 GPU'da eğitilmiştir. "
           f"Tüm testte doğruluk nano için {ms(*nA)}, small için {ms(*nS)} bulunmuş; "
           "küçük modelden büyük modele geçiş anlamlı bir kazanç sağlamamıştır. Giriş çözünürlüğünü 224'ten 384'e çıkarmak "
           f"nano modelin doğruluğunu {ms(*n3)} değerine ve makro F1'i {ek(f(nF[0]), 'den')} {ek(f(n3F[0]), 'e')} yükseltmiştir. "
           f"Hataların çoğu aynı bitki organı içindedir: meyve külleme örneklerinin {ek(pct(pmf_gm, 0), 'iyelik')} kurşuni küf olarak "
           f"tahmin edilmiş, yaprak/meyve/çiçek grupları arasındaki karışma ise nano için {pct(gk_n)}, small için {pct(gk_s)} kalmıştır.",
           {"bold": True})],
         girinti=False, size=9)
para([("Anahtar kelimeler", {"bold": True, "italic": True}),
      ("—çilek hastalıkları, görüntü sınıflandırma, YOLO26, veri sızıntısı, karmaşıklık matrisi", {"bold": True})],
     girinti=False, size=9, after=4)

# ================================================================ I. GİRİŞ
h1("Giriş")
para("Çilek yetiştiriciliğinde yaprak, meyve ve çiçekte görülen mantar ve bakteri hastalıkları verim ve kalite kaybının "
     "başlıca nedenlerindendir; hastalığın erken ve doğru tanınması ilaçlama kararlarını doğrudan etkiler. Derin öğrenmeye "
     "dayalı görüntü tanıma, uzman gözüne dayanan bu işin sahada telefon kamerasıyla yapılabilmesini sağlayabilir.")
para("Afzaal ve ark. [1], yedi çilek hastalığı için çokgen (poligon) maskeli 2.500 görüntülük bir veri kümesi yayımlamış "
     "ve örnek bölütleme için Mask R-CNN [2] kullanmıştır. Bu raporda aynı veri kümesi ve aynı resmî bölme ile daha basit bir "
     "soru incelenmektedir: görüntünün tümüne bakarak hangi hastalığın bulunduğu ne kadar doğru söylenebilir? Ödevin "
     "ikinci aşamasında aynı veri ile bölütleme (YOLO-seg ve RF-DETR) karşılaştırılacaktır.")
para("Çalışmanın katkıları şunlardır: (i) çokgen etiketlerin görüntü etiketine çevrilmesi için açık bir kural ve etiket "
     "temizliği; (ii) bölmeler arası yakın-yinelenen görüntülerin ölçülmesi ve test sonuçlarının sızıntıdan arındırılmış "
     "alt kümede de verilmesi; (iii) YOLO26 nano/small modellerinin üç tohumla, güven aralıkları ve McNemar testiyle "
     "karşılaştırılması; (iv) meyve ve yaprak hastalıklarının birbirine karışma oranının karmaşıklık matrisiyle incelenmesi.")

# ================================================================ II. VERİ
h1("Veri ve Ön İşleme")
h2("Veri kümesi")
tot_inst = sum(st[s]["n_instances"] for s in ("train", "val", "test"))
para(f"Veri kümesi Kaggle'dan [3] indirilmiştir (sürüm 1; zip dosyasının SHA-256 özeti kayıtlıdır). Tüm görüntüler "
     f"419×419 piksel boyutundadır ve etiketler LabelMe [4] JSON biçiminde, toplam {n(tot_inst)} çokgen olarak verilmiştir. "
     "Yedi sınıf, hastalığın görüldüğü organa göre üç gruba ayrılmıştır: yaprak — köşeli yaprak lekesi (Angular Leafspot), "
     "yaprak lekesi (Leaf Spot), yaprak küllemesi (Powdery Mildew Leaf); meyve — antraknoz (Anthracnose Fruit Rot), kurşuni "
     "küf (Gray Mold), meyve küllemesi (Powdery Mildew Fruit); çiçek — çiçek yanıklığı (Blossom Blight). Ödev kuralı "
     "gereği resmî eğitim/doğrulama/test bölmesi değiştirilmemiştir. Veri kümesindeki “Test Disease Severity Level” "
     f"klasörünün {n(S['severity_check']['identical_to_test'])} görüntüsünün tamamı test klasörünün birebir kopyası "
     "olduğundan ek veri olarak kullanılmamıştır.")

satir = []
for c in sorted(C.CLASSES, key=lambda c: (C.GROUPS.index(C.GROUP_OF[c]), c)):
    satir.append([c, C.GROUP_OF[c]] + [n(st[s]["images_per_label"][c]) for s in ("train", "val", "test")])
satir.append(["Toplam", ""] + [n(st[s]["n_images"]) for s in ("train", "val", "test")])
tablo("Görüntü düzeyi etiketlere göre sınıf dağılımı",
      ["Sınıf", "Grup", "Eğitim", "Doğr.", "Test"], satir, [3.3, 1.3, 1.2, 1.2, 1.2],
      hizalar=["l", "l", "r", "r", "r"], kalin_satirlar=(len(satir) - 1,))

lo = min(st["train"]["images_per_label"].values())
hi = max(st["train"]["images_per_label"].values())
para(f"Tablo I'de görüldüğü gibi sınıflar dengesizdir: eğitim kümesinde en kalabalık sınıf (Leaf Spot, {n(hi)}) en küçük "
     f"sınıfın (Anthracnose Fruit Rot, {n(lo)}) yaklaşık {f(hi / lo, 1)} katıdır. Meyve küllemesi ve antraknoz, hem en az örneğe "
     "sahip hem de görsel olarak birbirine ve kurşuni küfe en çok benzeyen sınıflardır.")

h2("Görüntü düzeyi etiket kuralı")
para("Sınıflandırma her görüntüye tek etiket ister. Görüntüdeki tüm çokgenler aynı sınıftansa o sınıf, birden fazla sınıf "
     f"varsa toplam çokgen alanı en büyük olan sınıf görüntü etiketi olarak alınmıştır. Bu kural yalnız {n_multi} görüntüyü "
     "etkilemektedir (eğitimde 6, testte 3); bunların hepsi aynı karede yaprak ve meyve hastalığı birlikte bulunan "
     "görüntülerdir ve meyve/yaprak karışması tartışması için doğal örneklerdir.")

h2("Etiket temizliği ve biçim dönüşümleri")
lic = S["label_issue_counts"]
para(f"Kendini kesen {lic['kendini_kesen:onarildi']} çokgen geometrik olarak onarılmış (en büyük geçerli parça tutulmuş), "
     f"görüntü dışına taşan {lic['goruntu_disi:kirpildi']} çokgen görüntü sınırına kırpılmıştır; 3'ten az noktalı ya da sıfır "
     "alanlı çokgen bulunmamıştır. Eğitim kümesindeki gray_mold1 görüntüsünün tek çokgeni “Blossom Blight” "
     "etiketli olmasına karşın görsel olarak kurşuni küfe benzemektedir; uzman etiketi değiştirilmemiş, etiket gürültüsü "
     "örneği olarak not edilmiştir.")
para("Çokgenler ikinci aşama için YOLO-seg ve COCO biçimlerine de çevrilmiş, dönüşüm geri çevrilerek doğrulanmıştır: "
     f"{n(rt['n_instances'])} örneğin hepsinde sınıf ve sayı korunmuş, ortalama IoU YOLO-seg için {f(rt['iou_yolo_mean'], 5)}, "
     f"COCO için {f(rt['iou_coco_mean'], 5)} bulunmuştur. Ayrıca veri Roboflow'a [5] yüklenip oradan COCO olarak indirilmiş ve "
     f"bizim dönüşümümüzle örnek örnek karşılaştırılmıştır: {n(RF['ornek_karsilastirilan'])} örneğin ortalama IoU'su "
     f"{ek(f(RF['iou_ortalama'], 5), 'dir')}. Roboflow'un atladığı {RF['atlanan_goruntu']} görüntünün tamamı bölme içi bayt bayt "
     f"kopyalardır (eğitimde {RF['atlanan_bolmeye_gore']['train']}, testte {RF['atlanan_bolmeye_gore']['test']}).")

h2("Yakın-yinelenenler ve veri sızıntısı")
para(f"Bölmeler arasında birebir aynı dosya yoktur; ancak {dup['exact_groups']} bayt kopyası bölmelerin kendi içindedir. "
     "Test içindeki kopyalar, test skorunda aynı görüntünün iki kez sayılması anlamına gelir. Daha önemlisi, aynı sahnenin "
     "hafif kaydırılmış ya da kırpılmış kareleri farklı bölmelere düşmüştür. Bunu ölçmek için her görüntünün 64 bitlik "
     f"algısal özeti (pHash) [6] hesaplanmış ve Hamming uzaklığı ≤ {dup['phash_threshold']} olan çiftler, eşiğin görsel "
     "kalibrasyonuyla, “aynı sahne” kabul edilmiştir. Bu ölçüte göre test görüntülerinin "
     f"{ek(n(dup['test_leaked']), 'belirtme')} ({pct(dup['test_leaked'] / st['test']['n_images'])}) ve doğrulama görüntülerinin "
     f"{ek(n(dup['val_leaked']), 'belirtme')} ({pct(dup['val_leaked'] / st['val']['n_images'])}) eğitimde bir benzerine sahiptir. Eşiğin "
     "üzerinde de gerçek yinelenenler görüldüğü için bu değerler bir alt sınırdır.")
para(f"Resmî bölme korunmuş, bunun yerine test metrikleri hem tüm testte ({n(st['test']['n_images'])}) hem de eğitim veya "
     f"doğrulamada benzeri olmayan temiz testte ({n(dup['test_clean'])}) verilmiştir. Doğrulama kümesindeki yüksek sızıntı "
     "nedeniyle doğrulama skorlarının iyimser olduğu kabul edilmiştir; bölme listeleri SHA-256 özetleriyle kayıt altındadır.")

# ================================================================ III. YÖNTEM
h1("Yöntem")
h2("Modeller")
p_n = EFF[EFF_K["n224"]].parametre
p_s = EFF[EFF_K["s224"]].parametre
para(f"Ödevin istediği Ultralytics YOLO sınıflandırma modelleri [7] kullanılmıştır. Deneylerin yapıldığı tarihteki en güncel "
     f"aile olan YOLO26'nın iki boyu seçilmiştir: yolo26n-cls ({f(p_n / 1e6, 2)} M parametre) ve yolo26s-cls "
     f"({f(p_s / 1e6, 2)} M parametre). Her iki model de ImageNet [8] üzerinde ön eğitimli ağırlıklarla başlatılmış ve "
     "son katmanı yedi sınıfa uyarlanarak tüm ağ ince ayarla eğitilmiştir.")

h2("Eğitim düzeni")
ea = ENV["egitim_ayarlari"]
lr_max = EG[SECILEN]["lr_max"]
tablo("Eğitim ayarları",
      ["Ayar", "Değer"],
      [["Giriş boyutu", "224×224 (ablasyon: 384×384)"],
       ["En çok epoch / erken durdurma", f"{ea['epochs']} / {ea['patience']} epoch iyileşme yoksa"],
       ["Yığın (batch) boyutu", str(ea["batch"])],
       ["Optimizasyon", f"Ultralytics “auto”; en yüksek öğr. oranı {f(lr_max * 1e4, 2)}×10⁻⁴"],
       ["Veri artırma", "RandAugment, yatay çevirme (0,5), HSV, rastgele silme (0,4)"],
       ["Tohum (seed)", "0, 1, 2 (deterministik mod)"],
       ["Donanım", f"Google Colab, {ENV['cihaz']}"]],
      [3.1, 5.1], hizalar=["l", "l"])
ep = {k: [EG[x]["epoch"] for x in kosular(k)] for k in GRUPLAR}
dk = {k: [EG[x]["sure_dk"] for x in kosular(k)] for k in GRUPLAR}
para("Ana deney iki model boyu × üç tohum olmak üzere altı koşudan oluşur. Küçük lezyonlarda çözünürlüğün etkisini görmek "
     "için nano model ayrıca 384×384 girişle üç tohumla eğitilmiştir (ablasyon). Veri artırma için Ultralytics'in "
     "sınıflandırma varsayılanları (RandAugment [9] ve rastgele silme [10] dahil) kullanılmıştır. Hiçbir koşu 100 epoch'a "
     f"ulaşmamış, tamamı erken durdurmayla {min(min(v) for v in ep.values())}–{max(max(v) for v in ep.values())}. epoch "
     f"arasında durmuştur. Bir koşunun T4'teki süresi nano-224 için {f(min(dk['n224']), 0)}–{f(max(dk['n224']), 0)} dk, "
     f"small için {f(min(dk['s224']), 0)}–{f(max(dk['s224']), 0)} dk, nano-384 için {f(min(dk['n384']), 0)}–"
     f"{f(max(dk['n384']), 0)} dk'dır. Her koşuda doğrulama doğruluğu en yüksek olan ağırlık saklanmıştır.")

h2("Değerlendirme")
para("Test kümesi yalnızca en sonda, bir kez kullanılmıştır; model seçimi doğrulama doğruluğuyla yapılmıştır. "
     "Tüm metrikler Ultralytics'ten bağımsız kendi değerlendirme kodumuzla, kaydedilmiş tahminlerden hesaplanmıştır: "
     "doğruluk, sınıf bazında kesinlik (P), duyarlılık (R) ve F1, makro ve ağırlıklı F1, 7×7 karmaşıklık matrisi ve bunun "
     "yaprak/meyve/çiçek gruplarına indirgenmiş 3×3 hâli. Tohumlar arası değişkenlik ortalama ± standart sapma ile, tek "
     "koşu içindeki belirsizlik 2.000 tekrarlı bootstrap [11] ile %95 güven aralığı olarak verilmiştir. Aynı tohumla "
     "eğitilmiş nano ve small modelin test tahminleri McNemar testi [12] ile karşılaştırılmıştır. Hız, T4'te yığın boyu 1 ile "
     "test görüntülerinin tamamı üzerinde ölçülmüştür.")

# ================================================================ IV. SONUÇLAR
h1("Deneysel Sonuçlar")
h2("Genel başarım")
satir = []
for k in ("n224", "s224", "n384"):
    _, _, imgsz, ad = GRUPLAR[k]
    e = EFF[EFF_K[k]]
    satir.append([f"{GRUPLAR[k][1]}-{imgsz}", f(e.parametre / 1e6, 2),
                  ms(*agg(k, "tum_test", "accuracy")), ms(*agg(k, "temiz_test", "accuracy")),
                  ms(*agg(k, "tum_test", "macro_f1")), f"{f(e.ms_cikarim, 1)} / {f(e.ms_toplam_duvar, 1)}"])
tablo("Test sonuçları (üç tohum ortalaması ± std)",
      ["Model", "Param. (M)", "Doğruluk (tüm)", "Doğruluk (temiz)", "Makro F1 (tüm)", "ms: model / toplam"],
      satir, [0.95, 1.0, 1.65, 1.65, 1.65, 1.3], yazi=7)
para("n = YOLO26n-cls, s = YOLO26s-cls; sayı giriş boyutudur. Süreler T4'te, yığın boyu 1 ile görüntü başına; \u201ctoplam\u201d ön ve son işlemeyi de içerir.",
     girinti=False, size=8, after=4)
fark = [100 * (agg(k, 'tum_test', 'accuracy')[0] - agg(k, 'temiz_test', 'accuracy')[0]) for k in GRUPLAR]
para(f"Tablo III ana sonuçları özetler. Ortalama doğrulukta nano model ({f(nA[0])}) small modelden ({f(nS[0])}) geride "
     f"değildir; makro F1'de de durum aynıdır ({f(nF[0])} ve {f(sF[0])}). Nano modelin tohumlar arası değişkenliği daha "
     f"büyüktür (std {f(nA[1])}), yani tek bir koşuya bakarak iki model arasında hüküm vermek yanıltıcı olabilir. Temiz test "
     f"skorları tüm test skorlarından yalnızca {f(min(fark), 1)}–{f(max(fark), 1)} puan düşüktür; "
     "test sızıntısının sonuçları belirgin biçimde şişirmediği görülmektedir.")
mc = {int(r.seed): r for r in mcn.itertuples()}
para(f"Aynı tohumla eğitilen nano ve small modellerin test tahminleri McNemar testiyle karşılaştırıldığında tohum 0 ve 2'de "
     f"fark yoktur (yalnız birinin doğru bildiği görüntü sayıları {mc[0].yalniz_A_dogru}/{mc[0].yalniz_B_dogru} ve "
     f"{mc[2].yalniz_A_dogru}/{mc[2].yalniz_B_dogru}, p = {f(mc[0].p_degeri, 2)}); tohum 1'de fark nano lehinedir "
     f"({mc[1].yalniz_A_dogru}/{mc[1].yalniz_B_dogru}, p = {f(mc[1].p_degeri, 4)}). Bu veri boyutunda {f(p_s / p_n, 1)} kat "
     "daha fazla parametre başarımı artırmamaktadır.")
para([f"Doğrulama doğruluğuyla seçilen ana model nano-224, tohum 1 koşusu (doğrulama {f(sec_tum.val_acc_en_iyi)}) "
      f"olmuştur. Bu modelin test doğruluğu {ga(sec_boot['doğruluk'])} "
      f"(%95 GA), temiz testte {f(sec_tmz.accuracy)}, makro F1'i {ek(f(sec_tum.macro_f1), 'dir')}. Doğrulama ile test arasındaki "
      f"yaklaşık {f(100 * (sec_tum.val_acc_en_iyi - sec_tum.accuracy), 0)} puanlık fark, doğrulama sızıntısının yanında "
      "bölmeler arası dağılım farkına da işaret eder: test görüntülerinde görüntü başına daha az çokgen (ortalama "
      f"{f(st['test']['instances_per_image']['mean'], 2)}, eğitimde {f(st['train']['instances_per_image']['mean'], 2)}) ve "
      f"daha büyük lezyonlar (ortanca alan oranı {f(st['test']['instance_area_ratio']['median'], 3)}, eğitimde "
      f"{f(st['train']['instance_area_ratio']['median'], 3)}) vardır."])

h2("Sınıf bazında başarım")
satir = []
for c in sorted(C.CLASSES, key=lambda c: (C.GROUPS.index(C.GROUP_OF[c]), c)):
    sup = int(pc[(pc.sinif == c) & (pc.alt_kume == "tum_test")].support.iloc[0])
    satir.append([c, f(pcm("n224", c, "precision"), 2), f(pcm("n224", c, "recall"), 2), f(pcm("n224", c, "f1"), 2),
                  f(pcm("s224", c, "f1"), 2), f(pcm("n384", c, "f1"), 2), n(sup)])
tablo("Sınıf bazında test sonuçları (tüm test, üç tohum ortalaması)",
      ["Sınıf", "n P", "n R", "n F1", "s F1", "n384 F1", "Destek"],
      satir, [2.6, 0.75, 0.75, 0.8, 0.8, 1.15, 1.05], hizalar=["l", "c", "c", "c", "c", "c", "r"], yazi=7)
pmfF = {k: pcm(k, "Powdery Mildew Fruit", "f1") for k in GRUPLAR}
antF = {k: pcm(k, "Anthracnose Fruit Rot", "f1") for k in GRUPLAR}
para(f"Tablo IV'e göre çiçek yanıklığı ve yaprak küllemesi neredeyse kusursuz tanınmaktadır (F1 ≥ "
     f"{f(min(pcm('n224', 'Blossom Blight', 'f1'), pcm('n224', 'Powdery Mildew Leaf', 'f1')), 2)}). En zor iki sınıf meyve "
     f"hastalıklarıdır: meyve küllemesi (nano F1 {f(pmfF['n224'], 2)}, small {f(pmfF['s224'], 2)}) ve antraknoz "
     f"(nano {f(antF['n224'], 2)}, small {f(antF['s224'], 2)}). Meyve küllemesinin düşük duyarlılığı "
     f"(R = {f(pcm('n224', 'Powdery Mildew Fruit', 'recall'), 2)}) ile kurşuni küfün düşük kesinliği "
     f"(P = {f(pcm('n224', 'Gray Mold', 'precision'), 2)}) aynı hatanın iki yüzüdür: meyve küllemesi örnekleri kurşuni küf "
     "sınıfına çekilmektedir (Bölüm V).")

h2("Çözünürlük ablasyonu ve verimlilik")
para(f"Nano modelin girişini 224'ten 384'e çıkarmak ortalama doğruluğu {ek(f(nA[0]), 'den')} {ek(f(n3[0]), 'e')}, makro F1'i "
     f"{ek(f(nF[0]), 'den')} {ek(f(n3F[0]), 'e')} yükseltmiş, tohumlar arası değişkenliği de azaltmıştır (std {f(n3[1])}). En büyük "
     f"kazanç meyve küllemesindedir (F1 {f(pmfF['n224'], 2)} → {f(pmfF['n384'], 2)}); bu sınıfta beyaz, ince toz görünümlü "
     "miselin ayrıntısı düşük çözünürlükte kaybolmaktadır. Bedeli küçüktür: T4'te modelin kendi çıkarım süresi "
     f"{f(EFF[EFF_K['n224']].ms_cikarim, 1)} ms'den {f(EFF[EFF_K['n384']].ms_cikarim, 1)} ms'ye pratikte değişmemiş, ön işleme "
     f"dahil toplam süre {f(EFF[EFF_K['n224']].ms_toplam_duvar, 1)} ms'den {f(EFF[EFF_K['n384']].ms_toplam_duvar, 1)} ms'ye "
     f"çıkmıştır. Small model ise nanodan daha yavaştır ({f(EFF[EFF_K['s224']].ms_toplam_duvar, 1)} ms) ama daha iyi değildir. "
     "Ablasyon koşuları ana model seçimine katılmamış, yalnız çözünürlüğün etkisini göstermek için raporlanmıştır.")

# ================================================================ V. HATA ANALİZİ
h1("Hata Analizi ve Meyve/Yaprak Karışması")
h2("Karmaşıklık matrisi")
sekil(FIG / "cls_karmasiklik_n.png",
      "YOLO26n-cls (224) karmaşıklık matrisi; üç tohumun test tahminleri toplanmış, satırlar gerçek sınıfa göre normalize "
      "edilmiştir (parantez içinde görüntü sayısı).", SUTUN - 0.1)
ls_al = oran("n224", "Leaf Spot", "Angular Leafspot")
ant_gm = oran("n224", "Anthracnose Fruit Rot", "Gray Mold")
pmf_ant = oran("n224", "Powdery Mildew Fruit", "Anthracnose Fruit Rot")
para(f"Şekil 1'deki karmaşıklık matrisinde en büyük hata hücresi meyve küllemesi → kurşuni küf hücresidir: meyve "
     f"küllemesi görüntülerinin {ek(pct(pmf_gm, 0), 'iyelik')} kurşuni küf, {ek(pct(pmf_ant, 0), 'iyelik')} antraknoz olarak tahmin edilmiş, "
     f"yalnız {ek(pct(oran('n224', 'Powdery Mildew Fruit', 'Powdery Mildew Fruit'), 0), 'iyelik')} doğru bilinmiştir. Antraknozun da "
     f"{ek(pct(ant_gm, 0), 'iyelik')} kurşuni küfe gitmektedir. Üç meyve hastalığı da meyve yüzeyinde beyazımsı/grimsi örtü veya "
     "kahverengi çürük doku olarak görünür ve küçük sınıflardır; model belirsiz durumda en kalabalık meyve sınıfı olan "
     f"kurşuni küfü seçme eğilimindedir. Yaprak grubunda belirgin karışma Leaf Spot → Angular Leafspot yönündedir ({pct(ls_al, 0)}).")

h2("Gruplar arası karışma (Dikkat maddesi)")
satir = []
for key, ad in (("n224", "nano"), ("s224", "small")):
    g = M.group_confusion(CM[key]).to_numpy()
    for i, a in enumerate(C.GROUPS):
        top = g[i].sum()
        satir.append([ad if i == 0 else "", a] + [f"{n(g[i, j])} ({pct(g[i, j] / top)})" for j in range(3)])
tablo("Organ grupları arası karmaşıklık (üç tohum toplamı, satır yüzdesi)",
      ["Model", "Gerçek", "→ yaprak", "→ meyve", "→ çiçek"], satir, [1.0, 1.2, 2.0, 2.0, 2.0],
      hizalar=["l", "l", "r", "r", "r"])
y_m_n = float(gk.loc["yaprak->meyve", "yolo26n-cls"])
y_m_s = float(gk.loc["yaprak->meyve", "yolo26s-cls"])
m_y_n = float(gk.loc["meyve->yaprak", "yolo26n-cls"])
para(f"Hocanın “Dikkat” maddesindeki soruyu yanıtlamak için 7×7 matris yaprak/meyve/çiçek gruplarına "
     f"indirgenmiştir (Tablo V). Gruplar arası toplam karışma nano için {pct(gk_n)}, small için {ek(pct(gk_s), 'dir')}. Karışma "
     f"asimetriktir: yaprak görüntülerinin {ek(pct(y_m_n), 'iyelik')} (small {pct(y_m_s)}) meyve hastalığı olarak tahmin edilirken "
     f"meyve görüntülerinin yalnız {ek(pct(m_y_n), 'iyelik')} yaprak hastalığına gitmektedir; çiçek yanıklığı hiçbir grupla "
     "karışmamaktadır. Olası bir neden, bazı fotoğraflarda yaprak ile meyvenin aynı kadrajda yer alması ve bu "
     "görüntülerde tek etiket kuralının belirsizlik yaratmasıdır. Sonuç olarak model organı (yaprak mı "
     "meyve mi) büyük ölçüde doğru ayırmakta, asıl zorluk aynı organ üzerindeki benzer görünümlü hastalıkları ayırmaktadır.")

h2("En kötü 10 tahmin")
para(f"Seçilen modelin (nano-224, tohum 1) test kümesinde gerçek sınıfa en düşük olasılığı verdiği on görüntü Şekil 2'de "
     "gösterilmiştir. Bu hatalar dört türe ayrılmaktadır:")
grp_arasi = w10[w10.gruplar_arasi.astype(str) == "True"]


def okunur(stem):
    """angular_leafspot412 → Angular Leafspot 412 (dar sütunda uzun dosya adları satırı bozuyor)"""
    import re
    m = re.match(r"(.*?)(\d+)$", stem)
    g = w10.set_index("stem").gercek.get(stem, m.group(1).replace("_", " ").title())
    return f"{g} {m.group(2)}"
madde = [
    ("Meyve hastalıklarının görsel benzerliği. ",
     "On hatanın sekizi meyve hastalığıdır ve yedisi kurşuni küf olarak tahmin edilmiştir. Meyve küllemesi örneklerinde "
     "(148, 160, 164, 137) kırmızı meyvelerin üstündeki beyaz toz örtüsü, kurşuni küfün grimsi miselyumuna çok "
     "benzemekte; model bu görüntülerde ≥ 0,97 güvenle yanılmaktadır."),
    ("Olgunlaşmamış meyvede antraknoz. ",
     "94, 100 ve 101 numaralı antraknoz görüntülerinde meyve beyaz/yeşilimsi (ham) ve lezyon koyu çökük bir lekedir. "
     "Eğitimdeki antraknoz örneklerinin az olması ve ham meyvenin kurşuni küf örneklerindeki soluk dokuya benzemesi "
     "hatayı açıklayabilir. 92 numaralı antraknoz görüntüsü ise modelin kararsız kaldığı (güven 0,54) tek örnektir."),
    ("Yaprak lekeleri arası karışma. ",
     "602 numaralı yaprak lekesi görüntüsündeki mor halkalı küçük beyaz lekeler Angular Leafspot olarak tahmin edilmiştir; iki yaprak lekesi "
     "hastalığı da küçük, dağınık lekeler hâlinde görünür."),
    ("Alışılmadık kadraj. ",
     f"Tek gruplar arası hata ({', '.join(okunur(x) for x in grp_arasi.stem)}) elle tutulan, çok sayıda koyu leke taşıyan bir yapraktır; "
     "model bunu meyve küllemesi olarak tahmin etmiştir; kadrajda el ve çok sayıda başka yaprak da bulunmaktadır."),
]
for bas, metin in madde:
    para([(bas, {"italic": True}), metin])
para("Ayrıca on hata aslında sekiz farklı sahnedir: meyve küllemesi 137 ve 164 testte neredeyse aynı fotoğraftır "
     "(pHash uzaklığı 0), antraknoz 94 ve 100 ise aynı fotoğrafın farklı kırpılmış hâlleridir (uzaklık 16). "
     "Test içindeki bu yinelenenler aynı hatanın skora iki kez yansımasına yol açmaktadır.")

yeni_bolum(1)
sekil(FIG / "cls_en_kotu10.png",
      f"Seçilen modelin (nano-224, tohum 1) test kümesindeki en kötü 10 tahmini: başlıklarda gerçek sınıf, tahmin ve "
      f"tahminin güveni. Yalnız {okunur(grp_arasi.stem.iloc[0])} organ grupları arası bir hatadır.", METIN_GENISLIK - 0.4)
yeni_bolum(2)

# ================================================================ VI. SINIRLILIKLAR, VII. SONUÇ
h1("Sınırlılıklar")
para("(i) Doğrulama kümesinin en az %35'i eğitimdeki görüntülerin yakın kopyasıdır; erken durdurma ve model seçimi bu "
     "iyimser skorlarla yapılmıştır. (ii) pHash eşiği görsel kalibrasyonla seçilmiştir ve ölçülen sızıntı bir alt "
     f"sınırdır. (iii) Test içindeki {RF['atlanan_bolmeye_gore']['test']} bayt kopyası skora iki kez girmektedir. "
     f"(iv) Az örnekli sınıflarda (testte Anthracnose {n(st['test']['images_per_label']['Anthracnose Fruit Rot'])}, meyve "
     f"küllemesi {n(st['test']['images_per_label']['Powdery Mildew Fruit'])} görüntü) sınıf bazında skorların belirsizliği "
     "yüksektir. (v) Çok sınıflı az sayıdaki görüntüde tek etiket kuralı bilgi kaybına yol açar; ayrıca en az bir etiket "
     "hatası (gray_mold1) bulunmuştur. (vi) Hiperparametre araması yapılmamış, Ultralytics varsayılanları kullanılmıştır.")

h1("Sonuç")
para(f"YOLO26 sınıflandırma modelleri çilek hastalıklarını görüntü düzeyinde yaklaşık {pct(nA[0], 0)} doğrulukla "
     f"tanımaktadır. Sızıntıdan arındırılmış testte skor yalnızca biraz düşmektedir. {f(p_n / 1e6, 2)} M parametreli nano "
     f"model, {f(p_s / 1e6, 2)} M parametreli small modelden kötü değildir ve daha hızlıdır. Çözünürlüğü 384'e çıkarmak, "
     "özellikle meyve küllemesinde belirgin kazanç sağlamaktadır. Hataların çoğu meyve hastalıkları arasındadır; organ "
     "grupları arası karışma küçüktür ve büyük ölçüde yapraktan meyveye doğrudur. İkinci aşamada çokgen maskelerle "
     "yapılacak bölütleme, aynı karede birden çok hastalık bulunan görüntülerde tek etiket kısıtını ortadan kaldıracaktır.")

h1("Yapay zekâ araçlarının kullanımı", numarasiz=True)
para("Bu çalışmada kodların, notebook'ların ve bu rapor taslağının hazırlanmasında Anthropic'in Claude (Claude Code) "
     "yapay zekâ asistanından yararlanılmıştır. Deney tasarımına ilişkin kararlar yazar tarafından verilmiş, eğitimler "
     "yazar tarafından Google Colab'da çalıştırılmıştır. Rapordaki tüm sayılar gerçek çalıştırma çıktılarından bir betikle "
     "aktarılmış; metin yazar tarafından okunup düzeltilmiştir ve içerikten yazar sorumludur.", size=9)

# ================================================================ KAYNAKÇA
h1("Kaynakça", numarasiz=True)
KAYNAK = [
    "U. Afzaal, B. Bhattarai, Y. R. Pandeya ve J. Lee, “An instance segmentation model for strawberry diseases based "
    "on Mask R-CNN,” Sensors, c. 21, sy. 19, s. 6565, 2021, doi: 10.3390/s21196565.",
    "K. He, G. Gkioxari, P. Dollár ve R. Girshick, “Mask R-CNN,” Proc. IEEE Int. Conf. Comput. Vis. (ICCV), "
    "2017, ss. 2961–2969.",
    "U. Afzaal, “Strawberry Disease Detection Dataset,” Kaggle, 2021. [Çevrimiçi]. Erişim: https://www.kaggle.com/"
    "datasets/usmanafzaal/strawberry-disease-detection-dataset",
    "B. C. Russell, A. Torralba, K. P. Murphy ve W. T. Freeman, “LabelMe: A database and web-based tool for image "
    "annotation,” Int. J. Comput. Vis., c. 77, ss. 157–173, 2008.",
    "Roboflow, Inc., “Roboflow,” 2026. [Çevrimiçi]. Erişim: https://roboflow.com",
    "C. Zauner, “Implementation and benchmarking of perceptual image hash functions,” Yüksek lisans tezi, Upper "
    "Austria Univ. of Applied Sciences, Hagenberg, 2010.",
    f"G. Jocher, J. Qiu ve A. Chaurasia, “Ultralytics YOLO,” sürüm {ENV['ultralytics']}, 2026. [Çevrimiçi]. "
    "Erişim: https://github.com/ultralytics/ultralytics",
    "J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li ve L. Fei-Fei, “ImageNet: A large-scale hierarchical image "
    "database,” Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2009, ss. 248–255.",
    "E. D. Cubuk, B. Zoph, J. Shlens ve Q. V. Le, “RandAugment: Practical automated data augmentation with a reduced "
    "search space,” Proc. IEEE/CVF CVPR Workshops, 2020, ss. 702–703.",
    "Z. Zhong, L. Zheng, G. Kang, S. Li ve Y. Yang, “Random erasing data augmentation,” Proc. AAAI Conf. Artif. "
    "Intell., c. 34, sy. 7, ss. 13001–13008, 2020.",
    "B. Efron ve R. J. Tibshirani, An Introduction to the Bootstrap. New York: Chapman & Hall, 1993.",
    "Q. McNemar, “Note on the sampling error of the difference between correlated proportions or percentages,” "
    "Psychometrika, c. 12, sy. 2, ss. 153–157, 1947.",
]
for i, k in enumerate(KAYNAK, 1):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.left_indent = Cm(0.6)
    p.paragraph_format.first_line_indent = Cm(-0.6)
    p.paragraph_format.space_after = Pt(1)
    run(p, f"[{i}]\t{k}", size=8)

# ================================================================ EK
h1("Ek: Kod, ortam ve tekrarlanabilirlik", numarasiz=True)
REPO_URL = "https://github.com/toprakozkale/derin-ogrenme-odev13"
COLAB_URL = "https://colab.research.google.com/github/toprakozkale/derin-ogrenme-odev13/blob/main/notebooks/"
para([("Çalıştırılabilir Colab defteri (sınıflandırma): ", {"bold": True}), COLAB_URL + "02_siniflandirma.ipynb"],
     girinti=False, size=9, align=WD_ALIGN_PARAGRAPH.LEFT)
para([("Veri hazırlama defteri: ", {"bold": True}), COLAB_URL + "01_veri_hazirlama.ipynb"],
     girinti=False, size=9, align=WD_ALIGN_PARAGRAPH.LEFT)
para([("Kod deposu: ", {"bold": True}), REPO_URL],
     girinti=False, size=9, align=WD_ALIGN_PARAGRAPH.LEFT)
para("Defterler gizli anahtar gerektirmez: Colab'da Çalışma zamanı → Çalışma zamanı türünü değiştir → T4 GPU seçilip "
     "Tümünü çalıştır denmesi yeterlidir. Veri Kaggle'dan kendiliğinden iner (yaklaşık 1 dk); sınıflandırma defteri dokuz "
     "eğitimi T4'te yaklaşık 2 saatte yeniden yapar ve bu rapordaki tablo ve şekilleri aynı kodla üretir.",
     girinti=False, size=9)
para([f"Ortam: Python {ENV['python']}, PyTorch {ENV['torch']}, Ultralytics {ENV['ultralytics']}, CUDA {ENV['cuda']}, "
      f"{ENV['cihaz']} (Google Colab, {ENV['tarih'][:10]}). Bölme listelerinin SHA-256 özetleri (ilk 12 karakter): "
      + ", ".join(f"{k} {v[:12]}" for k, v in S["split_list_sha256"].items()) + "."],
     girinti=False, size=9)

OUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(OUT)
print(f"yazıldı: {OUT.relative_to(ROOT)}  (tablo {sayaclar()[0]}, şekil {sayaclar()[1]})")
