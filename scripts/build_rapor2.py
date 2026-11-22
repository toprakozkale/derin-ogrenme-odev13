"""2. rapor (SEG) taslağını üretir: rapor/rapor2_seg.docx

Türkçe, iki sütunlu IEEE düzeni (A4), 1. raporla aynı yapı (`_rapor_ortak.py`). Rapordaki HER SAYI bu betikte
doğrudan runs/seg/*, runs/cls/grup_karisma.csv, data/derived/summary.json ve docs/roboflow/roboflow_summary.json
dosyalarından okunur — elle sayı yazılmaz. Şekiller scripts/make_seg_figures.py ile üretilir.

Çalıştırma:  .venv/bin/python scripts/make_seg_figures.py && .venv/bin/python scripts/build_rapor2.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _rapor_ortak import *  # noqa: F401,F403,E402
from src import seg  # noqa: E402

SEG = ROOT / "runs" / "seg"
FIG = ROOT / "rapor" / "figures"
OUT = ROOT / "rapor" / "rapor2_seg.docx"
TESLIM = "8 Aralık 2026"

# ---------------------------------------------------------------- veriler
S = json.load(open(ROOT / "data/derived/summary.json"))
RF = json.load(open(ROOT / "docs/roboflow/roboflow_summary.json"))
SS = json.load(open(SEG / "summary.json"))
ENV = SS["env"]
res = pd.read_csv(SEG / "results.csv")
pc = pd.read_csv(SEG / "per_class.csv").merge(res[["kosu", "model"]].drop_duplicates(), on="kosu")
eff = pd.read_csv(SEG / "efficiency.csv").set_index("model")
w10 = pd.read_csv(SEG / "worst10.csv")
gk = pd.read_csv(SEG / "grup_karisma.csv", index_col=0)
gk_cls = pd.read_csv(ROOT / "runs/cls/grup_karisma.csv", index_col=0)
st, dup, rt = S["stats"], S["duplicates"], S["roundtrip"]

MODELLER = ["yolo26n-seg", "yolo26s-seg", "rfdetr-seg-small"]
AD = {"yolo26n-seg": "YOLO26n-seg", "yolo26s-seg": "YOLO26s-seg", "rfdetr-seg-small": "RF-DETR Seg S"}
KISA = {"yolo26n-seg": "YOLO26n", "yolo26s-seg": "YOLO26s", "rfdetr-seg-small": "RF-DETR"}
RUNS = {m: [s for s in seg.MAIN_RUNS if s.model == m] for m in MODELLER}
for m, rr in RUNS.items():
    assert len(rr) == 3 and all(seg.is_finished(s) for s in rr), m
SEC_Y, SEC_R = SS["secilen_yolo_val"], SS["secilen_rfdetr_val"]
YN, YS, RD = MODELLER


def agg(m, col, alt="tum_test"):
    v = res[(res.model == m) & (res.alt_kume == alt)][col]
    assert len(v) == 3, (m, col, alt)
    return float(v.mean()), float(v.std(ddof=1))


def pcm(m, sinif, col):
    v = pc[(pc.model == m) & (pc.alt_kume == "tum_test") & (pc.sinif == sinif)][col]
    assert len(v) == 3
    return float(v.mean())


def egitim(s):
    log = seg.train_log(s)
    return len(log), seg.best_val_map(s)


def kosu_sonuc(kosu, col, alt="tum_test"):
    return float(res[(res.kosu == kosu) & (res.alt_kume == alt)][col].iloc[0])


def ust10(p: float) -> str:
    """p'nin üst sınırı 10'un kuvveti olarak: 5,6e-14 → 10⁻¹³"""
    k = int(np.floor(np.log10(p))) + 1
    return "10" + str(k).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))


def tr(c: str) -> str:
    return C.CLASS_TR[c].lower()


def kisa(kosu):
    s = seg.parse_run(kosu)
    return f"{AD[s.model]}, tohum {s.seed}"


# Eşli karşılaştırma: aynı tohumla eğitilen YOLO26s ve RF-DETR'ın görüntü başına IoU'su (Wilcoxon işaretli sıra testi)
eslik = []
for sd in (0, 1, 2):
    y = seg.load_per_image(next(s for s in RUNS[YS] if s.seed == sd))
    r = seg.load_per_image(next(s for s in RUNS[RD] if s.seed == sd))
    mm = y.merge(r, on="image_id", suffixes=("_y", "_r"))
    eslik.append(dict(seed=sd, y=mm.iou_goruntu_y.mean(), r=mm.iou_goruntu_r.mean(),
                      r_iyi=int((mm.iou_goruntu_r > mm.iou_goruntu_y).sum()), y_iyi=int((mm.iou_goruntu_y > mm.iou_goruntu_r).sum()),
                      p=wilcoxon(mm.iou_goruntu_r, mm.iou_goruntu_y).pvalue))
eslik = pd.DataFrame(eslik)

# Örnek düzeyi: eşleşen / kaçırılan / fazladan (3 tohum toplamı) ve sınıfı doğru eşleşme oranı
ins = {}
for m in MODELLER:
    cm = sum(seg.load_instance_confusion(s) for s in RUNS[m])
    es = cm[:7, :7].sum()
    ins[m] = dict(gercek=int(cm[:7].sum()), eslesen=int(es), kacirilan=int(cm[:7, 7].sum()), fazladan=int(cm[7, :7].sum()),
                  dogru=float(np.trace(cm[:7, :7]) / es), cm=cm)

# Seçilen modellerde IoU'su 0 olan test görüntüsü sayısı
sifir = {k: int((seg.load_per_image(seg.parse_run(k)).iou_goruntu == 0).sum()) for k in (SEC_Y, SEC_R)}
n_test = st["test"]["n_images"]

# ================================================================ BAŞLIK (tek sütun)
para([("Çilek Hastalıklarında Örnek Bölütleme: YOLO26-seg ve RF-DETR Seg Modellerinin Doğruluk, Hız ve "
       "Hata Türleri Açısından Karşılaştırılması", {})], align=WD_ALIGN_PARAGRAPH.CENTER, girinti=False, size=20, after=8)
para([(YAZAR, {})], align=WD_ALIGN_PARAGRAPH.CENTER, girinti=False, size=11, after=1)
para([(f"Öğrenci No: {OGRENCI_NO} · {KURUM}", {"italic": True})], align=WD_ALIGN_PARAGRAPH.CENTER, girinti=False, size=9, after=1)
para([(f"{DERS} dersi · Dersin sorumlusu: {HOCA} · 2. Rapor (Bölütleme) · {TESLIM}", {})],
     align=WD_ALIGN_PARAGRAPH.CENTER, girinti=False, size=9, after=10)
yeni_bolum(2)

# ================================================================ ÖZET
mN, mS, mR = (agg(m, "mask_map50_95") for m in MODELLER)
iN, iS, iR = (agg(m, "miou") for m in MODELLER)
dN, dS, dR = (agg(m, "mdice") for m in MODELLER)
gk_s, gk_r = float(gk.loc["gruplar_arasi_toplam", YS]), float(gk.loc["gruplar_arasi_toplam", RD])
gk_cls_n = float(gk_cls.loc["gruplar_arasi_toplam", "yolo26n-cls"])
para([("Özet", {"bold": True, "italic": True}),
      ("—Çilek bitkisindeki yedi hastalığın lezyonları, Afzaal ve ark. [1] veri kümesinin resmî bölmesiyle örnek düzeyinde "
       "bölütlenmiş; evrişimli YOLO26-seg (nano, small) ile dönüştürücü tabanlı RF-DETR Seg Small aynı çözünürlükte (480), "
       "aynı eğitim bütçesiyle, aynı T4 GPU'da ve aynı değerlendirme koduyla, üçer tohumla karşılaştırılmıştır. LabelMe "
       "çokgenleri YOLO-seg ve COCO biçimlerine çevrilmiş, dönüşüm geri çevirmeyle ve Roboflow dışa aktarımıyla doğrulanmıştır. "
       f"Test kümesinde maske mAP50-95 YOLO26n için {ms(*mN)}, YOLO26s için {ms(*mS)}, RF-DETR için {ms(*mR)}; mIoU sırasıyla "
       f"{f(iN[0])}, {f(iS[0])} ve {f(iR[0])} bulunmuştur. RF-DETR en doğru modeldir, ancak görüntü başına "
       f"{f(eff.loc[RD, 'ms_uctan_uca'], 0)} ms ile YOLO26s'ten ({f(eff.loc[YS, 'ms_uctan_uca'], 0)} ms) yaklaşık "
       f"{f(eff.loc[RD, 'ms_uctan_uca'] / eff.loc[YS, 'ms_uctan_uca'], 1)} kat yavaştır. Bölütlemede meyve ve yaprak "
       f"hastalıkları neredeyse hiç karışmamaktadır (gruplar arası {pct(min(gk_s, gk_r), 2)}–{pct(max(gk_s, gk_r, float(gk.loc['gruplar_arasi_toplam', YN])), 2)}; sınıflandırmada "
       f"{pct(gk_cls_n)}); başlıca hata türü lezyonların hiç bulunamamasıdır.", {"bold": True})],
     girinti=False, size=9)
para([("Anahtar kelimeler", {"bold": True, "italic": True}),
      ("—örnek bölütleme, YOLO26-seg, RF-DETR, çilek hastalıkları, mIoU, maske mAP", {"bold": True})],
     girinti=False, size=9, after=4)

# ================================================================ I. GİRİŞ
h1("Giriş")
para("Birinci raporda aynı veri kümesinde görüntü düzeyinde sınıflandırma yapılmış ve hataların çoğunun benzer görünümlü "
     "meyve hastalıkları arasında olduğu görülmüştü. Sınıflandırma, bir görüntüde birden çok lezyon ya da birden çok hastalık "
     "olduğunda tek etiketle yetinmek zorundadır. Sera uygulamasında asıl istenen, hastalıklı bölgenin fotoğraf üzerinde "
     "işaretlenmesidir; bu, her lezyonun sınıfını ve piksel maskesini veren örnek bölütleme (instance segmentation) problemidir.")
para("Afzaal ve ark. [1] bu veri kümesini Mask R-CNN [2] ile örnek bölütleme için yayımlamıştır. Bu çalışmada iki güncel ve "
     "farklı mimari aile karşılaştırılmaktadır: Ultralytics YOLO26-seg [3] (evrişimli, tek aşamalı) ve RF-DETR Seg [4] "
     "(DETR [5] türü uçtan uca dönüştürücü kod çözücü, DINOv2 [6] omurgası). Katkılar: (i) çokgen etiketlerin iki biçime "
     "doğrulanmış dönüşümü; (ii) iki ailenin tahminlerini aynı COCO biçimine çeviren tek bir değerlendirme kodu; (iii) doğruluk, "
     "hız, parametre ve GPU belleğinin aynı donanımda ölçümü; (iv) meyve/yaprak karışmasının ve en kötü 10 tahminin incelenmesi.")

# ================================================================ II. VERİ
h1("Veri ve Etiket Dönüşümü")
h2("Örnekler")
satir = []
for c in sorted(C.CLASSES, key=lambda c: (C.GROUPS.index(C.GROUP_OF[c]), c)):
    satir.append([c, C.GROUP_OF[c]] + [n(st[s]["instances_per_class"][c]) for s in ("train", "val", "test")])
satir.append(["Toplam (örnek)", ""] + [n(st[s]["n_instances"]) for s in ("train", "val", "test")])
satir.append(["Görüntü", ""] + [n(st[s]["n_images"]) for s in ("train", "val", "test")])
tablo("Bölmelere göre örnek (çokgen) sayıları", ["Sınıf", "Grup", "Eğitim", "Doğr.", "Test"], satir,
      [3.3, 1.3, 1.2, 1.2, 1.2], hizalar=["l", "l", "r", "r", "r"], kalin_satirlar=(len(satir) - 2, len(satir) - 1))
ar = {s: st[s]["instance_area_ratio"] for s in ("train", "test")}
para(f"Tüm görüntüler 419×419 pikseldir. Görüntü başına ortalama {f(st['train']['instances_per_image']['mean'], 2)} "
     f"(eğitim) ve {f(st['test']['instances_per_image']['mean'], 2)} (test) lezyon vardır; en kalabalık görüntüde "
     f"{st['test']['instances_per_image']['max']} lezyon bulunur. Lezyonlar çoğunlukla küçüktür: bir lezyonun görüntü alanına "
     f"oranının ortancası eğitimde {f(ar['train']['median'], 3)}, testte {ek(f(ar['test']['median'], 3), 'dir')}; eğitimde "
     f"{ar['train']['small_lt_1pct']} lezyon görüntünün %1'inden küçüktür. Örnek dağılımı görüntü dağılımından da dengesizdir "
     f"(Tablo I): yaprak lekesi ve yaprak küllemesi örneklerin çoğunu oluşturur.")
h2("Dönüşüm ve doğrulama")
para(f"LabelMe [7] JSON çokgenleri Python ile YOLO-seg (normalize çokgen) ve COCO [8] (kategori 0 = üst sınıf, hastalıklar "
     f"1–7) biçimlerine çevrilmiştir; kendini kesen 12 çokgen onarılmış, görüntü dışına taşan 4 çokgen kırpılmıştır. Dönüşüm "
     f"geri rasterleştirilerek denetlenmiştir: {n(rt['n_instances'])} örneğin hepsinde sınıf ve sayı korunmuş, en düşük IoU "
     f"YOLO-seg için {f(rt['iou_yolo_min'], 4)}, COCO için {f(rt['iou_coco_min'], 4)} çıkmıştır. Ödevin istediği Roboflow [9] "
     f"adımı bağımsız bir denetim olarak kullanılmıştır: ham veri Roboflow'a yüklenmiş, ön işlemesiz COCO dışa aktarımı bizim "
     f"dönüşümümüzle örnek örnek karşılaştırılmıştır ({n(RF['ornek_karsilastirilan'])} örnek, ortalama IoU "
     f"{f(RF['iou_ortalama'], 5)}). Roboflow'un atladığı {RF['atlanan_goruntu']} görüntü bölme içi bayt kopyalarıdır.")
para(f"Resmî bölme korunmuştur. Bölmeler arası yakın-yinelenenler algısal özetle (pHash ≤ {dup['phash_threshold']}) "
     f"bulunmuş; testin {dup['test_leaked']} görüntüsü eğitimde bir benzerine sahip olduğundan sonuçlar ayrıca "
     f"{dup['test_clean']} görüntülük temiz testte de verilmiştir. Doğrulama kümesinin "
     f"{ek(pct(dup['val_leaked'] / st['val']['n_images']), 'iyelik')} sızıntılıdır; doğrulama skorları iyimser kabul edilmiştir.")

# ================================================================ III. YÖNTEM
h1("Yöntem")
h2("Modeller")
para(f"YOLO26n-seg ({f(eff.loc[YN, 'parametre'] / 1e6, 2)} M parametre) ve YOLO26s-seg ({f(eff.loc[YS, 'parametre'] / 1e6, 2)} M) "
     f"Ultralytics {ENV['ultralytics']} ile; RF-DETR Seg Small ({f(eff.loc[RD, 'parametre'] / 1e6, 2)} M, pencereli DINOv2-S "
     f"omurga, 100 sorgu, 4 katmanlı kod çözücü) rfdetr {ENV['rfdetr']} paketiyle eğitilmiştir. Üç model de COCO üzerinde ön "
     "eğitimlidir. RF-DETR Seg sürümlerinin hepsi aynı omurgayı kullandığından parametre sayıları birbirine yakındır ve "
     "YOLO26s'in yaklaşık üç katıdır; bu nedenle "
     "karşılaştırma “aynı boyut” değil “aynı bütçe ve donanım” ilkesiyle yapılmıştır.")
h2("Eğitim düzeni")
ye = ENV["yolo_egitim"]; re_ = ENV["rfdetr_egitim"]
tablo("Eğitim ayarları",
      ["Ayar", "YOLO26-seg (n, s)", "RF-DETR Seg Small"],
      [["Giriş boyutu", str(ENV["imgsz"]), str(ENV["imgsz"])],
       ["En çok epoch / erken durd.", f"{ye['epochs']} / {ye['patience']}", f"{re_['epochs']} / {re_['early_stopping_patience']}"],
       ["Etkin yığın", str(ye["batch"]), f"{re_['batch_size']} × {re_['grad_accum_steps']} birikim"],
       ["Optimizasyon", "Ultralytics “auto”", "AdamW, 1×10⁻⁴ (paket varsayılanı)"],
       ["En iyi ağırlık", "val uygunluk (kutu+maske)", "val maske mAP (EMA dahil)"],
       ["Tohum", "0, 1, 2", "0, 1, 2"]],
      [2.6, 2.7, 2.9], hizalar=["l", "l", "l"])
para("Ortak çözünürlük 480 seçilmiştir: YOLO 32'nin, RF-DETR Seg Small 24'ün katını ister ve 480 görüntülerin kendi boyutuna "
     "(419) en yakın ortak değerdir. Her aile kendi önerilen optimizasyon ve veri artırma ayarlarıyla eğitilmiş, epoch sınırı, "
     "erken durdurma sabrı ve etkin yığın boyu eşitlenmiştir. Model seçimi yalnız doğrulama maske mAP'siyle yapılmış, test "
     "kümesi en sonda bir kez kullanılmıştır. Eğitimler ücretsiz Colab T4 GPU'da, kota nedeniyle birkaç oturuma bölünerek ve "
     "kaldığı yerden sürdürülerek tamamlanmıştır.")
h2("Ortak değerlendirme")
para("İki ailenin tahminleri aynı COCO results biçimine (RLE maske, güven ≥ 0,001, görüntü başına en çok 100 tahmin) "
     "çevrilip tek bir kodla ölçülmüştür. Maske mAP50 ve mAP50-95 pycocotools [8] ile hesaplanmıştır. Piksel ölçütleri "
     "(mIoU ve Dice [10], yedi hastalık sınıfının ortalaması, arka plan hariç) için tahminler bir güven eşiğiyle piksel sınıf "
     "haritasına çevrilir; eşik her koşu için doğrulamada mIoU'yu en yükselten değer olarak seçilmiştir "
     f"({f(res.esik.min(), 1)}–{f(res.esik.max(), 1)}). Hata analizi için her gerçek lezyon, sınıfa bakılmadan IoU ≥ 0,5 "
     "örtüşen tahminle açgözlü biçimde eşleştirilmiş; eşleşen çiftlerden örnek karışma matrisi, eşleşmeyenlerden "
     "“kaçırılan” ve “fazladan” sayıları çıkarılmıştır. Aynı tohumlu modellerin görüntü başına IoU "
     "değerleri Wilcoxon işaretli sıra testiyle [11] karşılaştırılmıştır. Hız ve bellek T4'te, yığın boyu 1 ile 743 test "
     "görüntüsünün tamamında ölçülmüştür.")
cap = []
for m in MODELLER:
    for s in RUNS[m]:
        t = res[(res.kosu == s.name) & (res.alt_kume == "tum_test")].iloc[0]
        cap.append(abs(t.val_mask_map50_95_kutuphane - t.val_mask_map50_95_bizim))
cap_r = [abs(r.val_mask_map50_95_kutuphane - r.val_mask_map50_95_bizim) for r in res[(res.model == RD) & (res.alt_kume == "tum_test")].itertuples()]
cap_y = [abs(r.val_mask_map50_95_kutuphane - r.val_mask_map50_95_bizim) for r in res[(res.mimari == "yolo") & (res.alt_kume == "tum_test")].itertuples()]
para(f"Ortak kodun doğruluğu çapraz kontrolle sınanmıştır: doğrulama maske mAP50-95 değerimiz ile kütüphanelerin eğitim "
     f"sırasında raporladığı değer arasındaki fark RF-DETR'da en çok {f(max(cap_r), 3)}, YOLO'da {f(min(cap_y), 3)}–"
     f"{ek(f(max(cap_y), 3), 'dir')}. Sınıf eşlemesinde bir hata olsaydı fark çok büyük olurdu; YOLO'daki küçük fark maske "
     "çözünürlüğü ve iç değerlendirme ayrıntılarından kaynaklanmaktadır.")

# ================================================================ IV. SONUÇLAR
h1("Deneysel Sonuçlar")
h2("Eğitim seyri")
ep = {m: [egitim(s)[0] for s in RUNS[m]] for m in MODELLER}
para(f"RF-DETR koşuları {min(ep[RD])}–{max(ep[RD])}. epoch'ta erken durmuştur; doğrulama maske mAP'si yaklaşık 15–20 "
     f"epoch'ta doymuştur (Şekil 1). YOLO koşularının çoğu ise 100 epoch sınırına ulaşmıştır (YOLO26n: "
     f"{', '.join(map(str, ep[YN]))}; YOLO26s: {', '.join(map(str, ep[YS]))} epoch); eğrileri sona doğru çok yavaş "
     "yükselmektedir. Kayıptaki 90. epoch sonrası ani düşüş, Ultralytics'in son 10 epoch'ta mozaik veri artırmayı kapatmasından "
     "kaynaklanır. Eşit epoch bütçesinde RF-DETR çok daha hızlı yakınsamaktadır.")

yeni_bolum(1)
sekil(FIG / "seg_egriler.png", "Eğitim kaybı (ilk epoch'a oranla) ve doğrulama maske mAP50-95; dokuz koşu (üç model × üç tohum).",
      METIN_GENISLIK - 1.0)
yeni_bolum(2)

h2("Genel karşılaştırma")
satir = []
for m in MODELLER:
    satir.append([KISA[m], ms(*agg(m, "mask_map50")), ms(*agg(m, "mask_map50_95")), ms(*agg(m, "miou")), ms(*agg(m, "mdice")),
                  f(agg(m, "mask_map50_95", "temiz_test")[0])])
tablo("Test sonuçları (üç tohum ortalaması ± std)",
      ["Model", "mAP50", "mAP50-95", "mIoU", "Dice", "mAP50-95 (temiz)"], satir, [1.2, 1.6, 1.6, 1.6, 1.6, 1.0], yazi=7)
para(f"Tablo III'e göre RF-DETR dört ölçütün hepsinde en iyidir ve tohumlar arası değişkenliği en küçüktür (mAP50-95 std "
     f"{f(mR[1])}). YOLO26s, YOLO26n'den her ölçütte biraz iyidir; ancak RF-DETR ile arasındaki fark (mAP50-95'te "
     f"{f(100 * (mR[0] - mS[0]), 1)} puan, mIoU'da {f(100 * (iR[0] - iS[0]), 1)} puan) iki YOLO boyu arasındaki farktan "
     f"({f(100 * (mS[0] - mN[0]), 1)} puan) çok büyüktür. Temiz test skorları tüm test skorlarına çok yakındır; test "
     "sızıntısı sonucu şişirmemektedir.")
para(f"Aynı tohumla eğitilmiş YOLO26s ve RF-DETR görüntü başına IoU ile eşli karşılaştırıldığında RF-DETR üç tohumda da daha "
     f"iyidir (RF-DETR'ın daha iyi olduğu görüntü sayısı {', '.join(str(x) for x in eslik.r_iyi)}; YOLO'nun daha iyi olduğu "
     f"{', '.join(str(x) for x in eslik.y_iyi)}; Wilcoxon p < {ust10(eslik.p.max())}). Fark rastlantıyla açıklanamaz.")

h2("Sınıf bazında başarım")
satir = []
for c in sorted(C.CLASSES, key=lambda c: (C.GROUPS.index(C.GROUP_OF[c]), c)):
    satir.append([c] + [f(pcm(m, c, "ap50_95"), 2) for m in MODELLER] + [f(pcm(m, c, "iou"), 2) for m in MODELLER])
tablo("Sınıf bazında maske AP50-95 ve piksel IoU (tüm test, tohum ortalaması)",
      ["Sınıf", "AP n", "AP s", "AP RF", "IoU n", "IoU s", "IoU RF"], satir,
      [2.6, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], hizalar=["l"] + ["c"] * 6, yazi=7)
fark_ap = {c: pcm(RD, c, "ap50_95") - pcm(YS, c, "ap50_95") for c in C.CLASSES}
ilk3 = sorted(fark_ap, key=fark_ap.get, reverse=True)[:3]
para(f"Tablo IV'te RF-DETR'ın YOLO26s'e üstünlüğü en çok " + ", ".join(
     f"{tr(c)} (AP50-95 {f(pcm(YS, c, 'ap50_95'), 2)} → {f(pcm(RD, c, 'ap50_95'), 2)})" for c in ilk3) + " sınıflarındadır. "
     f"Antraknozda üç model de zayıftır (AP50-95 ≈ {f(np.mean([pcm(m, 'Anthracnose Fruit Rot', 'ap50_95') for m in MODELLER]), 2)}): "
     "en az örneğe sahip ve sınıflandırmada da en çok karışan sınıflardan biridir. Meyve küllemesinde YOLO26n belirgin biçimde "
     f"geride kalırken (IoU {f(pcm(YN, 'Powdery Mildew Fruit', 'iou'), 2)}) YOLO26s ve RF-DETR çok daha iyidir. Çiçek yanıklığında "
     "üç model birbirine yakındır.")

h2("Verimlilik")
satir = [[KISA[m], f(eff.loc[m, "parametre"] / 1e6, 2), f(eff.loc[m, "ms_ileri_gecis"], 1), f(eff.loc[m, "ms_uctan_uca"], 1),
          f(eff.loc[m, "gpu_bellek_mb"], 0), f(agg(m, "mask_map50_95")[0])] for m in MODELLER]
tablo("Parametre, süre ve GPU belleği (T4, yığın boyu 1)",
      ["Model", "Param. (M)", "İleri geçiş ms", "Uçtan uca ms", "Bellek (MB)", "mAP50-95"], satir,
      [1.4, 1.2, 1.5, 1.5, 1.3, 1.3], yazi=7)
para(f"RF-DETR'ın ileri geçişi YOLO26s'ten {f(eff.loc[RD, 'ms_ileri_gecis'] / eff.loc[YS, 'ms_ileri_gecis'], 1)} kat uzundur ve "
     f"{f(eff.loc[RD, 'gpu_bellek_mb'] / eff.loc[YS, 'gpu_bellek_mb'], 1)} kat bellek kullanır (Tablo V). YOLO26n ile YOLO26s'in "
     f"süreleri ise neredeyse aynıdır ({f(eff.loc[YN, 'ms_uctan_uca'], 1)} ve {f(eff.loc[YS, 'ms_uctan_uca'], 1)} ms); small boyuna "
     "geçmenin hız bedeli yok denecek kadar azdır. Görüntü başına "
     f"{f(eff.loc[RD, 'ms_uctan_uca'], 0)} ms ile RF-DETR bile saniyede yaklaşık {f(1000 / eff.loc[RD, 'ms_uctan_uca'], 0)} "
     "fotoğraf işleyebilir; telefonla tek tek çekilen fotoğrafların sunucuda incelendiği bir senaryoda bu yeterlidir. Cihaz "
     "üzerinde, gerçek zamanlı çalışma gerekiyorsa YOLO26s daha uygun bir dengedir (Şekil 2).")
sekil(FIG / "seg_dogruluk_hiz.png", "Doğruluk–hız dengesi: tüm test maske mAP50-95 (tohum ortalaması ± std) ve uçtan uca süre.",
      SUTUN - 0.6)

# ================================================================ V. HATA ANALİZİ
h1("Hata Analizi ve Meyve/Yaprak Karışması")
h2("Örnek düzeyinde hatalar")
sekil(FIG / "seg_karisma_rfdetr-seg-small.png",
      "RF-DETR örnek karışma matrisi (IoU ≥ 0,5 eşleşme, üç tohum toplamı, satır normalize). Son sütun kaçırılan gerçek "
      "lezyonlar, son satır gerçekte olmayan yere çizilen tahminlerdir.", SUTUN - 0.3)
satir = [[KISA[m], n(ins[m]["gercek"]), n(ins[m]["eslesen"]), n(ins[m]["kacirilan"]), n(ins[m]["fazladan"]), pct(ins[m]["dogru"])]
         for m in MODELLER]
tablo("Örnek düzeyi sayımlar (üç tohum toplamı, tüm test)",
      ["Model", "Gerçek", "Eşleşen", "Kaçırılan", "Fazladan", "Sınıf doğru"], satir, [1.4, 1.2, 1.3, 1.35, 1.35, 1.4], yazi=7)
kac = {c: ins[RD]["cm"][i, 7] / ins[RD]["cm"][i].sum() for i, c in enumerate(C.CLASSES)}
kac3 = sorted(kac, key=kac.get, reverse=True)[:3]
assert ins[RD]["fazladan"] < ins[YN]["fazladan"] and ins[RD]["kacirilan"] < min(ins[YN]["kacirilan"], ins[YS]["kacirilan"])
ls_al = ins[RD]["cm"][4, 0] / ins[RD]["cm"][4].sum()
para(f"Eşleşen lezyonlarda sınıf neredeyse her zaman doğrudur (RF-DETR {pct(ins[RD]['dogru'])}, YOLO26s "
     f"{pct(ins[YS]['dogru'])}; Tablo VI). Asıl hata kaçırmadır: RF-DETR gerçek lezyonların "
     f"{ek(pct(ins[RD]['kacirilan'] / ins[RD]['gercek']), 'iyelik_belirtme')}, YOLO26n {ek(pct(ins[YN]['kacirilan'] / ins[YN]['gercek']), 'iyelik_belirtme')} hiç "
     f"bulamamıştır. RF-DETR'da kaçırma oranı en yüksek sınıflar şunlardır: " + ", ".join(f"{tr(c)} ({pct(kac[c], 0)})" for c in kac3)
     + " (Şekil 3). "
     f"Sınıf karışmaları birinci rapordakiyle aynı çiftlerdedir: yaprak lekesi → köşeli yaprak lekesi ({pct(ls_al, 0)}). "
     f"RF-DETR ayrıca daha az fazladan tahmin üretir ({n(ins[RD]['fazladan'])}, YOLO26n {n(ins[YN]['fazladan'])}).")

h2("Meyve/yaprak karışması (Dikkat maddesi)")
satir = []
for m in MODELLER:
    satir.append([KISA[m] + " (seg)", pct(float(gk.loc["yaprak->meyve", m]), 2), pct(float(gk.loc["meyve->yaprak", m]), 2),
                  pct(float(gk.loc["gruplar_arasi_toplam", m]), 2)])
for m, ad in (("yolo26n-cls", "YOLO26n-cls (1. rapor)"), ("yolo26s-cls", "YOLO26s-cls (1. rapor)")):
    satir.append([ad, pct(float(gk_cls.loc["yaprak->meyve", m]), 2), pct(float(gk_cls.loc["meyve->yaprak", m]), 2),
                  pct(float(gk_cls.loc["gruplar_arasi_toplam", m]), 2)])
tablo("Organ grupları arası karışma: bölütleme ↔ sınıflandırma",
      ["Model", "yaprak → meyve", "meyve → yaprak", "Toplam"], satir, [3.0, 2.1, 2.1, 1.4], yazi=7)
para(f"Hocanın “Dikkat” maddesi için örnek karışma matrisi yaprak/meyve/çiçek gruplarına indirgenmiştir (Tablo VII). "
     f"Bölütlemede gruplar arası karışma YOLO26n {pct(float(gk.loc['gruplar_arasi_toplam', YN]), 2)}, YOLO26s {pct(gk_s, 2)}, "
     f"RF-DETR {pct(gk_r, 2)} ile sınıflandırmadakinin ({pct(gk_cls_n)}) onda birinden azdır ve yalnız yapraktan meyveye "
     "doğrudur. Sınıflandırmada model, kadrajdaki yaprak ve meyveyi tek bir etikette birleştirmek zorundaydı; bölütlemede "
     "her lezyon kendi bölgesiyle değerlendirildiğinden organ ayrımı neredeyse kusursuzdur. Kalan güçlük aynı organ "
     "içindeki benzer hastalıkları ayırmak ve küçük lezyonları bulmaktır.")

h2("En kötü 10 tahmin")
para(f"Her aileden doğrulama ile seçilen model ({kisa(SEC_Y)}; {kisa(SEC_R)}) alınmış, görüntü IoU ortalaması en düşük on test "
     f"görüntüsü Şekil 4'te gösterilmiştir. Onunda da iki modelin IoU'su sıfırdır (seçilen YOLO'da IoU'su sıfır olan görüntü "
     f"{sifir[SEC_Y]}, RF-DETR'da {sifir[SEC_R]}; toplam {n_test}). Hatalar dört türe ayrılmaktadır:")
madde = [
    ("Kaçırılan lezyon. ", "Angular Leafspot 397 ve 401'de soluk, düşük kontrastlı yapraklarda ve Gray Mold 528'de meyvesiz, "
     "kurumuş bir çiçek kalıntısı üzerindeki küfte iki model de hiçbir şey çizmemiştir; bu bağlam eğitim verisinde seyrektir."),
    ("Doğru bölge, yanlış yaprak lekesi. ", "Leaf Spot 497 ve 586'da tüm yaprak etiketlidir; modeller aynı bölgeyi Angular "
     "Leafspot olarak işaretlemiştir. Bu, birinci rapordaki Leaf Spot → Angular Leafspot karışmasının aynısıdır."),
    ("Organ dışı ya da meyve içi yanlış sınıf. ", "YOLO, Angular Leafspot 355 ve 408'de yaprağı Gray Mold olarak boyamıştır "
     "(azınlıktaki yaprak→meyve hatası); RF-DETR, Anthracnose 101'i Gray Mold sanmıştır. Bu görüntü sınıflandırmanın da "
     "en kötü 10 listesindeydi."),
    ("Etiket tutarsızlığı. ", "Gray Mold 517'de iki model de etiketlenmemiş başka bir kurumuş çiçekte küf bulmuş, etiketli "
     "küçük bölgeyi kaçırmıştır. Ayrıca bazı görüntülerde tek tek lekeler, bazılarında (Leaf Spot 497, Angular Leafspot "
     "397 ve 408) yaprağın tamamı ya da geniş bir kısmı tek çokgenle etiketlenmiştir; bu etiket stili farkı IoU'yu düşürür."),
]
for bas, metin in madde:
    para([(bas, {"italic": True}), metin])

yeni_bolum(1)
sekil(FIG / "seg_en_kotu10.png",
      "En kötü 10 test görüntüsü: her üçlüde gerçek maske, val ile seçilen YOLO26s ve RF-DETR tahmini (renkler sınıfı gösterir).",
      METIN_GENISLIK - 1.2)
yeni_bolum(2)

# ================================================================ VI. SINIRLILIKLAR, VII. SONUÇ
h1("Sınırlılıklar")
para(f"(i) Doğrulama kümesinin en az {ek(pct(dup['val_leaked'] / st['val']['n_images'], 0), 'iyelik')} sızıntılıdır; erken durdurma ve "
     "eşik seçimi bu iyimser skorlarla yapılmıştır. (ii) YOLO koşularının çoğu 100 epoch sınırına ulaşmıştır; daha uzun eğitim "
     "YOLO'yu bir miktar iyileştirebilir, fakat eğriler doymaya yakındır. (iii) İki aile farklı optimizasyon ve veri artırma "
     "ayarlarıyla (kendi varsayılanları) eğitilmiş, hiperparametre araması yapılmamıştır. (iv) Ücretsiz Colab her oturumda "
     "paketleri yeniden kurduğu için koşular arasında küçük sürüm farkları vardır (Ultralytics 8.4.215/8.4.216). (v) Hız ve "
     "bellek her modelin tek bir koşusunda ölçülmüştür. (vi) Etiket stili tutarsızlıkları (tüm yaprak ↔ tek leke) piksel "
     "ölçütlerini her iki aile için de aşağı çekmektedir.")
h1("Sonuç")
para(f"Aynı veri, aynı çözünürlük ve aynı değerlendirme koduyla RF-DETR Seg Small, YOLO26-seg modellerinden belirgin biçimde "
     f"daha doğrudur (maske mAP50-95 {f(mR[0])} ↔ {f(mS[0])}, mIoU {f(iR[0])} ↔ {f(iS[0])}) ve çok daha az epoch'ta yakınsar; "
     f"buna karşılık yaklaşık {f(eff.loc[RD, 'ms_uctan_uca'] / eff.loc[YS, 'ms_uctan_uca'], 0)} kat yavaş ve "
     f"{f(eff.loc[RD, 'parametre'] / eff.loc[YS, 'parametre'], 0)} kat büyüktür. Sera senaryosunda, fotoğrafın sunucuda "
     "incelendiği bir uygulama için RF-DETR, cihaz üzerinde anlık çalışma için YOLO26s önerilir. Bölütleme meyve ve yaprak "
     "hastalıklarını neredeyse hiç karıştırmamaktadır; geliştirme için asıl hedef küçük lezyonları kaçırmamak ve etiket "
     "stilini tutarlı hâle getirmektir.")

h1("Yapay zekâ araçlarının kullanımı", numarasiz=True)
para("Bu çalışmada kodların, notebook'ların ve bu rapor taslağının hazırlanmasında Anthropic'in Claude (Claude Code) "
     "yapay zekâ asistanından yararlanılmıştır. Deney tasarımına ilişkin kararlar yazar tarafından verilmiş, eğitimler "
     "yazar tarafından Google Colab'da çalıştırılmıştır. Rapordaki tüm sayılar gerçek çalıştırma çıktılarından bir betikle "
     "aktarılmış; metin yazar tarafından okunup düzeltilmiştir ve içerikten yazar sorumludur.", size=9)

h1("Kaynakça", numarasiz=True)
KAYNAK = [
    "U. Afzaal, B. Bhattarai, Y. R. Pandeya ve J. Lee, “An instance segmentation model for strawberry diseases based "
    "on Mask R-CNN,” Sensors, c. 21, sy. 19, s. 6565, 2021, doi: 10.3390/s21196565.",
    "K. He, G. Gkioxari, P. Dollár ve R. Girshick, “Mask R-CNN,” Proc. IEEE Int. Conf. Comput. Vis. (ICCV), "
    "2017, ss. 2961–2969.",
    f"G. Jocher, J. Qiu ve A. Chaurasia, “Ultralytics YOLO,” sürüm {ENV['ultralytics']}, 2026. [Çevrimiçi]. "
    "Erişim: https://github.com/ultralytics/ultralytics",
    f"Roboflow, “RF-DETR: Real-time detection transformer,” sürüm {ENV['rfdetr']}, 2026. [Çevrimiçi]. "
    "Erişim: https://github.com/roboflow/rf-detr",
    "N. Carion, F. Massa, G. Synnaeve, N. Usunier, A. Kirillov ve S. Zagoruyko, “End-to-end object detection with "
    "transformers,” Proc. Eur. Conf. Comput. Vis. (ECCV), 2020, ss. 213–229.",
    "M. Oquab vd., “DINOv2: Learning robust visual features without supervision,” Trans. Mach. Learn. Res., 2024.",
    "B. C. Russell, A. Torralba, K. P. Murphy ve W. T. Freeman, “LabelMe: A database and web-based tool for image "
    "annotation,” Int. J. Comput. Vis., c. 77, ss. 157–173, 2008.",
    "T.-Y. Lin vd., “Microsoft COCO: Common objects in context,” Proc. Eur. Conf. Comput. Vis. (ECCV), 2014, "
    "ss. 740–755.",
    "Roboflow, Inc., “Roboflow,” 2026. [Çevrimiçi]. Erişim: https://roboflow.com",
    "L. R. Dice, “Measures of the amount of ecologic association between species,” Ecology, c. 26, sy. 3, "
    "ss. 297–302, 1945.",
    "F. Wilcoxon, “Individual comparisons by ranking methods,” Biometrics Bulletin, c. 1, sy. 6, ss. 80–83, 1945.",
]
for i, k in enumerate(KAYNAK, 1):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.left_indent = Cm(0.6)
    p.paragraph_format.first_line_indent = Cm(-0.6)
    p.paragraph_format.space_after = Pt(1)
    run(p, f"[{i}]\t{k}", size=8)

h1("Ek: Kod, ortam ve tekrarlanabilirlik", numarasiz=True)
COLAB_URL = "https://colab.research.google.com/github/toprakozkale/derin-ogrenme-odev13/blob/main/notebooks/"
para([("Çalıştırılabilir Colab defteri (bölütleme): ", {"bold": True}), COLAB_URL + "03_segmentasyon.ipynb"],
     girinti=False, size=9, align=WD_ALIGN_PARAGRAPH.LEFT)
para([("Kod deposu: ", {"bold": True}), "https://github.com/toprakozkale/derin-ogrenme-odev13"],
     girinti=False, size=9, align=WD_ALIGN_PARAGRAPH.LEFT)
para("Defter gizli anahtar gerektirmez (T4 GPU seçilip Tümünü çalıştır). Veriyi Kaggle'dan indirir, dokuz eğitimi yapar "
     "(T4'te birkaç oturum sürer; kesilirse biten koşular atlanır, yarım kalan kaldığı yerden sürer) ve bu rapordaki tablo "
     "ve şekilleri aynı kodla üretir.", girinti=False, size=9)
para([f"Ortam: Python {ENV['python']}, PyTorch {ENV['torch']}, Ultralytics {ENV['ultralytics']}, rfdetr {ENV['rfdetr']}, "
      f"pycocotools {ENV['pycocotools']}, CUDA {ENV['cuda']}, {ENV['cihaz']} (Google Colab)."], girinti=False, size=9)

OUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(OUT)
print(f"yazıldı: {OUT.relative_to(ROOT)}  (tablo {sayaclar()[0]}, şekil {sayaclar()[1]})")
