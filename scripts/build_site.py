"""Panelin salt-okunur statik kopyasını üretir: site/ (Vercel'e yüklenir).

Streamlit paneli sürekli çalışan bir sunucu ister; Vercel ise statik dosya dağıtır. Bu betik panelin kullandığı aynı
kaynaklardan (docs/durum.yaml, docs/gunluk.md, runs/cls, runs/seg, data/derived/summary.json, rapor/) düz HTML sayfaları
üretir. Etkileşimli veri gezgini sayfaları (2.500 fotoğrafın kendisine ihtiyaç duyar) sitede yoktur; yerel panelde kalır.

    .venv/bin/python scripts/build_site.py          # → site/
    cd site && vercel deploy                        # yükleme (önce `vercel login`)
"""
from __future__ import annotations

import html
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import markdown
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config as C  # noqa: E402

SITE = ROOT / "site"
FIG = ROOT / "rapor" / "figures"
REPO = "https://github.com/toprakozkale/derin-ogrenme-odev13"
DURUM = yaml.safe_load((ROOT / "docs" / "durum.yaml").read_text())
SUMMARY = json.loads((ROOT / "data" / "derived" / "summary.json").read_text())
ICON = {"tamam": "✅", "devam": "⏳", "bekliyor": "⬜"}
DURUM_AD = {"tamam": "tamam", "devam": "sürüyor", "bekliyor": "bekliyor"}


def f(x: float, d: int = 3) -> str:
    return f"{x:.{d}f}".replace(".", ",")


def n(x) -> str:
    return f"{int(x):,}".replace(",", ".")


def ms(v: pd.Series, d: int = 3) -> str:
    return f"{f(v.mean(), d)} ± {f(v.std(ddof=1), d)}"


def esc(s) -> str:
    return html.escape(str(s))


def tablo(basliklar, satirlar, sayisal_baslangic=1, vurgu=None, sayisal_bitis=None) -> str:
    bit = sayisal_bitis if sayisal_bitis is not None else len(basliklar)
    sayi = lambda i: sayisal_baslangic <= i < bit  # noqa: E731
    th = "".join(f"<th{' class=num' if sayi(i) else ''}>{esc(b)}</th>" for i, b in enumerate(basliklar))
    trs = []
    for k, r in enumerate(satirlar):
        cls = ' class="vurgu"' if vurgu is not None and k == vurgu else ""
        tds = "".join(f"<td{' class=num' if sayi(i) else ''}>{c}</td>" for i, c in enumerate(r))
        trs.append(f"<tr{cls}>{tds}</tr>")
    return f'<div class="tablo"><table><thead><tr>{th}</tr></thead><tbody>{"".join(trs)}</tbody></table></div>'


def sekil(ad: str, aciklama: str) -> str:
    p = FIG / f"{ad}.png"
    if not p.exists():
        return ""
    return (f'<figure><a href="fig/{ad}.png"><img src="fig/{ad}.png" alt="{esc(aciklama)}" loading="lazy"></a>'
            f"<figcaption>{esc(aciklama)}</figcaption></figure>")


SAYFALAR = [("index.html", "Genel Bakış"), ("raporlar.html", "Raporlar"), ("veri.html", "Veri"),
            ("siniflandirma.html", "Sınıflandırma"), ("segmentasyon.html", "Segmentasyon"), ("gunluk.html", "Günlük")]

CSS = """
:root{--bg:#f7f6f2;--yuzey:#ffffff;--metin:#1d1d1b;--soluk:#5f5e5a;--cizgi:#e3e1da;--vurgu:#c2410c;--vurgu-z:#fff1e8;
--ok:#15803d;--devam:#b45309;--bekle:#6b7280;--link:#1d4ed8;color-scheme:light}
@media (prefers-color-scheme:dark){:root{--bg:#161614;--yuzey:#1f1f1c;--metin:#ecebe6;--soluk:#a3a29c;--cizgi:#34332f;
--vurgu:#fb923c;--vurgu-z:#3a2517;--ok:#4ade80;--devam:#fbbf24;--bekle:#9ca3af;--link:#93c5fd;color-scheme:dark}
figure img{background:#fff}}
*{box-sizing:border-box}html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--metin);font:16px/1.6 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
a{color:var(--link)}header{position:sticky;top:0;z-index:5;background:var(--yuzey);border-bottom:1px solid var(--cizgi)}
.ust{max-width:1100px;margin:0 auto;padding:10px 16px;display:flex;flex-wrap:wrap;gap:6px 18px;align-items:center}
.logo{font-weight:700;text-decoration:none;color:var(--metin);margin-right:8px}
nav{display:flex;flex-wrap:wrap;gap:4px}nav a{text-decoration:none;color:var(--soluk);padding:4px 10px;border-radius:999px;font-size:.92rem}
nav a.aktif{background:var(--vurgu-z);color:var(--vurgu);font-weight:600}
main{max-width:1100px;margin:0 auto;padding:24px 16px 64px}h1{font-size:1.9rem;line-height:1.25;margin:.2em 0 .3em}
h2{font-size:1.35rem;margin:2em 0 .6em;padding-top:.4em;border-top:1px solid var(--cizgi)}h3{font-size:1.1rem;margin:1.4em 0 .4em}
.alt{color:var(--soluk);margin-top:0}.kutu{background:var(--yuzey);border:1px solid var(--cizgi);border-radius:12px;padding:14px 16px}
.izgara{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(210px,1fr))}
.kpi .deger{font-size:1.7rem;font-weight:700;font-variant-numeric:tabular-nums}.kpi .etiket{color:var(--soluk);font-size:.9rem}
.basitce{background:var(--vurgu-z);border-left:4px solid var(--vurgu);padding:10px 14px;border-radius:8px;margin:12px 0}
.tablo{overflow-x:auto;margin:10px 0}table{border-collapse:collapse;width:100%;font-size:.93rem;background:var(--yuzey)}
th,td{padding:7px 10px;border-bottom:1px solid var(--cizgi);text-align:left;vertical-align:top}th{font-weight:600;white-space:nowrap}
.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}tr.vurgu td{background:var(--vurgu-z);font-weight:600}
.cubuk{height:8px;background:var(--cizgi);border-radius:99px;overflow:hidden;margin:6px 0}.cubuk>i{display:block;height:100%;background:var(--ok)}
ul.liste{list-style:none;padding:0;margin:6px 0}ul.liste li{padding:3px 0}
figure{margin:14px 0}figure img{max-width:100%;height:auto;border:1px solid var(--cizgi);border-radius:8px}
figcaption{color:var(--soluk);font-size:.88rem}.iki{display:grid;gap:16px;grid-template-columns:repeat(auto-fit,minmax(320px,1fr))}
.pdf{width:100%;height:80vh;border:1px solid var(--cizgi);border-radius:8px;background:#fff}
.dugme{display:inline-block;padding:8px 14px;border-radius:8px;background:var(--vurgu);color:#fff;text-decoration:none;font-weight:600;margin:4px 6px 4px 0}
.gunluk table{font-size:.88rem}.gunluk h2{font-size:1.2rem}footer{color:var(--soluk);font-size:.85rem;max-width:1100px;margin:0 auto;padding:0 16px 32px}
@media (max-width:560px){h1{font-size:1.5rem}.pdf{height:60vh}}
"""


def sayfa(dosya: str, baslik: str, govde: str) -> None:
    nav = "".join(f'<a href="{d}"{" class=aktif" if d == dosya else ""}>{esc(a)}</a>' for d, a in SAYFALAR)
    uretim = datetime.now().strftime("%d.%m.%Y %H:%M")
    (SITE / dosya).write_text(f"""<!doctype html><html lang="tr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><meta name="robots" content="noindex">
<title>{esc(baslik)} · Ödev 13</title><link rel="stylesheet" href="stil.css">
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🍓</text></svg>">
</head><body><header><div class="ust"><a class="logo" href="index.html">🍓 Ödev 13 · Çilek Hastalığı</a><nav>{nav}</nav></div></header>
<main>{govde}</main><footer>Kaynak dosyalardan otomatik üretildi ({uretim}) · <code>scripts/build_site.py</code> ·
<a href="{REPO}">kod deposu</a> (teslim gününe kadar gizli)</footer></body></html>""")


# ------------------------------------------------------------------ Genel Bakış
def genel() -> str:
    simdi = datetime.now()
    kart = []
    for t in DURUM["teslimler"]:
        due = datetime.strptime(t["tarih"], "%Y-%m-%d %H:%M")
        kalan = (due - simdi).days
        kart.append(f'<div class="kutu kpi"><div class="etiket">{esc(t["ad"])}</div><div class="deger">'
                    f'{f"{kalan} gün" if due > simdi else "süre doldu"}</div><div class="etiket">son teslim {due:%d.%m.%Y %H:%M}</div></div>')
    res_c = pd.read_csv(ROOT / "runs/cls/results.csv")
    res_s = pd.read_csv(ROOT / "runs/seg/results.csv")
    acc_n = res_c[(res_c.deney == "ana") & (res_c.boyut == "n") & (res_c.alt_kume == "tum_test")].accuracy
    map_r = res_s[(res_s.model == "rfdetr-seg-small") & (res_s.alt_kume == "tum_test")].mask_map50_95
    map_y = res_s[(res_s.model == "yolo26s-seg") & (res_s.alt_kume == "tum_test")].mask_map50_95
    kart.append(f'<div class="kutu kpi"><div class="etiket">Sınıflandırma doğruluğu (YOLO26n-cls)</div>'
                f'<div class="deger">{f(acc_n.mean())}</div><div class="etiket">test, 3 tohum ortalaması</div></div>')
    kart.append(f'<div class="kutu kpi"><div class="etiket">Bölütleme maske mAP50-95</div><div class="deger">{f(map_r.mean())}</div>'
                f'<div class="etiket">RF-DETR · YOLO26s: {f(map_y.mean())}</div></div>')
    puan = [[ICON[p["durum"]], esc(p["bilesen"]), f"%{p['agirlik']}", esc(p["not"])] for p in DURUM["puanlama"]]
    asama = []
    for a in DURUM["asamalar"]:
        ok = sum(s == "tamam" for _, s in a["adimlar"])
        maddeler = "".join(f"<li>{ICON[s]} {esc(ad)}</li>" for ad, s in a["adimlar"])
        asama.append(f'<details{" open" if a["durum"] == "devam" else ""} class="kutu" style="margin:8px 0"><summary><b>{ICON[a["durum"]]} '
                     f'{esc(a["ad"])}</b> — {ok}/{len(a["adimlar"])} adım</summary><div class="cubuk"><i style="width:{100 * ok / len(a["adimlar"]):.0f}%"></i></div>'
                     f'<ul class="liste">{maddeler}</ul></details>')
    kos = DURUM.get("seg_kosulari", [])
    kos_t = tablo(["", "Koşu", "Not"], [[ICON[s], f"<code>{esc(k)}</code>", esc(nt)] for k, s, nt in kos], sayisal_baslangic=9)
    st = SUMMARY["stats"]
    veri = [["Görüntü", n(sum(v["n_images"] for v in st.values()))],
            ["Hastalıklı bölge (çokgen)", n(sum(v["n_instances"] for v in st.values()))],
            ["Hastalık sınıfı", len(SUMMARY["classes"])],
            ["Temiz test görüntüsü", f'{SUMMARY["duplicates"]["test_clean"]} (sızan {SUMMARY["duplicates"]["test_leaked"]})']]
    return f"""<h1>Çilek Hastalığı: Sınıflandırma ve Bölütleme</h1>
<p class="alt">Derin Öğrenme-Sinir Ağları dersi · Görev 13 · Strawberry Disease Detection Dataset (Afzaal ve ark., 2021)</p>
<div class="basitce">Bir sera işletmesi, telefonla çekilen çilek fotoğraflarından <b>hangi hastalığın olduğunu</b> (sınıflandırma) ve
<b>hastalıklı bölgenin tam olarak nerede olduğunu</b> (bölütleme) otomatik bulmak istiyor. Bu sayfa ödevin her adımında ne
yapıldığını ve sonuçları gösterir; bütün sayılar gerçek çalıştırma çıktılarından otomatik alınır.</div>
<div class="izgara">{"".join(kart)}</div>
<h2>Puanlama ve durum</h2>{tablo(["", "Bileşen", "Ağırlık", "Not"], puan, sayisal_baslangic=2, sayisal_bitis=3)}
<h2>Aşamalar</h2>{"".join(asama)}
<h2>Colab bölütleme eğitimleri — {sum(s == "tamam" for _, s, _ in kos)}/{len(kos)} koşu</h2>{kos_t}
<h2>Veri özeti</h2>{tablo(["", "Değer"], veri)}"""


# ------------------------------------------------------------------ Raporlar
def raporlar() -> str:
    (SITE / "rapor").mkdir(exist_ok=True)
    out = ['<h1>Raporlar ve Teslim</h1><div class="basitce">Hocaya gidecek raporların son hâli. Raporlar bir betikle '
           'üretilir: bütün sayılar sonuç dosyalarından okunur, elle sayı yazılmaz.</div>']
    for r in DURUM["raporlar"]:
        ok = sum(s == "tamam" for _, s in r["kontrol"])
        maddeler = "".join(f"<li>{ICON[s]} {esc(m)}</li>" for m, s in r["kontrol"])
        due = datetime.strptime(r["teslim"], "%Y-%m-%d %H:%M")
        pdf = ROOT / r["pdf"]
        parca = [f'<h2>{esc(r["ad"])}</h2><p class="alt">Son teslim {due:%d.%m.%Y %H:%M} · kontrol listesi {ok}/{len(r["kontrol"])}</p>',
                 f'<div class="cubuk"><i style="width:{100 * ok / len(r["kontrol"]):.0f}%"></i></div><ul class="liste">{maddeler}</ul>']
        if pdf.exists():
            shutil.copy2(pdf, SITE / "rapor" / pdf.name)
            parca.append(f'<p><a class="dugme" href="rapor/{pdf.name}">PDF\'i aç / indir</a>'
                         f'<a href="{esc(r["colab"])}">Colab defteri</a></p>'
                         f'<object class="pdf" data="rapor/{pdf.name}" type="application/pdf">'
                         f'<p>Tarayıcınız PDF\'i gösteremiyor: <a href="rapor/{pdf.name}">indirin</a>.</p></object>')
        else:
            parca.append('<p class="kutu">Bu raporun PDF\'i henüz yok.</p>')
        out.append("".join(parca))
    return "".join(out)


# ------------------------------------------------------------------ Veri
def veri() -> str:
    st, dup = SUMMARY["stats"], SUMMARY["duplicates"]
    satir = []
    for c in sorted(C.CLASSES, key=lambda c: (C.GROUPS.index(C.GROUP_OF[c]), c)):
        satir.append([esc(c), esc(C.CLASS_TR[c]), C.GROUP_OF[c]] + [n(st[s]["images_per_label"][c]) for s in C.SPLITS]
                     + [n(st[s]["instances_per_class"][c]) for s in C.SPLITS])
    satir.append(["<b>Toplam</b>", "", ""] + [f"<b>{n(st[s]['n_images'])}</b>" for s in C.SPLITS]
                 + [f"<b>{n(st[s]['n_instances'])}</b>" for s in C.SPLITS])
    rt = SUMMARY["roundtrip"]
    return f"""<h1>Veri ve Hazırlık</h1>
<div class="basitce">2.500 fotoğraf (419×419), 7 hastalık; her hastalıklı bölge uzmanlarca çokgenle çizilmiş. Resmî eğitim / doğrulama /
test bölmesi korunmuştur. Aynı sahnenin neredeyse aynı kareleri farklı bölmelere düştüğü için test sonuçları ayrıca
<b>temiz test</b> kümesinde ({dup['test_clean']} görüntü) de verilir.</div>
<h2>Sınıflar ve bölmeler</h2>{tablo(["Sınıf", "Türkçe", "Grup", "Görüntü: eğitim", "doğr.", "test", "Çokgen: eğitim", "doğr.", "test"], satir, sayisal_baslangic=3)}
<div class="iki">{sekil("sinif_dagilimi", "Bölmelere göre sınıf dağılımı (görüntü düzeyi etiket).")}{sekil("ornekler", "Her sınıftan bir eğitim görüntüsü, uzman çokgenleri bindirilmiş.")}</div>
<h2>Etiket dönüşümü ve denetim</h2>
<ul><li>LabelMe JSON → YOLO-seg ve COCO; geri dönüş denetimi: {n(rt['n_instances'])} çokgen, en düşük IoU {f(rt['iou_yolo_min'], 4)} (YOLO) / {f(rt['iou_coco_min'], 4)} (COCO).</li>
<li>Kendini kesen 12 çokgen onarıldı, görüntü dışına taşan 4 çokgen kırpıldı.</li>
<li>Roboflow'a yükleme ve COCO dışa aktarımıyla bağımsız çapraz kontrol: ortalama IoU 0,99995.</li></ul>
{sekil("roboflow_health_check", "Roboflow Health Check ekranı.")}
<h2>Sızıntı (yakın-yinelenen görüntüler)</h2>
<p>Algısal özet (pHash) uzaklığı ≤ {dup['phash_threshold']} olan çiftler aynı sahne sayıldı: testin {dup['test_leaked']} görüntüsü
(%{f(100 * dup['test_leaked'] / st['test']['n_images'], 1)}) ve doğrulamanın {dup['val_leaked']} görüntüsü
(%{f(100 * dup['val_leaked'] / st['val']['n_images'], 1)}) eğitimde bir benzerine sahip.</p>
<div class="iki">{sekil("sizinti_esik", "Eşiğe göre sızan test/doğrulama oranı; kesikli çizgi seçilen eşik.")}{sekil("sizinti_ornek", "Bölmeler arası yakın-yinelenen örnek çiftler.")}</div>"""


# ------------------------------------------------------------------ Sınıflandırma
def siniflandirma() -> str:
    res = pd.read_csv(ROOT / "runs/cls/results.csv")
    eff = pd.read_csv(ROOT / "runs/cls/efficiency.csv")
    gr = {"YOLO26n-cls, 224": ("ana", "n", 224), "YOLO26s-cls, 224": ("ana", "s", 224), "YOLO26n-cls, 384 (ablasyon)": ("abl384", "n", 384)}
    satir = []
    for ad, (dn, b, im) in gr.items():
        d = res[(res.deney == dn) & (res.boyut == b) & (res.imgsz == im)]
        e = eff[(eff.model == f"yolo26{b}-cls") & (eff.imgsz == im)].iloc[0]
        satir.append([ad, f(e.parametre / 1e6, 2), ms(d[d.alt_kume == "tum_test"].accuracy), ms(d[d.alt_kume == "temiz_test"].accuracy),
                      ms(d[d.alt_kume == "tum_test"].macro_f1), f(e.ms_toplam_duvar, 1)])
    pc = pd.read_csv(ROOT / "runs/cls/per_class.csv")
    pc = pc[pc.alt_kume == "tum_test"]
    pcs = []
    for c in C.CLASSES:
        row = [esc(c)]
        for k in ("ana_yolo26n-cls_im224", "ana_yolo26s-cls_im224", "abl384_yolo26n-cls_im384"):
            row.append(f(pc[pc.kosu.str.startswith(k) & (pc.sinif == c)].f1.mean(), 2))
        row.append(n(pc[pc.sinif == c].support.iloc[0]))
        pcs.append(row)
    gk = pd.read_csv(ROOT / "runs/cls/grup_karisma.csv", index_col=0)
    gks = [[esc(i.replace("->", " → ")), f"%{f(100 * r['yolo26n-cls'], 1)}", f"%{f(100 * r['yolo26s-cls'], 1)}"] for i, r in gk.iterrows()]
    return f"""<h1>Sınıflandırma (1. rapor)</h1>
<div class="basitce">Model fotoğrafın tamamına bakıp hangi hastalığın olduğunu söyler. Ultralytics YOLO26 sınıflandırma modelleri
(nano ve small) üçer tohumla eğitildi; ayrıca nano modelle 384 çözünürlük denendi.</div>
<h2>Test sonuçları (3 tohum ortalaması ± std)</h2>{tablo(["Model", "Param. (M)", "Doğruluk (tüm)", "Doğruluk (temiz)", "Makro F1", "ms/görüntü"], satir)}
<h2>Sınıf bazında F1 (tüm test)</h2>{tablo(["Sınıf", "nano", "small", "nano-384", "Görüntü"], pcs)}
<h2>Karmaşıklık ve meyve/yaprak karışması</h2>
<div class="iki">{sekil("cls_karmasiklik_n", "YOLO26n-cls karmaşıklık matrisi (3 tohum toplamı, satır normalize).")}{sekil("cls_karmasiklik_s", "YOLO26s-cls karmaşıklık matrisi.")}</div>
{tablo(["Organ grupları arası", "nano", "small"], gks)}
<h2>En kötü 10 tahmin</h2>{sekil("cls_en_kotu10", "Doğrulama ile seçilen modelin en kötü 10 test tahmini.")}
<h2>Eğitim eğrileri</h2>{sekil("cls_egriler", "Eğitim kaybı ve doğrulama doğruluğu.")}"""


# ------------------------------------------------------------------ Segmentasyon
def segmentasyon() -> str:
    res = pd.read_csv(ROOT / "runs/seg/results.csv")
    eff = pd.read_csv(ROOT / "runs/seg/efficiency.csv").set_index("model")
    ad = {"yolo26n-seg": "YOLO26n-seg", "yolo26s-seg": "YOLO26s-seg", "rfdetr-seg-small": "RF-DETR Seg Small"}
    satir, ver = [], []
    for m, a in ad.items():
        d = res[(res.model == m) & (res.alt_kume == "tum_test")]
        satir.append([a, ms(d.mask_map50), ms(d.mask_map50_95), ms(d.miou), ms(d.mdice),
                      f(res[(res.model == m) & (res.alt_kume == "temiz_test")].mask_map50_95.mean())])
        e = eff.loc[m]
        ver.append([a, f(e.parametre / 1e6, 2), f(e.ms_ileri_gecis, 1), f(e.ms_uctan_uca, 1), f(e.gpu_bellek_mb, 0)])
    pc = pd.read_csv(ROOT / "runs/seg/per_class.csv").merge(res[["kosu", "model"]].drop_duplicates(), on="kosu")
    pc = pc[pc.alt_kume == "tum_test"]
    pcs = [[esc(c)] + [f(pc[(pc.model == m) & (pc.sinif == c)].iou.mean(), 2) for m in ad] for c in C.CLASSES]
    gk = pd.read_csv(ROOT / "runs/seg/grup_karisma.csv", index_col=0)
    gks = [[esc(i.replace("->", " → "))] + [f"%{f(100 * r[m], 2)}" for m in ad] for i, r in gk.iterrows()]
    capr = res[res.alt_kume == "tum_test"][["kosu", "val_mask_map50_95_kutuphane", "val_mask_map50_95_bizim"]]
    capt = [[f"<code>{esc(r.kosu)}</code>", f(r.val_mask_map50_95_kutuphane), f(r.val_mask_map50_95_bizim)] for r in capr.itertuples()]
    return f"""<h1>Bölütleme (2. rapor)</h1>
<div class="basitce">Model hastalığın adının yanında hastalıklı bölgenin sınırını piksel piksel çizer. Evrişimli YOLO26-seg (nano, small)
ile dönüştürücü tabanlı RF-DETR Seg Small aynı çözünürlükte (480), aynı T4 GPU'da ve aynı ölçüm koduyla, üçer tohumla karşılaştırıldı.</div>
<h2>Test sonuçları (3 tohum ortalaması ± std)</h2>{tablo(["Model", "mAP50", "mAP50-95", "mIoU", "Dice", "mAP50-95 (temiz)"], satir, vurgu=2)}
<h2>Hız, boyut, bellek (T4, yığın boyu 1)</h2>{tablo(["Model", "Param. (M)", "İleri geçiş ms", "Uçtan uca ms", "GPU bellek (MB)"], ver)}
<div class="iki">{sekil("seg_dogruluk_hiz", "Doğruluk–hız dengesi.")}{sekil("seg_egriler", "Eğitim kaybı ve doğrulama maske mAP50-95 (9 koşu).")}</div>
<h2>Sınıf bazında piksel IoU (tüm test)</h2>{tablo(["Sınıf"] + list(ad.values()), pcs)}
{sekil("seg_sinif_bazinda", "Sınıf bazında maske AP50-95 ve piksel IoU.")}
<h2>Sınıf karışması ve meyve/yaprak (Dikkat maddesi)</h2>{tablo(["Organ grupları arası"] + list(ad.values()), gks)}
<div class="iki">{sekil("seg_karisma_rfdetr-seg-small", "RF-DETR örnek karışma matrisi.")}{sekil("seg_karisma_yolo26s-seg", "YOLO26s-seg örnek karışma matrisi.")}</div>
<h2>En kötü 10 test görüntüsü</h2>{sekil("seg_en_kotu10", "Gerçek maske | YOLO26s | RF-DETR (doğrulama ile seçilen modeller).")}
<h2>Çapraz kontrol (doğrulama maske mAP50-95)</h2><p>Bizim ortak ölçüm kodumuz ile kütüphanenin eğitimde raporladığı değer yakın olmalı.</p>
{tablo(["Koşu", "Kütüphane", "Bizim kod"], capt)}"""


# ------------------------------------------------------------------ Günlük
def gunluk() -> str:
    md = (ROOT / "docs" / "gunluk.md").read_text()
    return '<div class="gunluk">' + markdown.markdown(md, extensions=["tables", "fenced_code"]) + "</div>"


if __name__ == "__main__":
    # site/ her seferinde sıfırdan üretilir; yalnız Vercel proje bağlantısı (.vercel/) korunur
    if SITE.exists():
        for p in SITE.iterdir():
            if p.name != ".vercel":
                shutil.rmtree(p) if p.is_dir() else p.unlink()
    (SITE / "fig").mkdir(parents=True)
    for p in FIG.glob("*.png"):
        if "duman" not in p.stem:
            shutil.copy2(p, SITE / "fig" / p.name)
    (SITE / "stil.css").write_text(CSS)
    for dosya, baslik, govde in (("index.html", "Genel Bakış", genel()), ("raporlar.html", "Raporlar", raporlar()),
                                 ("veri.html", "Veri", veri()), ("siniflandirma.html", "Sınıflandırma", siniflandirma()),
                                 ("segmentasyon.html", "Segmentasyon", segmentasyon()), ("gunluk.html", "Günlük", gunluk())):
        sayfa(dosya, baslik, govde)
    # Vercel: temiz adresler (/veri → veri.html)
    (SITE / "vercel.json").write_text(json.dumps({"cleanUrls": True}, indent=2))
    boyut = sum(p.stat().st_size for p in SITE.rglob("*") if p.is_file())
    print(f"site/ üretildi: {len(list(SITE.rglob('*.html')))} sayfa, {boyut / 1e6:.1f} MB")
