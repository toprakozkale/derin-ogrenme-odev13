"""Aşama 3 — örnek (instance) segmentasyonu: YOLO26-seg (n, s) ↔ RF-DETR Seg Small.

Eğitim, tahmin, ORTAK değerlendirme ve verimlilik ölçümü. İki model ailesi de tahminlerini aynı biçime (COCO "results":
görüntü, sınıf, skor, RLE maske) çevirir; bütün metrikler `src/metrics.py` ile aynı kodla hesaplanır.

Çıktılar (panel ve rapor bunları okur; * = git'te):
  runs/seg/train/<koşu>/                    eğitim klasörü. YOLO: weights/best.pt, results.csv*, args.yaml*;
                                            RF-DETR: checkpoint_best_total.pth, metrics.csv*, training_config.json*; DONE*
  runs/seg/preds/<koşu>_<val|test>.json.gz  ham tahminler (büyük → yalnız Drive)
  runs/seg/preds/<koşu>_test_goruntu.csv*   görüntü başına IoU (en kötü 10 bunun üzerinden)
  runs/seg/preds/<koşu>_test_ornek_karisma.csv*  IoU ≥ 0,5 eşleşmeli örnek karışma matrisi (8 × 8)
  runs/seg/results.csv*                     koşu × alt küme: mask mAP50, mAP50-95, mIoU, mDice, fg IoU
  runs/seg/per_class.csv*                   koşu × alt küme × sınıf: AP50, AP50-95, IoU, Dice
  runs/seg/esikler.csv*                     piksel metrikleri için val ile seçilen güven eşiği
  runs/seg/efficiency.csv* · env.json*
"""
from __future__ import annotations

import gzip
import json
import os
import platform
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import metrics as M

YOLO_DATA = C.PROCESSED / "yoloseg"
COCO_DATA = C.PROCESSED / "coco"
COCO_DIR = {"train": "train", "val": "valid", "test": "test"}
OUT = C.RUNS / "seg"
SMOKE_OUT = C.RUNS / "seg_duman"      # duman testi çıktıları: asıl sonuçlarla ASLA karışmaz, panel okumaz
WEIGHTS = C.ROOT / "weights"

# ---- Deney düzeni (12 Kas 2026 kararları: ortak çözünürlük 480, RF-DETR Seg Small) -----------------------------------
IMGSZ = 480                 # 419×419 görüntülere en yakın, iki paketin kuralına birlikte uyan boyut (YOLO: 32'nin, RF-DETR: 24'ün katı)
SEEDS = tuple(int(s) for s in os.environ.get("ODEV13_SEG_SEEDS", "0,1,2").split(","))
YOLO_FAMILY = "yolo26"
RF_CLASS = "RFDETRSegSmall"
CONF_MIN = 0.001            # mAP için tahminler bu skordan itibaren tutulur (her iki model)
MAX_DET = 100               # görüntü başına en çok tahmin (RF-DETR Seg Small'ın sorgu sayısı da 100)
THRESHOLDS = tuple(np.round(np.arange(0.05, 0.951, 0.05), 2))   # piksel metrikleri için val'de taranan güven eşikleri

# Eşit bütçe: en çok 100 epoch, 20 epoch iyileşme yoksa erken durdurma, etkin batch 16 (RF-DETR: 4 × 4 birikim).
YOLO_TRAIN_ARGS = dict(epochs=100, patience=20, batch=16, optimizer="auto", pretrained=True, deterministic=True,
                       workers=2, plots=True, exist_ok=True, save_period=-1)
RF_TRAIN_ARGS = dict(epochs=100, early_stopping=True, early_stopping_patience=20, batch_size=4, grad_accum_steps=4,
                     num_workers=2, checkpoint_interval=1000,   # ara arşiv yok (her biri ~535 MB olurdu)
                     tensorboard=False)
RF_KEEP = ("checkpoint_best_total.pth", "metrics.csv", "training_config.json", "DONE")


def use_smoke_output(on: bool = True) -> Path:
    """Duman testi modunda tüm çıktılar runs/seg_duman/ altına yazılır."""
    global OUT
    OUT = SMOKE_OUT if on else C.RUNS / "seg"
    return OUT


@dataclass(frozen=True)
class RunSpec:
    arch: str            # "yolo" | "rfdetr"
    size: str            # YOLO: "n" | "s"; RF-DETR: "small"
    seed: int
    imgsz: int = IMGSZ
    tag: str = "ana"

    @property
    def model(self) -> str:
        return f"{YOLO_FAMILY}{self.size}-seg" if self.arch == "yolo" else f"rfdetr-seg-{self.size}"

    @property
    def name(self) -> str:
        return f"{self.tag}_{self.model}_im{self.imgsz}_s{self.seed}"

    @property
    def dir(self) -> Path:
        return OUT / "train" / self.name

    @property
    def weights(self) -> Path:
        return self.dir / ("weights/best.pt" if self.arch == "yolo" else "checkpoint_best_total.pth")


MODELS = (("yolo", "n"), ("yolo", "s"), ("rfdetr", "small"))
# Sıra seed'e göre: önce her modelin seed 0'ı biter → Colab kotası yetmezse elde en az bir tam karşılaştırma olur.
MAIN_RUNS = [RunSpec(a, sz, s) for s in SEEDS for a, sz in MODELS]


def is_done(spec: RunSpec) -> bool:
    return (spec.dir / "DONE").exists() and spec.weights.exists()


def is_finished(spec: RunSpec) -> bool:
    """Eğitim bitti mi (ağırlık bu bilgisayarda olmasa da: DONE işareti git'le gelir)."""
    return (spec.dir / "DONE").exists()


def _rf_device(device) -> str | None:
    if device is None:
        return None
    d = str(device).lower()
    return "cuda" if d.isdigit() or d.startswith("cuda") else d


# ----------------------------------------------------------------------------- eğitim

def pretrained_yolo(model: str) -> Path:
    from ultralytics.utils.downloads import attempt_download_asset
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    p = WEIGHTS / f"{model}.pt"
    if not p.exists():
        attempt_download_asset(str(p))
    return p


def make_coco_mini(dst: Path, per_class: dict | None = None) -> Path:
    """Duman testi için küçük COCO alt kümesi: her bölmede her sınıftan birkaç görüntü."""
    per_class = per_class or {"train": 3, "valid": 2, "test": 2}
    if (dst / "train" / "_annotations.coco.json").exists():
        return dst
    for sp, k in per_class.items():
        d = json.loads((COCO_DATA / sp / "_annotations.coco.json").read_text())
        by_cat: dict[int, list[int]] = {}
        for a in d["annotations"]:
            by_cat.setdefault(a["category_id"], []).append(a["image_id"])
        ids = set()
        for l in by_cat.values():
            ids |= set(sorted(set(l))[:k])
        (dst / sp).mkdir(parents=True, exist_ok=True)
        d["images"] = [i for i in d["images"] if i["id"] in ids]
        d["annotations"] = [a for a in d["annotations"] if a["image_id"] in ids]
        for i in d["images"]:
            shutil.copy2(COCO_DATA / sp / i["file_name"], dst / sp / i["file_name"])
        (dst / sp / "_annotations.coco.json").write_text(json.dumps(d))
    return dst


def _cleanup_rf(spec: RunSpec) -> float:
    """RF-DETR bitince yalnız en iyi ağırlık + günlükler kalır (last.ckpt ~535 MB, diğerleri ~134 MB). Silinen MB."""
    freed = 0.0
    for f in spec.dir.iterdir():
        if f.is_file() and f.name not in RF_KEEP:
            freed += f.stat().st_size / 1e6
            f.unlink()
    return freed


def train(spec: RunSpec, device=None, smoke: bool = False, **override) -> Path:
    """Bir koşuyu eğitir. Bitmişse atlar; yarıda kaldıysa (Colab bağlantısı koptuysa) kaldığı yerden devam eder.

    Devam ederken ayarlar DEĞİŞTİRİLMEMELİ (özellikle çözünürlük) — RF-DETR farklı ayarla kaldığı yerden yükleyemez.
    """
    if is_done(spec):
        print(f"✓ {spec.name}: zaten bitmiş, atlanıyor")
        return spec.weights
    spec.dir.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if spec.arch == "yolo":
        from ultralytics import YOLO
        last = spec.dir / "weights" / "last.pt"
        if last.exists():
            print(f"↻ {spec.name}: yarıda kalmış, kaldığı yerden devam ediliyor")
            YOLO(str(last)).train(resume=True)
        else:
            args = {**YOLO_TRAIN_ARGS, **override}
            if smoke:
                args.update(epochs=1, fraction=0.05, batch=4, workers=0, plots=False)
            print(f"▶ {spec.name}: eğitim başlıyor ({args['epochs']} epoch, imgsz={spec.imgsz}, seed={spec.seed})")
            YOLO(str(pretrained_yolo(spec.model))).train(
                data=str(YOLO_DATA / "data.yaml"), imgsz=spec.imgsz, seed=spec.seed, device=device,
                project=str(OUT / "train"), name=spec.name, **args)
        freed = 0.0
    else:
        import rfdetr
        args = {**RF_TRAIN_ARGS, **override}
        data = COCO_DATA
        if smoke:
            args.update(epochs=1, batch_size=2, grad_accum_steps=1, num_workers=0)
            data = make_coco_mini(SMOKE_OUT / "coco_mini")
        last = spec.dir / "last.ckpt"
        if last.exists():
            print(f"↻ {spec.name}: yarıda kalmış, kaldığı yerden devam ediliyor")
            args["resume"] = str(last)
        else:
            print(f"▶ {spec.name}: eğitim başlıyor ({args['epochs']} epoch, çözünürlük={spec.imgsz}, seed={spec.seed})")
        spec.dir.mkdir(parents=True, exist_ok=True)
        model = getattr(rfdetr, RF_CLASS)()
        model.train(dataset_dir=str(data), output_dir=str(spec.dir), resolution=spec.imgsz, seed=spec.seed,
                    device=_rf_device(device), **args)
        freed = _cleanup_rf(spec)
    (spec.dir / "DONE").write_text(json.dumps({"bitis": time.strftime("%Y-%m-%d %H:%M:%S"),
                                               "sure_sn_bu_oturum": round(time.time() - t0, 1),
                                               "silinen_mb": round(freed, 1), "smoke": smoke, **asdict(spec)},
                                              ensure_ascii=False))
    return spec.weights


def train_log(spec: RunSpec) -> pd.DataFrame:
    """Epoch başına ortak sütunlar: epoch (1'den), egitim_kaybi, val_mask_map50, val_mask_map50_95."""
    if spec.arch == "yolo":
        p = spec.dir / "results.csv"
        if not p.exists():
            return pd.DataFrame()
        df = pd.read_csv(p)
        df.columns = [c.strip() for c in df.columns]
        loss_cols = [c for c in df.columns if c.startswith("train/") and c.endswith("_loss")]
        return pd.DataFrame({"epoch": df["epoch"], "egitim_kaybi": df[loss_cols].sum(1),
                             "val_mask_map50": df.get("metrics/mAP50(M)"),
                             "val_mask_map50_95": df.get("metrics/mAP50-95(M)")})
    p = spec.dir / "metrics.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p).groupby("epoch").first().reset_index()     # eğitim ve val satırları ayrı yazılıyor
    pick = lambda a, b: df[[c for c in (a, b) if c in df]].max(1)  # noqa: E731 — en iyi ağırlık EMA/normalden büyüğü
    return pd.DataFrame({"epoch": df["epoch"] + 1, "egitim_kaybi": df.get("train/loss"),
                         "val_mask_map50": pick("val/segm_mAP_50", "val/ema_segm_mAP_50"),
                         "val_mask_map50_95": pick("val/segm_mAP_50_95", "val/ema_segm_mAP_50_95")}).dropna(how="all", subset=["val_mask_map50_95", "egitim_kaybi"])


def best_val_map(spec: RunSpec) -> float:
    log = train_log(spec)
    return float(log["val_mask_map50_95"].max()) if len(log) else float("nan")


def select_by_val(specs: list[RunSpec]) -> RunSpec:
    """Doğrulama mask mAP50-95'i en yüksek koşu (seçim yalnız val ile; test bakılmaz)."""
    done = [s for s in specs if is_finished(s)]
    return max(done, key=best_val_map)


# ----------------------------------------------------------------------------- tahmin

_GT_CACHE: dict[str, dict] = {}


def load_gt(split: str) -> dict:
    if split not in _GT_CACHE:
        _GT_CACHE[split] = json.loads((COCO_DATA / COCO_DIR[split] / "_annotations.coco.json").read_text())
    return _GT_CACHE[split]


def split_images(split: str, limit: int | None = None) -> list[dict]:
    imgs = sorted(load_gt(split)["images"], key=lambda i: i["file_name"])
    return imgs[:limit] if limit else imgs


def _load_model(spec: RunSpec):
    if spec.arch == "yolo":
        from ultralytics import YOLO
        return YOLO(str(spec.weights))
    import rfdetr
    return getattr(rfdetr, RF_CLASS)(pretrain_weights=str(spec.weights), num_classes=len(C.CLASSES),
                                     resolution=spec.imgsz)


def _predict_batch(spec: RunSpec, model, paths: list[str], device, conf: float) -> list[tuple]:
    """Her görüntü için (maskeler [N,H,W] bool, sınıf [N] (bizim kimlik), skor [N])."""
    out = []
    if spec.arch == "yolo":
        names = model.names
        for r in model.predict(source=paths, imgsz=spec.imgsz, conf=conf, max_det=MAX_DET, retina_masks=True,
                               device=device, verbose=False, stream=True, batch=len(paths)):
            h, w = r.orig_shape
            if r.masks is None or len(r.boxes) == 0:
                out.append((np.zeros((0, h, w), bool), np.zeros(0, int), np.zeros(0)))
                continue
            masks = r.masks.data.cpu().numpy() > 0.5
            if masks.shape[1:] != (h, w):
                import cv2
                masks = np.stack([cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
                                  for m in masks])
            cls = np.array([C.CLASS_TO_ID[names[int(k)]] for k in r.boxes.cls.cpu().numpy()], dtype=int)
            out.append((masks, cls, r.boxes.conf.cpu().numpy().astype(float)))
    else:
        dets = model.predict(paths, threshold=conf, include_source_image=False)
        dets = dets if isinstance(dets, list) else [dets]
        for d in dets:
            if len(d) == 0 or d.mask is None:
                out.append((np.zeros((0, 0, 0), bool), np.zeros(0, int), np.zeros(0)))
                continue
            names = np.asarray(d.data["class_name"])
            # RF-DETR düşük eşikte "__background__" (nesne yok) yuvasını da döndürüyor → hastalık değil, atılır
            keep = np.array([n in C.CLASS_TO_ID for n in names], dtype=bool)
            idx = np.where(keep)[0]
            order = idx[np.argsort(-d.confidence[idx], kind="stable")][:MAX_DET]
            cls = np.array([C.CLASS_TO_ID[n] for n in names[order]], dtype=int)
            out.append((np.asarray(d.mask)[order].astype(bool), cls, np.asarray(d.confidence)[order].astype(float)))
    return out


def preds_path(spec: RunSpec, split: str) -> Path:
    return OUT / "preds" / f"{spec.name}_{split}.json.gz"


def predict(spec: RunSpec, split: str = "test", device=None, limit: int | None = None, batch: int = 8,
            reuse: bool = True) -> dict:
    """Bir bölmenin tahminlerini COCO results biçiminde üretir ve kaydeder.

    reuse=True: kayıtlı tahmin ağırlıktan yeniyse yeniden hesaplanmaz (Colab bağlantısı koparsa zaman kazandırır).
    """
    p = preds_path(spec, split)
    if reuse and p.exists() and (not spec.weights.exists() or p.stat().st_mtime > spec.weights.stat().st_mtime):
        d = json.loads(gzip.decompress(p.read_bytes()))
        if limit is None or len(d["image_ids"]) == len(split_images(split, limit)):
            return d
    model = _load_model(spec)
    imgs = split_images(split, limit)
    folder = COCO_DATA / COCO_DIR[split]
    results = []
    for i in range(0, len(imgs), batch):
        chunk = imgs[i:i + batch]
        for im, (masks, cls, scores) in zip(chunk, _predict_batch(spec, model, [str(folder / x["file_name"]) for x in chunk],
                                                                    device, CONF_MIN)):
            for m, c, s in zip(masks, cls, scores):
                if m.any():
                    results.append({"image_id": im["id"], "category_id": int(c) + 1,
                                    "segmentation": M.mask_to_rle(m), "score": round(float(s), 5)})
    d = {"kosu": spec.name, "split": split, "image_ids": [im["id"] for im in imgs], "results": results}
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(gzip.compress(json.dumps(d).encode()))
    return d


# ----------------------------------------------------------------------------- ortak değerlendirme

def _gt_by_image(split: str) -> dict[int, tuple[list, np.ndarray, list]]:
    """Görüntü id → (gerçek maskeler, sınıflar, RLE'ler)."""
    from pycocotools import mask as mu
    gt = load_gt(split)
    size = {im["id"]: (im["height"], im["width"]) for im in gt["images"]}
    out: dict[int, tuple[list, list, list]] = {i: ([], [], []) for i in size}
    for a in gt["annotations"]:
        h, w = size[a["image_id"]]
        rle = mu.merge(mu.frPyObjects(a["segmentation"], h, w))
        m = mu.decode(rle).astype(bool)
        rle["counts"] = rle["counts"].decode("ascii")
        out[a["image_id"]][0].append(m); out[a["image_id"]][1].append(a["category_id"] - 1); out[a["image_id"]][2].append(rle)
    return {k: (v[0], np.array(v[1], dtype=int), v[2]) for k, v in out.items()}


def _pred_by_image(results: list[dict], min_score: float) -> dict[int, list[dict]]:
    by: dict[int, list[dict]] = {}
    for r in results:
        if r["score"] >= min_score:
            by.setdefault(r["image_id"], []).append(r)
    return by


def _sems(gt_img, preds: list[dict], shape, thresholds) -> tuple[np.ndarray, dict[float, np.ndarray]]:
    g_masks, g_cls, _ = gt_img
    g_sem = M.semantic_map(g_masks, g_cls, order_key=[-m.sum() for m in g_masks], shape=shape) if g_masks \
        else np.full(shape, M.BG, np.uint8)
    p_masks = [M.rle_to_mask(r["segmentation"]) for r in preds]
    p_cls = np.array([r["category_id"] - 1 for r in preds], dtype=int)
    p_sc = np.array([r["score"] for r in preds])
    sems = {}
    for t in thresholds:
        k = np.where(p_sc >= t)[0]
        sems[t] = M.semantic_map([p_masks[i] for i in k], p_cls[k], order_key=p_sc[k], shape=shape) if len(k) \
            else np.full(shape, M.BG, np.uint8)
    return g_sem, sems


def choose_threshold(spec: RunSpec, val_preds: dict) -> tuple[float, pd.DataFrame]:
    """Piksel metrikleri (mIoU, Dice) bir güven eşiği ister: val'de mIoU'yu en yükselten eşik seçilir (test bakılmaz)."""
    gts = _gt_by_image("val")
    shape_of = {im["id"]: (im["height"], im["width"]) for im in load_gt("val")["images"]}
    by = _pred_by_image(val_preds["results"], min(THRESHOLDS))
    cms = {t: np.zeros((M.BG + 1, M.BG + 1), np.int64) for t in THRESHOLDS}
    for iid in val_preds["image_ids"]:
        g_sem, sems = _sems(gts[iid], by.get(iid, []), shape_of[iid], THRESHOLDS)
        for t in THRESHOLDS:
            cms[t] += M.pixel_confusion(g_sem, sems[t])
    tab = pd.DataFrame([{"esik": t, "val_miou": M.seg_semantic_metrics(cms[t])["miou"]} for t in THRESHOLDS])
    best = float(tab.loc[tab["val_miou"].idxmax(), "esik"])
    return best, tab


def clean_test_ids() -> set[int]:
    p = C.DERIVED / "splits" / "test_clean.txt"
    stems = {Path(x).stem for x in p.read_text().split()} if p.exists() else set()
    return {im["id"] for im in load_gt("test")["images"] if Path(im["file_name"]).stem in stems}


def evaluate_run(spec: RunSpec, device=None, limit: int | None = None) -> tuple[list[dict], list[pd.DataFrame], dict]:
    """Val'de eşik seçer, testte bütün metrikleri hesaplar (tüm test + temiz test). Ara tabloları preds/'e yazar."""
    val = predict(spec, "val", device, limit)
    test = predict(spec, "test", device, limit)
    thr, thr_tab = choose_threshold(spec, val)
    # Çapraz kontrol: bizim kodla val mask mAP50-95 ↔ kütüphanenin eğitimde raporladığı en iyi val değeri.
    # Yakın çıkmalı (değerlendirme ayrıntıları yüzünden birebir değil); çok farklıysa sınıf eşlemesinde hata vardır.
    val_map_ours = M.coco_mask_map(load_gt("val"), val["results"], img_ids=val["image_ids"])["map50_95"]
    gt = load_gt("test")
    gts = _gt_by_image("test")
    shape_of = {im["id"]: (im["height"], im["width"]) for im in gt["images"]}
    stem_of = {im["id"]: Path(im["file_name"]).stem for im in gt["images"]}
    clean = clean_test_ids()
    by = _pred_by_image(test["results"], thr)
    cm_pix = {"tum_test": np.zeros((M.BG + 1,) * 2, np.int64), "temiz_test": np.zeros((M.BG + 1,) * 2, np.int64)}
    cm_ins = {k: np.zeros((M.BG + 1,) * 2, int) for k in cm_pix}
    per_img = []
    for iid in test["image_ids"]:
        preds = by.get(iid, [])
        g_sem, sems = _sems(gts[iid], preds, shape_of[iid], [thr])
        cm = M.pixel_confusion(g_sem, sems[thr])
        pairs = M.match_instances(gts[iid][2], gts[iid][1], [r["segmentation"] for r in preds],
                                  [r["category_id"] - 1 for r in preds])
        ci = M.instance_confusion(pairs)
        for k in (("tum_test", "temiz_test") if iid in clean else ("tum_test",)):
            cm_pix[k] += cm; cm_ins[k] += ci
        fg = M.seg_semantic_metrics(cm)["fg_iou"]
        per_img.append({"stem": stem_of[iid], "image_id": iid, "temiz": iid in clean, "iou_goruntu": M.image_iou(g_sem, sems[thr]),
                        "fg_iou": fg if not np.isnan(fg) else 1.0, "n_gercek": len(gts[iid][1]), "n_tahmin": len(preds),
                        "gercek_siniflar": ";".join(sorted({C.CLASSES[c] for c in gts[iid][1]})),
                        "tahmin_siniflar": ";".join(sorted({C.CLASSES[r['category_id'] - 1] for r in preds}))})
    (OUT / "preds").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_img).to_csv(OUT / "preds" / f"{spec.name}_test_goruntu.csv", index=False)
    pd.DataFrame(cm_ins["tum_test"], index=M.SEG_LABELS, columns=M.SEG_LABELS).to_csv(
        OUT / "preds" / f"{spec.name}_test_ornek_karisma.csv")
    rows, pcs = [], []
    for subset, ids in (("tum_test", test["image_ids"]), ("temiz_test", [i for i in test["image_ids"] if i in clean])):
        if not ids:
            continue
        mp = M.coco_mask_map(gt, test["results"], img_ids=ids)
        sm = M.seg_semantic_metrics(cm_pix[subset])
        gr = M.group_confusion_rate(cm_ins[subset][:M.BG, :M.BG])
        ins = cm_ins[subset]
        rows.append({"kosu": spec.name, "model": spec.model, "mimari": spec.arch, "boyut": spec.size, "seed": spec.seed,
                     "imgsz": spec.imgsz, "deney": spec.tag, "alt_kume": subset, "n": len(ids),
                     "mask_map50": mp["map50"], "mask_map50_95": mp["map50_95"], "miou": sm["miou"], "mdice": sm["mdice"],
                     "fg_iou": sm["fg_iou"], "esik": thr, "val_mask_map50_95_kutuphane": best_val_map(spec),
                     "val_mask_map50_95_bizim": val_map_ours,
                     "ornek_eslesen": int(ins[:M.BG, :M.BG].sum()), "ornek_kacirilan": int(ins[:M.BG, M.BG].sum()),
                     "ornek_fazladan": int(ins[M.BG, :M.BG].sum()),
                     "sinif_dogru_eslesen_orani": float(np.trace(ins[:M.BG, :M.BG]) / max(ins[:M.BG, :M.BG].sum(), 1)),
                     "gruplar_arasi_karisma": gr["gruplar_arasi_toplam"]})
        pc = mp["per_class"].join(sm["per_class"].loc[list(C.CLASSES), ["iou", "dice", "gt_piksel"]])
        pc = pc.reset_index(names="sinif")
        pc.insert(0, "alt_kume", subset); pc.insert(0, "kosu", spec.name)
        pcs.append(pc)
    thr_tab.insert(0, "kosu", spec.name)
    return rows, pcs, {"kosu": spec.name, "esik": thr, "tablo": thr_tab}


def evaluate_all(specs: list[RunSpec], device=None, limit: int | None = None) -> pd.DataFrame:
    rows, pcs, thrs = [], [], []
    for s in specs:
        if not s.weights.exists() and not preds_path(s, "test").exists():
            print(f"– {s.name}: ağırlık yok, atlanıyor"); continue
        t0 = time.time()
        r, p, th = evaluate_run(s, device, limit)
        rows += r; pcs += p; thrs.append(th["tablo"])
        print(f"✓ {s.name}: değerlendirildi (eşik {th['esik']}, {time.time() - t0:.0f} sn)")
    res = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "results.csv", index=False)
    if pcs:
        pd.concat(pcs).to_csv(OUT / "per_class.csv", index=False)
        pd.concat(thrs).to_csv(OUT / "esikler.csv", index=False)
    return res


def summary_table(res: pd.DataFrame) -> pd.DataFrame:
    """Model × alt küme için 'ortalama ± std (seed sayısı)'."""
    g = res.groupby(["model", "imgsz", "alt_kume"], sort=False)
    out = g.agg(seed=("seed", "count")).reset_index()
    for m in ("mask_map50", "mask_map50_95", "miou", "mdice", "fg_iou"):
        out[m] = g[m].apply(lambda v: M.mean_std(v.to_numpy())).to_numpy()
    return out


def load_per_image(spec: RunSpec) -> pd.DataFrame:
    return pd.read_csv(OUT / "preds" / f"{spec.name}_test_goruntu.csv")


def load_instance_confusion(spec: RunSpec) -> np.ndarray:
    return pd.read_csv(OUT / "preds" / f"{spec.name}_test_ornek_karisma.csv", index_col=0).to_numpy()


def worst_images(specs: list[RunSpec], k: int = 10) -> pd.DataFrame:
    """Verilen koşuların (ör. val ile seçilen en iyi YOLO ve en iyi RF-DETR) ortalama görüntü IoU'su en düşük k test görüntüsü."""
    tabs = []
    for s in specs:
        t = load_per_image(s)[["stem", "image_id", "iou_goruntu", "gercek_siniflar", "tahmin_siniflar"]].copy()
        t = t.rename(columns={"iou_goruntu": f"iou_{s.arch}", "tahmin_siniflar": f"tahmin_{s.arch}"})
        tabs.append(t.set_index(["stem", "image_id", "gercek_siniflar"]))
    w = pd.concat(tabs, axis=1).reset_index()
    w["iou_ortalama"] = w[[f"iou_{s.arch}" for s in specs]].mean(1)
    return w.sort_values("iou_ortalama").head(k).reset_index(drop=True)


def semantic_from_results(preds: list[dict], shape: tuple[int, int], thr: float) -> np.ndarray:
    """Bir görüntünün kayıtlı tahminlerinden (COCO results) piksel sınıf haritası; eşiğin altındakiler atılır."""
    keep = [r for r in preds if r["score"] >= thr]
    _, sems = _sems(([], np.zeros(0, int), []), keep, shape, [thr])
    return sems[thr]


def masks_for(spec: RunSpec, split: str, image_id: int, thr: float) -> np.ndarray:
    """Kayıtlı tahminlerden bir görüntünün piksel sınıf haritası (şekiller için)."""
    d = json.loads(gzip.decompress(preds_path(spec, split).read_bytes()))
    im = next(i for i in load_gt(split)["images"] if i["id"] == image_id)
    preds = [r for r in d["results"] if r["image_id"] == image_id]
    return semantic_from_results(preds, (im["height"], im["width"]), thr)


def parse_run(name: str) -> RunSpec | None:
    """Klasör adından RunSpec ('{tag}_{model}_im{imgsz}_s{seed}'); tanınmazsa None."""
    import re
    m = re.fullmatch(r"(?P<tag>[^_]+)_(?P<model>.+)_im(?P<imgsz>\d+)_s(?P<seed>\d+)", name)
    if not m:
        return None
    model = m["model"]
    if model.startswith(YOLO_FAMILY) and model.endswith("-seg"):
        arch, size = "yolo", model[len(YOLO_FAMILY):-len("-seg")]
    elif model.startswith("rfdetr-seg-"):
        arch, size = "rfdetr", model[len("rfdetr-seg-"):]
    else:
        return None
    return RunSpec(arch, size, int(m["seed"]), int(m["imgsz"]), m["tag"])


def gt_semantic(split: str, image_id: int) -> np.ndarray:
    im = next(i for i in load_gt(split)["images"] if i["id"] == image_id)
    g = _gt_by_image(split)[image_id]
    g_sem, _ = _sems(g, [], (im["height"], im["width"]), [])
    return g_sem


# ----------------------------------------------------------------------------- verimlilik

def efficiency(spec: RunSpec, device=None, n_warmup: int = 20, n_images: int | None = None,
               thr: float | None = None) -> dict:
    """Parametre sayısı, ms/görüntü (batch = 1) ve tepe GPU belleği.

    ms_ileri_gecis: yalnız ağın hesabı, GERÇEK tahmin sırasında ölçülür (ön/son işlem hariç). YOLO bunu kendisi raporlar
                    (r.speed["inference"]); RF-DETR'da ağın girişine/çıkışına zamanlayıcı kancası takılır.
    ms_uctan_uca:   dosya yolundan maskelere kadar her şey (okuma, ön işlem, ağ, son işlem, maskeyi orijinal boyuta getirme);
                    güven eşiği olarak val'de seçilen eşik kullanılır (gerçek kullanım koşulu).
    """
    import torch
    model = _load_model(spec)
    net = model.model if spec.arch == "yolo" else model.model.model
    n_params = int(sum(p.numel() for p in net.parameters()))
    cuda = torch.cuda.is_available() and str(device).lower() != "cpu"
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    thr = thr if thr is not None else 0.5
    folder = COCO_DATA / COCO_DIR["test"]
    paths = [str(folder / im["file_name"]) for im in split_images("test", n_images)]

    fw: list[float] = []
    if spec.arch == "yolo":
        def run(p):
            r = model.predict(p, imgsz=spec.imgsz, conf=thr, max_det=MAX_DET, retina_masks=True, device=device,
                              verbose=False)[0]
            if r.masks is not None:
                r.masks.data.cpu().numpy()          # maskeleri CPU'ya almak da uçtan uca sürenin parçası
            return r.speed["inference"]
    else:
        start = [0.0]
        fw_tmp: list[float] = []

        def _pre(module, inputs):           # kancalar None döndürmeli (aksi halde PyTorch girdiyi/çıktıyı değiştirir)
            sync(); start[0] = time.perf_counter()

        def _post(module, inputs, output):
            sync(); fw_tmp.append((time.perf_counter() - start[0]) * 1000)

        hooks = [net.register_forward_pre_hook(_pre), net.register_forward_hook(_post)]

        def run(p):
            fw_tmp.clear()
            _predict_batch(spec, model, [p], device, thr)
            return fw_tmp[-1] if fw_tmp else float("nan")
    for p in paths[:n_warmup]:
        run(p)
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    wall = []
    for p in paths:
        sync(); t = time.perf_counter()
        fw.append(run(p))
        sync(); wall.append((time.perf_counter() - t) * 1000)
    mem = torch.cuda.max_memory_allocated() / 2**20 if cuda else float("nan")
    if spec.arch != "yolo":
        for h in hooks:
            h.remove()
    return {"kosu": spec.name, "model": spec.model, "mimari": spec.arch, "imgsz": spec.imgsz, "parametre": n_params,
            "ms_ileri_gecis": float(np.nanmean(fw)), "ms_uctan_uca": float(np.mean(wall)),
            "ms_uctan_uca_medyan": float(np.median(wall)), "gpu_bellek_mb": mem, "esik": thr,
            "n_goruntu": len(paths), "cihaz": device_name(device)}


def device_name(device=None) -> str:
    import torch
    if str(device).lower() == "cpu":
        return "CPU"
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "Apple MPS"
    return "CPU"


def write_env() -> dict:
    import importlib.metadata as md
    import torch
    v = lambda p: md.version(p) if _has(p) else None  # noqa: E731
    env = {"python": platform.python_version(), "torch": torch.__version__, "ultralytics": v("ultralytics"),
           "rfdetr": v("rfdetr"), "supervision": v("supervision"), "pycocotools": v("pycocotools"),
           "cuda": torch.version.cuda, "cihaz": device_name(), "tarih": time.strftime("%Y-%m-%d %H:%M:%S"),
           "imgsz": IMGSZ, "seedler": list(SEEDS), "yolo_egitim": YOLO_TRAIN_ARGS, "rfdetr_model": RF_CLASS,
           "rfdetr_egitim": RF_TRAIN_ARGS, "conf_min": CONF_MIN, "max_det": MAX_DET}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "env.json").write_text(json.dumps(env, indent=2, ensure_ascii=False))
    return env


def _has(pkg: str) -> bool:
    import importlib.metadata as md
    try:
        md.version(pkg)
        return True
    except md.PackageNotFoundError:
        return False
