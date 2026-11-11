"""Aşama 2 — YOLO-cls eğitimi, tahmin ve değerlendirme.

Çıktılar (panel ve rapor bunları okur):
  runs/cls/train/<koşu>/               Ultralytics eğitim klasörü (weights/best.pt, results.csv, DONE işareti)
  runs/cls/preds/<koşu>_test.csv       test tahminleri (her görüntü: gerçek, tahmin, 7 olasılık)
  runs/cls/results.csv                 koşu × alt küme (tüm test / temiz test) metrikleri
  runs/cls/per_class.csv               koşu × alt küme × sınıf P/R/F1
  runs/cls/efficiency.csv              parametre sayısı, ms/görüntü
  runs/cls/env.json                    sürümler ve GPU bilgisi
"""
from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .metrics import cls_metrics

CLS_DATA = C.PROCESSED / "cls"
OUT = C.RUNS / "cls"
SMOKE_OUT = C.RUNS / "cls_duman"      # duman testi çıktıları: asıl sonuçlarla ASLA karışmaz, panel okumaz
WEIGHTS = C.ROOT / "weights"          # ön-eğitimli ağırlıklar (git'e girmez)


def use_smoke_output(on: bool = True) -> Path:
    """Duman testi modunda tüm çıktılar runs/cls_duman/ altına yazılır."""
    global OUT
    OUT = SMOKE_OUT if on else C.RUNS / "cls"
    return OUT

MODEL_FAMILY = "yolo26"               # Kasım 2026'daki güncel Ultralytics ailesi
TRAIN_ARGS = dict(                    # rapordaki hiperparametre tablosu bu sözlükten üretilir
    epochs=100, patience=20, batch=64, optimizer="auto", pretrained=True,
    deterministic=True, workers=2, plots=True, exist_ok=True, save_period=-1,
)
SLUG_TO_ID = {C.CLASS_SLUG[c]: i for i, c in enumerate(C.CLASSES)}


@dataclass(frozen=True)
class RunSpec:
    size: str            # "n" veya "s"
    seed: int
    imgsz: int = 224
    tag: str = "ana"     # "ana" ya da ablasyon adı

    @property
    def model(self) -> str:
        return f"{MODEL_FAMILY}{self.size}-cls"

    @property
    def name(self) -> str:
        return f"{self.tag}_{self.model}_im{self.imgsz}_s{self.seed}"

    @property
    def dir(self) -> Path:
        return OUT / "train" / self.name


MAIN_RUNS = [RunSpec(sz, s) for sz in ("n", "s") for s in (0, 1, 2)]
ABLATION_RUNS = [RunSpec("n", s, imgsz=384, tag="abl384") for s in (0, 1, 2)]


def pretrained(model: str) -> Path:
    """Ön-eğitimli ağırlığı weights/ klasörüne indirir (çalışma dizinini kirletmemek için)."""
    from ultralytics.utils.downloads import attempt_download_asset
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    p = WEIGHTS / f"{model}.pt"
    if not p.exists():
        attempt_download_asset(str(p))
    return p


def is_done(spec: RunSpec) -> bool:
    return (spec.dir / "DONE").exists() and (spec.dir / "weights" / "best.pt").exists()


def train(spec: RunSpec, device=None, smoke: bool = False, **override) -> Path:
    """Bir koşuyu eğitir. Bitmişse atlar; yarıda kaldıysa (Colab bağlantısı koptuysa) kaldığı yerden devam eder."""
    from ultralytics import YOLO
    best, last = spec.dir / "weights" / "best.pt", spec.dir / "weights" / "last.pt"
    if is_done(spec):
        print(f"✓ {spec.name}: zaten bitmiş, atlanıyor")
        return best
    t0 = time.time()
    if last.exists():
        print(f"↻ {spec.name}: yarıda kalmış, kaldığı yerden devam ediliyor")
        YOLO(str(last)).train(resume=True)
    else:
        args = {**TRAIN_ARGS, **override}
        if smoke:
            args.update(epochs=1, fraction=0.05, batch=16, workers=0, plots=False)
        print(f"▶ {spec.name}: eğitim başlıyor ({args['epochs']} epoch, imgsz={spec.imgsz}, seed={spec.seed})")
        YOLO(str(pretrained(spec.model))).train(
            data=str(CLS_DATA), imgsz=spec.imgsz, seed=spec.seed, device=device,
            project=str(OUT / "train"), name=spec.name, **args)
    (spec.dir / "DONE").write_text(json.dumps({"bitis": time.strftime("%Y-%m-%d %H:%M:%S"),
                                               "sure_sn_bu_oturum": round(time.time() - t0, 1),
                                               "smoke": smoke, **asdict(spec)}, ensure_ascii=False))
    return best


def train_log(spec: RunSpec) -> pd.DataFrame:
    p = spec.dir / "results.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    df.columns = [c.strip() for c in df.columns]
    return df


def best_val_acc(spec: RunSpec) -> float:
    df = train_log(spec)
    col = next((c for c in df.columns if "accuracy_top1" in c), None)
    return float(df[col].max()) if col else float("nan")


# ----------------------------------------------------------------------------- tahmin

def split_images(split: str) -> pd.DataFrame:
    rows = [{"stem": p.stem, "path": str(p), "y_true": SLUG_TO_ID[p.parent.name]}
            for p in sorted((CLS_DATA / split).glob("*/*.jpg"))]
    return pd.DataFrame(rows)


def predict(weights: Path, split: str = "test", imgsz: int = 224, device=None, batch: int = 32,
            limit: int | None = None) -> pd.DataFrame:
    """Görüntülerin 7 sınıf olasılıklarını döndürür. Sınıf sırası modelin names sözlüğünden eşlenir."""
    from ultralytics import YOLO
    m = YOLO(str(weights))
    col_of = {int(k): SLUG_TO_ID[v] for k, v in m.names.items()}     # model indeksi → bizim sınıf kimliği
    df = split_images(split)
    if limit:   # duman testi: her sınıftan eşit sayıda görüntü
        df = df.groupby("y_true", group_keys=False).head(max(1, limit // len(C.CLASSES))).reset_index(drop=True)
    probs = np.zeros((len(df), len(C.CLASSES)), dtype=float)
    for k, r in enumerate(m.predict(source=df["path"].tolist(), imgsz=imgsz, device=device, batch=batch,
                                    stream=True, verbose=False)):
        p = r.probs.data.cpu().numpy()
        for mi, ci in col_of.items():
            probs[k, ci] = p[mi]
    df["y_pred"] = probs.argmax(1)
    for i in range(len(C.CLASSES)):
        df[f"p{i}"] = probs[:, i]
    df["p_true"] = probs[np.arange(len(df)), df["y_true"].to_numpy()]
    return df


def clean_test_stems() -> set[str]:
    p = C.DERIVED / "splits" / "test_clean.txt"
    return {Path(x).stem for x in p.read_text().split()} if p.exists() else set()


def evaluate_run(spec: RunSpec, device=None, limit: int | None = None) -> tuple[list[dict], list[pd.DataFrame]]:
    """Test tahminlerini üretir/kaydeder ve tüm test + temiz test için metrik satırları döndürür."""
    (OUT / "preds").mkdir(parents=True, exist_ok=True)
    pred_path = OUT / "preds" / f"{spec.name}_test.csv"
    df = predict(spec.dir / "weights" / "best.pt", "test", spec.imgsz, device, limit=limit)
    df.to_csv(pred_path, index=False)
    clean = clean_test_stems()
    rows, pcs = [], []
    for subset, d in (("tum_test", df), ("temiz_test", df[df["stem"].isin(clean)])):
        if not len(d):
            continue
        m = cls_metrics(d["y_true"].to_numpy(), d["y_pred"].to_numpy(), d[[f"p{i}" for i in range(7)]].to_numpy())
        rows.append({"kosu": spec.name, "model": spec.model, "boyut": spec.size, "seed": spec.seed, "imgsz": spec.imgsz,
                     "deney": spec.tag, "alt_kume": subset, "n": m["n"], "accuracy": m["accuracy"],
                     "macro_f1": m["macro_f1"], "weighted_f1": m["weighted_f1"],
                     "macro_roc_auc_ovr": m.get("macro_roc_auc_ovr"), "val_acc_en_iyi": best_val_acc(spec)})
        pc = m["per_class"].reset_index(names="sinif")
        pc.insert(0, "alt_kume", subset); pc.insert(0, "kosu", spec.name)
        pcs.append(pc)
    return rows, pcs


def evaluate_all(specs: list[RunSpec], device=None, limit: int | None = None) -> pd.DataFrame:
    rows, pcs = [], []
    for s in specs:
        if not (s.dir / "weights" / "best.pt").exists():
            print(f"– {s.name}: ağırlık yok, atlanıyor"); continue
        r, p = evaluate_run(s, device, limit)
        rows += r; pcs += p
        print(f"✓ {s.name}: değerlendirildi")
    res = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "results.csv", index=False)
    if pcs:
        pd.concat(pcs).to_csv(OUT / "per_class.csv", index=False)
    return res


def load_preds(spec: RunSpec) -> pd.DataFrame:
    return pd.read_csv(OUT / "preds" / f"{spec.name}_test.csv")


def select_by_val(specs: list[RunSpec]) -> RunSpec:
    """Doğrulama doğruluğu en yüksek koşu (seçim yalnız val ile yapılır; test bakılmaz)."""
    done = [s for s in specs if (s.dir / "weights" / "best.pt").exists()]
    return max(done, key=best_val_acc)


def summary_table(res: pd.DataFrame) -> pd.DataFrame:
    """Koşu sonuçlarını model × deney × alt küme için 'ortalama ± std (seed sayısı)' biçiminde özetler."""
    from .metrics import mean_std
    g = res.groupby(["deney", "model", "imgsz", "alt_kume"], sort=False)
    out = g.agg(seed=("seed", "count")).reset_index()
    for m in ("accuracy", "macro_f1", "weighted_f1", "macro_roc_auc_ovr"):
        out[m] = g[m].apply(lambda v: mean_std(v.to_numpy())).to_numpy()
    return out


def worst_predictions(preds: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """Yanlış sınıflanan ve gerçek sınıfa en düşük olasılığı veren k test görüntüsü."""
    w = preds[preds["y_true"] != preds["y_pred"]].sort_values("p_true").head(k).copy()
    w["gercek"] = [C.CLASSES[i] for i in w["y_true"]]
    w["tahmin"] = [C.CLASSES[i] for i in w["y_pred"]]
    w["tahmin_guveni"] = w[[f"p{i}" for i in range(len(C.CLASSES))]].max(1)
    w["gruplar_arasi"] = [C.GROUP_OF[a] != C.GROUP_OF[b] for a, b in zip(w["gercek"], w["tahmin"])]
    return w[["stem", "path", "gercek", "tahmin", "p_true", "tahmin_guveni", "gruplar_arasi"]].reset_index(drop=True)


# ----------------------------------------------------------------------------- verimlilik

def efficiency(spec: RunSpec, device=None, n_warmup: int = 20, n_images: int | None = None) -> dict:
    """Parametre sayısı ve batch=1 çıkarım süresi (ms/görüntü). Ultralytics'in ön/çıkarım/son işlem süreleri ayrı ayrı."""
    import torch
    from ultralytics import YOLO
    m = YOLO(str(spec.dir / "weights" / "best.pt"))
    n_params = int(sum(p.numel() for p in m.model.parameters()))
    paths = split_images("test")["path"].tolist()
    paths = paths[:n_images] if n_images else paths
    for p in paths[:n_warmup]:
        m.predict(p, imgsz=spec.imgsz, device=device, verbose=False)
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    pre, inf, post, wall = [], [], [], []
    for p in paths:
        sync(); t = time.perf_counter()
        r = m.predict(p, imgsz=spec.imgsz, device=device, verbose=False)[0]
        sync(); wall.append((time.perf_counter() - t) * 1000)
        pre.append(r.speed["preprocess"]); inf.append(r.speed["inference"]); post.append(r.speed["postprocess"])
    return {"kosu": spec.name, "model": spec.model, "imgsz": spec.imgsz, "parametre": n_params,
            "ms_on_islem": float(np.mean(pre)), "ms_cikarim": float(np.mean(inf)), "ms_son_islem": float(np.mean(post)),
            "ms_toplam_duvar": float(np.mean(wall)), "n_goruntu": len(paths), "cihaz": device_name(device)}


def device_name(device=None) -> str:
    """Kullanılan cihazın adı (device='cpu' verildiyse GPU olsa bile CPU yazar)."""
    import torch
    if str(device).lower() == "cpu":
        return "CPU"
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "Apple MPS"
    return "CPU"


def write_env() -> dict:
    import torch, ultralytics
    env = {"python": platform.python_version(), "torch": torch.__version__, "ultralytics": ultralytics.__version__,
           "cuda": torch.version.cuda, "cihaz": device_name(), "tarih": time.strftime("%Y-%m-%d %H:%M:%S"),
           "model_ailesi": MODEL_FAMILY, "egitim_ayarlari": TRAIN_ARGS}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "env.json").write_text(json.dumps(env, indent=2, ensure_ascii=False))
    return env
