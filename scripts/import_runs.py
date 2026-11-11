"""Colab'da üretilip Drive'dan indirilen eğitim sonuçlarını yerel runs/ klasörüne alır.

Drive'da `odev13/runs/cls` (ya da `seg`) klasörüne sağ tıklayıp **İndir** deyince bir zip iner. Sonra:

    .venv/bin/python scripts/import_runs.py ~/Downloads/cls-20261115T101010Z-001.zip
    .venv/bin/python scripts/import_runs.py ~/Downloads/cls            # zaten açılmış klasör de olur

Büyük klasörlerde Drive birden fazla zip verir; hepsini aynı komuta sırayla verebilirsiniz.
Var olan dosyaların üzerine yazar (Colab'daki sürüm esas alınır). Panel sayfayı yenileyince yeni sonuçları gösterir.
"""
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KNOWN = ("cls", "seg")


def find_run_root(base: Path) -> tuple[str, Path]:
    """Zip içinde 'cls' veya 'seg' klasörünü (ya da doğrudan onun içeriğini) bulur."""
    for kind in KNOWN:
        hits = sorted(base.rglob(kind), key=lambda p: len(p.parts))
        hits = [h for h in hits if h.is_dir() and ((h / "train").exists() or (h / "results.csv").exists())]
        if hits:
            return kind, hits[0]
    if (base / "train").exists() or (base / "results.csv").exists():
        kind = "seg" if any("seg" in p.name for p in (base / "train").glob("*")) else "cls"
        return kind, base
    raise SystemExit(f"İçinde 'cls' ya da 'seg' sonuç klasörü bulunamadı: {base}")


def import_one(src: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        if src.suffix == ".zip":
            with zipfile.ZipFile(src) as z:
                z.extractall(tmp)
            base = Path(tmp)
        else:
            base = src
        kind, run_root = find_run_root(base)
        dest = ROOT / "runs" / kind
        n = 0
        for f in run_root.rglob("*"):
            if f.is_file():
                t = dest / f.relative_to(run_root)
                t.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, t)
                n += 1
        print(f"✓ {src.name}: {n} dosya → runs/{kind}/")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    for a in sys.argv[1:]:
        import_one(Path(a).expanduser())
    done = sorted(p.parent.name for p in (ROOT / "runs").glob("*/train/*/DONE"))
    print(f"Biten koşular ({len(done)}):", ", ".join(done) or "-")
