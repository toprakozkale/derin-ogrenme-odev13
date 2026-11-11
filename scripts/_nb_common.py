"""Notebook üreticilerinin ortak hücre kodu."""

REPO = "toprakozkale/derin-ogrenme-odev13"

# Colab'da: Drive'ı bağlar; proje kodu Drive'da yoksa GitHub'dan klonlar, varsa günceller.
# Depo private olduğu için Colab 🔑 Secrets'ta GITHUB_TOKEN (yalnız bu depoya salt-okuma izinli) gerekir.
# Token yalnız istek başlığında kullanılır; .git/config'e ya da çıktıya yazılmaz.
SETUP = f'''import os, sys, subprocess, base64
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive, userdata
    drive.mount("/content/drive")
    ROOT = Path(os.environ.setdefault("ODEV13_ROOT", "/content/drive/MyDrive/odev13"))   # kod + sonuçlar (kalıcı)
    os.environ.setdefault("ODEV13_DATA", "/content/odev13_data")                         # veri (hızlı yerel disk)
    # Gizli anahtarların hepsi EN BAŞTA okunur: Colab her yeni notebook için erişim izni penceresi açar; pencereler
    # birlikte çıksın ve uzun paket kurulumunun ardında gözden kaçıp zaman aşımına uğramasın (TimeoutException).
    print("Gizli anahtar erişimi istenirse açılan pencerelerde 'Erişim izni ver' deyin…")
    _gh = userdata.get("GITHUB_TOKEN")
    if not os.environ.get("KAGGLE_KEY"):
        os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME"); os.environ["KAGGLE_KEY"] = userdata.get("KAGGLE_KEY")
    auth = base64.b64encode(("x-access-token:" + _gh).encode()).decode()
    git = ["git", "-c", f"http.extraHeader=Authorization: Basic {{auth}}"]
    if not (ROOT / "src").exists():
        subprocess.run(git + ["clone", "-q", "https://github.com/{REPO}.git", str(ROOT)], check=True)
        print("Proje kodu GitHub'dan Drive'a klonlandı:", ROOT)
    else:
        r = subprocess.run(git + ["-C", str(ROOT), "pull", "-q", "--ff-only"], capture_output=True, text=True)
        print("Proje kodu güncellendi" if r.returncode == 0 else "UYARI: git pull başarısız: " + r.stderr[-300:])
    sürüm = subprocess.run(["git", "-C", str(ROOT), "log", "-1", "--format=%h · %ad · %s", "--date=format:%d.%m %H:%M"],
                           capture_output=True, text=True).stdout.strip()
    print("Kod sürümü:", sürüm)
    # Oturum açık kaldıysa Python eski src modüllerini hafızada tutar → yeni indirilen kodun kullanılması için temizle
    for m in [m for m in sys.modules if m == "src" or m.startswith("src.")]:
        del sys.modules[m]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", PIP_EXTRA, "imagehash", "kagglehub", "pycocotools", "shapely", "statsmodels"], check=True)
else:
    ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
    os.environ.setdefault("ODEV13_ROOT", str(ROOT))
sys.path.insert(0, str(ROOT))
assert (ROOT / "src" / "config.py").exists(), f"Proje kodu bulunamadı: {{ROOT}}"
'''


def setup_code(pip_extra: str = "pip") -> str:
    """pip_extra: Colab'a ek kurulacak paket (ör. 'ultralytics'); gerekmezse 'pip' (zararsız)."""
    return SETUP.replace("PIP_EXTRA", repr(pip_extra))
