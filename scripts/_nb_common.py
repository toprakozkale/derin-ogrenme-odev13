"""Notebook üreticilerinin ortak hücre kodu."""

REPO = "toprakozkale/derin-ogrenme-odev13"

# Colab'da iki kip:
#  * Sahip kipi (Colab 🔑 Secrets'ta GITHUB_TOKEN var): Drive'ı bağlar; proje kodu Drive'da yoksa GitHub'dan klonlar,
#    varsa günceller; sonuçlar Drive'da kalıcıdır. Token yalnız istek başlığında kullanılır; .git/config'e/çıktıya yazılmaz.
#  * Misafir kipi (gizli anahtar yok — ör. hocanın hesabı, depo herkese açıkken): Drive'a dokunmaz, depoyu Colab'ın
#    geçici diskine (/content/odev13) klonlar; veri Kaggle'dan anahtarsız indirilir (veri seti herkese açık).
SETUP = f'''import os, sys, subprocess, base64
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive, userdata

    def _gizli(ad):                      # gizli anahtar yoksa (ya da erişim verilmediyse) None
        try:
            return userdata.get(ad)
        except Exception:
            return None

    # Gizli anahtarların hepsi EN BAŞTA okunur: Colab her yeni notebook için erişim izni penceresi açar; pencereler
    # birlikte çıksın ve uzun paket kurulumunun ardında gözden kaçıp zaman aşımına uğramasın (TimeoutException).
    print("Gizli anahtar erişimi istenirse açılan pencerelerde 'Erişim izni ver' deyin…")
    _gh = _gizli("GITHUB_TOKEN")
    if not os.environ.get("KAGGLE_KEY"):
        _ku, _kk = _gizli("KAGGLE_USERNAME"), _gizli("KAGGLE_KEY")
        if _ku and _kk:
            os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"] = _ku, _kk
    if _gh:                              # sahip kipi: kod + sonuçlar Drive'da (kalıcı)
        drive.mount("/content/drive")
        ROOT = Path(os.environ.setdefault("ODEV13_ROOT", "/content/drive/MyDrive/odev13"))
        auth = base64.b64encode(("x-access-token:" + _gh).encode()).decode()
        git = ["git", "-c", f"http.extraHeader=Authorization: Basic {{auth}}"]
    else:                                # misafir kipi: Drive'a dokunulmaz, her şey Colab'ın geçici diskinde
        ROOT = Path(os.environ.setdefault("ODEV13_ROOT", "/content/odev13"))
        git = ["git"]
        print("Misafir kipi: gizli anahtar yok → kod GitHub'dan (herkese açık depo) Colab'ın geçici diskine iner, "
              "veri Kaggle'dan anahtarsız indirilir. Sonuçlar oturum kapanınca silinir.")
    os.environ.setdefault("ODEV13_DATA", "/content/odev13_data")                         # veri (hızlı yerel disk)
    if not (ROOT / "src").exists():
        subprocess.run(git + ["clone", "-q", "https://github.com/{REPO}.git", str(ROOT)], check=True)
        print("Proje kodu GitHub'dan Drive'a klonlandı:", ROOT)
    else:
        pull = lambda: subprocess.run(git + ["-C", str(ROOT), "pull", "-q", "--ff-only"], capture_output=True, text=True,
                                      env={{**os.environ, "LC_ALL": "C"}})   # git mesajları İngilizce (aşağıda metin aranıyor)
        r = pull()
        if r.returncode != 0 and "untracked working tree files would be overwritten" in r.stderr:
            # Drive'da git'te olmayan bir dosya (ör. Colab'ın çizdiği bir şekil) GitHub'daki aynı adlı dosyanın yolunu
            # kesiyor (12 Kas'ta oldu). Silmek yerine yedek klasörüne taşınır, sonra yeniden denenir.
            import shutil, time as _t
            yedek = ROOT / "_git_yedek" / _t.strftime("%Y%m%d-%H%M%S")
            for line in r.stderr.splitlines():
                f = line.strip()
                if line[:1] in ("\\t", " ") and f and (ROOT / f).is_file():
                    (yedek / f).parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(ROOT / f), str(yedek / f))
            print("Çakışan dosyalar yedeğe taşındı:", yedek)
            r = pull()
        if r.returncode != 0:
            # Eski kodla devam etmek yanıltıcı hatalara yol açar (ör. "cannot import name 'seg'") → burada dur
            raise RuntimeError("git pull başarısız, proje kodu güncellenemedi:\\n" + r.stderr[-600:])
        print("Proje kodu güncellendi")
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


def setup_code(*pip_extra: str) -> str:
    """pip_extra: Colab'a ek kurulacak paket(ler) (ör. 'ultralytics', 'rfdetr[train]'); verilmezse 'pip' (zararsız)."""
    return SETUP.replace("PIP_EXTRA", ", ".join(repr(p) for p in (pip_extra or ("pip",))))
