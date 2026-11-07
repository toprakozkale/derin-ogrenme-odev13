# Ortam bilgisi (tekrarlanabilirlik)

Bu dosya her aşamada gerçek çalıştırma çıktılarıyla doldurulur; değerler tahminle yazılmaz.

## Yerel (kod geliştirme ve CPU testleri)
- Makine: Apple M2, 8 GB RAM, macOS 26.6.2
- Python: 3.13.18 (Homebrew `python@3.13`), sanal ortam `.venv/` — Colab 3.13.16 ile aynı ana sürüm
- Jupyter çekirdeği: "Python 3.13 (odev13)"
- PyTorch / Ultralytics / RF-DETR: yerelde henüz kurulu değil (duman testi aşamasında eklenecek)
- Paket sürümleri: `requirements-local.txt` (pip freeze)

## Colab (eğitim)
- GPU: NVIDIA Tesla T4, 15.360 MiB (15 GB) bellek — `!nvidia-smi`, 7 Kasım 2026
- NVIDIA sürücüsü: 580.82.07 · CUDA (sürücü): 13.0
- Python: 3.13.16 (Colab, Eylül 2026 imajı; Ubuntu 24.04)
- PyTorch: _(notebook ilk hücresinden)_
- ultralytics / rfdetr / pycocotools sürümleri: _(pip freeze)_

## Seed'ler
- Bölme: 42 (resmi bölme yoksa)
- Eğitim: 0, 1, 2
