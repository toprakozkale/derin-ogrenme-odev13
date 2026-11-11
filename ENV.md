# Ortam bilgisi (tekrarlanabilirlik)

Bu dosya her aşamada gerçek çalıştırma çıktılarıyla doldurulur; değerler tahminle yazılmaz.

## Yerel (kod geliştirme ve CPU testleri)
- Makine: Apple M2, 8 GB RAM, macOS 26.6.2
- Python: 3.13.18 (Homebrew `python@3.13`), sanal ortam `.venv/` — Colab 3.13.16 ile aynı ana sürüm
- Jupyter çekirdeği: "Python 3.13 (odev13)"
- PyTorch 2.15.0 · torchvision 0.29.2 · Ultralytics 8.4.210 (yerel duman testi; Apple M2 CPU/MPS) — RF-DETR henüz kurulu değil
- Paket sürümleri: `requirements-local.txt` (pip freeze)

## Colab (eğitim)
- GPU: NVIDIA Tesla T4, 15.360 MiB (15 GB) bellek — `!nvidia-smi`, 7 Kasım 2026
- NVIDIA sürücüsü: 580.82.07 · CUDA (sürücü): 13.0
- Python: 3.13.16 (Colab, Eylül 2026 imajı; Ubuntu 24.04)
- PyTorch 2.14.1+cu130 · Ultralytics 8.4.210 · Python 3.13.16 (02 notebook, 11 Kas 2026; `runs/cls/env.json`)
- ultralytics / rfdetr / pycocotools sürümleri: _(pip freeze)_

## Seed'ler
- Bölme: 42 (resmi bölme yoksa)
- Eğitim: 0, 1, 2
