# Veri ilk inceleme bulguları (7 Kasım 2026)

Kaynak: Kaggle `usmanafzaal/strawberry-disease-detection-dataset`, sürüm 1, zip 559,7 MB,
SHA-256 `e728dd56eaab4e11e64d46da77b06eb377beb264c53d0a32c1eaf44d051de525`.
Bu dosyadaki tüm sayılar yerelde çalıştırılan keşif betiklerinin çıktısıdır; 01 notebook'u aynı kontrolleri
tekrarlanabilir biçimde yeniden üretecektir.

## Biçim
- Etiketler **LabelMe JSON** (v4.5.6): `shapes[]` → `label`, `points` (piksel koordinatı), `shape_type` (hepsi `polygon`).
  JSON içinde görüntünün base64 kopyası (`imageData`) da var.
- **Tüm görüntüler 419×419** (train/val/test). JSON'daki boyutlar ve `imagePath` dosyalarla birebir uyumlu; EXIF döndürme yok.

## Bölme ve sayılar
| Bölme | Görüntü | Örnek (poligon) | Görüntü başına örnek (ort / maks) |
|---|---|---|---|
| train | 1.450 | 3.540 | 2,44 / 15 |
| val | 307 | 751 | 2,45 / 12 |
| test | 743 | 1.416 | 1,91 / 17 |

Örnek sayısı / sınıf:
| Sınıf | train | val | test |
|---|---|---|---|
| Leaf Spot | 1.365 | 257 | 478 |
| Powdery Mildew Leaf | 1.110 | 255 | 343 |
| Gray Mold | 332 | 108 | 158 |
| Angular Leafspot | 294 | 52 | 193 |
| Blossom Blight | 186 | 44 | 70 |
| Powdery Mildew Fruit | 164 | 18 | 116 |
| Anthracnose Fruit Rot | 89 | 17 | 58 |

Görüntü düzeyinde (en az bir örneği olan) dağılım ciddi dengesiz: train'de Leaf Spot 382+ görüntü,
Anthracnose Fruit Rot 52 görüntü (~7×).

## "Test Disease Severity Level" klasörü
743 görüntünün **743'ü test/ ile aynı ad ve birebir aynı içerikte**, JSON'lar da birebir aynı.
→ Ayrı veri değil; test'in şiddet düzeyine göre (Level 1: 206, Level 2: 537) gruplanmış kopyası. Ek veri olarak kullanılmaz.

## Çok sınıflı / dosya adıyla uyuşmayan görüntüler (10 adet)
- train: `leaf_spot535, 551, 593, 636, 654, 671` → Leaf Spot + **Powdery Mildew Fruit**
- test: `leaf_spot508, 559, 679` → Leaf Spot + **Anthracnose Fruit Rot**
- train: `gray_mold1` → tek etiket **Blossom Blight** (dosya adı gray_mold) — olası etiket/ad hatası, gözle kontrol edilecek.
- Not: Yaprak + meyve hastalığının **aynı karede** bulunması, "Dikkat" maddesindeki meyve/yaprak karışma tartışması için doğrudan örnek.

## Poligon sorunları
- Kendini kesen (geçersiz) poligon: train 8, test 4 (ör. `angular_leafspot283`, `gray_mold270`, `leaf_spot318`).
- Görüntü dışına taşan koordinat: train 2 (`angular_leafspot65`, `powdery_mildew_leaf329`).
- Boş JSON, 3'ten az noktalı poligon, görüntüsüz etiket: **yok**.

## Yinelenen görüntüler ve sızıntı (önemli)
- Bayt düzeyinde aynı dosya: 41 çift — **train içinde 13, test içinde 28** (ör. `train/leaf_spot485 = leaf_spot600`, `test/leaf_spot487 = leaf_spot601`); bölmeler arası yok. *(11 Kas düzeltmesi: ilk sürümde "hepsi train içinde" yazıyordu.)* Test içindeki 28 kopya, test metriklerinde bu görüntülerin iki kez sayılması demektir (sızıntı değil, ama raporda not edilecek).
- Algısal hash (pHash, 64 bit) ile **bölmeler arası** benzer çiftler: gözle kontrol edilen örneklerde mesafe 0–12 aralığında
  çiftlerin hepsi aynı sahne (aynı yaprak/çiçek, hafif kayma veya kırpma — ardışık kare gibi).

| pHash mesafe ≤ | train/val'de benzeri olan **test** görüntüsü | train'de benzeri olan **val** görüntüsü |
|---|---|---|
| 0 | 29 / 743 (%3,9) | 18 / 307 (%5,9) |
| 4 | 37 (%5,0) | 44 (%14,3) |
| 8 | 47 (%6,3) | 83 (%27,0) |
| 12 | 51 (%6,9) | 108 (%35,2) |

Eşik henüz kalibre edilmedi (12'nin üstünde de aynı sahne olabilir); 01 notebook'unda görsel kontrolle eşik seçilecek.
Yorum: test sızıntısı küçük (~%4–7), **val sızıntısı büyük** → val skorları iyimser olur; erken durdurma/model seçimi etkilenebilir.

## Güncelleme (7 Kasım 2026, akşam)
- pHash eşiği görsel kalibrasyonla **12** seçildi (`docs/phash_calibration.csv`): test 51/743 (%6,9) sızan → **temiz test 692**;
  val 108/307 (%35,2). 14–16 aralığında da gerçek yinelenenler kaldığı için bu bir alt sınır.
- `train/gray_mold1`: tek poligon fotoğrafın ortasındaki gri tüylü meyveyi kapsıyor ama "Blossom Blight" etiketli —
  görsel olarak kurşuni küfe benziyor, **muhtemel etiket hatası**. Uzman etiketi değiştirilmedi; raporda etiket gürültüsü örneği.
- Kesin ve güncel sayılar: `data/derived/summary.json` (panel ve 01 notebook'u bunu okur).

## Roboflow çapraz kontrolü (11 Kasım 2026)
Ham veri Roboflow'a yüklendi, ön işlemesiz sürüm COCO olarak indirildi ve bizim dönüşümümüzle örnek örnek karşılaştırıldı:
2.459 eşleşen görüntünün hepsinde boyut, örnek sayısı ve sınıflar aynı; 5.571 örneğin ortalama IoU'su 0,99995, yalnız 1 örnek
0,99'un altında (`test/gray_mold539`, bizim poligon düzeltmemiz). Roboflow'un atladığı 41 görüntünün hepsi bölme içi bayt bayt
kopyalar (train 13, test 28). Ayrıntı: `docs/roboflow/`.
