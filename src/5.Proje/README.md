# DWT+DCT Görüntü Filigrulama (Watermarking) Projesi

Bu proje, gri seviye bir kapak (cover) görüntüsüne ikili (binary) bir filigranı gömme ve sonrasında bu filigranı geri çıkarma işlemlerini DWT (Discrete Wavelet Transform) ve DCT (Discrete Cosine Transform) kombinasyonu ile gerçekleştirir. Kod, `haar` dalgacığı ile 2 seviye ayrışım kullanır ve filigranı hem `LL2` hem de `LH1` bantlarına gömerek dayanıklılığı artırır.

## İçindekiler
- Proje Yapısı
- Kurulum
- Çalıştırma
- Algoritma (Özet ve Adımlar)
- Yapılandırma
- Girdi/Çıktı
- Metrikler (PSNR, NCC)

## Proje Yapısı
```
5.Proje/
  data/
    cover_grayscale.png
    watermark.png
  output/
    watermarked.png
    extracted_watermark.png
  requirements.txt
  README.md
  src/
    config.py
    main.py
    watermark/
      embedder.py
      extractor.py
      generator.py
      utils.py
```

- `src/main.py`: Uygulama giriş noktası; filigran üretir/okur, gömer ve çıkarır.
- `src/config.py`: Yol ve klasör ayarları (`data`, `output` vb.).
- `src/watermark/embedder.py`: DWT+DCT ile gömme mantığı (`DwtDctEmbedder`).
- `src/watermark/extractor.py`: Gömülü filigranı çıkarma mantığı (`DwtDctExtractor`).
- `src/watermark/generator.py`: Basit bir ikili filigran (logo/maske) üretici.
- `src/watermark/utils.py`: DWT/DCT yardımcıları, blok işlemleri, PSNR/NCC hesapları.

## Kurulum
Windows PowerShell için örnek adımlar:

```powershell
# Proje klasörüne gidin (requirements.txt burada)
cd "C:\Users\mikas\OneDrive\Documents\2209_Watermarking\2209_Watermarking\src\5.Proje"

# (Önerilir) Sanal ortam oluşturun ve aktifleştirin
py -m venv .venv
.\.venv\Scripts\Activate.ps1

# Bağımlılıkları kurun
py -m pip install -r .\requirements.txt
```

> Not: `py` komutu yoksa `python` kullanabilirsiniz.

## Çalıştırma
```powershell
# Uygulamayı çalıştırın
py .\src\main.py
```

İlk çalıştırmada `data/cover_grayscale.png` yoksa basit bir gri seviye gradyan kapak görüntüsü otomatik oluşturulur. `data/watermark.png` yoksa `WM` yazılı 64x64 bir ikili filigran üretilir.

Çıktılar `output/` klasörüne yazılır:
- `watermarked.png`: Filigran gömülmüş görüntü
- `extracted_watermark.png`: Çıkarılan filigran (ikili, 0/255)

Terminalde PSNR ve NCC değerleri yazdırılır.

## Algoritma (Özet ve Adımlar)
Bu bölüm, gömme ve çıkarma adımlarını özetler. Detaylı uygulama için `embedder.py` ve `extractor.py` dosyalarına bakınız.

### Gömme (Embedding) Adımları
1. Girdi olarak gri seviye kapak görüntüsü `cover` ve ikili filigran `watermark_bits` alınır.
2. `cover` üzerinde 2 seviyeli DWT uygulanır (`haar`):
   - 1. seviye bantları: `LL1`, `LH1`, `HL1`, `HH1`
   - 2. seviye bantları (LL1'in tekrar ayrışımı): `LL2`, `LH2`, `HL2`, `HH2`
3. Filigran kapasitesi, blok tabanlı yerleştirme ile belirlenir. Blok boyutu `block_size = 8`'dir:
   - `LL2` ve `LH1` bantları 8x8 bloklara ayrılır.
   - Kapasite, her banttaki blok sayısına göre hesaplanır ve filigran, kare olacak şekilde yeniden boyutlandırılır.
4. Her blok için 2B DCT uygulanır. İki katsayı çifti seçilir ve karşılaştırmalı gömme yapılır:
   - `LL2` bandında `(1,3)` ve `(3,1)` katsayıları
   - `LH1` bandında `(2,4)` ve `(4,2)` katsayıları
   - Bit=1 için ilk katsayı ikinciden büyük, bit=0 için tersi olacak şekilde `alpha` (gömme kuvveti) ile ayarlama yapılır.
5. DCT terslenir, bloklar birleştirilir ve DWT seviyeleri birleştirilerek filigranlı görüntü elde edilir.
6. Meta bilgiler (`block_size`, `wavelet`, `alpha`, `wm_shape` vb.) daha sonra çıkarma için saklanır.

### Çıkarma (Extraction) Adımları
1. Filigranlı görüntü üzerinde aynı 2 seviyeli DWT uygulanır.
2. `LL2` ve `LH1` bantları 8x8 bloklara ayrılır ve her blok için DCT alınır.
3. Gömmede kullanılan katsayı çiftlerinden bit tahmini yapılır:
   - `LL2`: `(1,3)` vs `(3,1)`
   - `LH1`: `(2,4)` vs `(4,2)`
4. İki banttan gelen bit dizileri üzerinde çoğunluk oylaması (majority voting) yapılır (eşitlikte `LL2` tercih edilir).
5. Bitler `wm_shape` boyutlarına göre yeniden şekillendirilip ikili (0/255) filigran görüntüsü üretilir.

### Parametreler ve Dayanıklılık
- `block_size = 8`: DCT için standart blok boyutu.
- `wavelet = "haar"`: Hızlı ve yaygın bir seçim.
- `alpha`: Gömme kuvveti (varsayılan `5.0`, örnek uygulamada `main.py` içinde `7.5` olarak kullanılmıştır). Daha yüksek değer görünür artefakt riskini artırabilir ama dayanıklılığı artırabilir.
- Çift bant (LL2 ve LH1) kullanımı, sıkıştırma/gürültü gibi bozucu etkilere karşı dayanıklılığı artırmayı amaçlar.

## Yapılandırma
- `src/config.py` dosyasında yollar ve klasörler tanımlıdır:
  - `DATA_DIR`, `OUTPUT_DIR`
  - `COVER_IMAGE_PATH`, `WATERMARK_IMAGE_PATH`
- Kendi kapak/filigran dosyalarınızı `data/` altına yerleştirip bu yolları güncelleyebilirsiniz.

## Girdi/Çıktı
- Girdi:
  - `data/cover_grayscale.png`: Gri seviye kapak görüntüsü (otomatik üretilebilir)
  - `data/watermark.png`: İkili filigran (otomatik üretilebilir)
- Çıktı:
  - `output/watermarked.png`
  - `output/extracted_watermark.png`

## Metrikler (PSNR, NCC)
- PSNR: Kapak ve filigranlı görüntü arasındaki bozulmayı ölçer (dB cinsinden). `utils.psnr` kullanılır.
- NCC: Orijinal ve çıkarılan filigran arasındaki benzerliği ölçer. `utils.ncc` kullanılır.

## Hızlı Sorun Giderme
- `ModuleNotFoundError`: Komutu `5.Proje` kök klasöründe çalıştırdığınızdan emin olun ve `requirements.txt` kurulu olsun.
- Çıktılar oluşmuyor: `output/` klasörünün yazılabilir olduğundan emin olun (kod, gerekirse otomatik oluşturur).
- Filigran çok zayıf/kuvvetli: `main.py` içinde `DwtDctEmbedder(alpha=...)` değerini ayarlayın.
