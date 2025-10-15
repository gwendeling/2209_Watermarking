# Multi-Band Directional Watermarking System

A novel digital image watermarking system that exploits the directionality of DWT sub-bands in conjunction with DCT for robust watermark embedding and extraction.

## Features

- **Multi-Band Directional Embedding**: Uses different DWT sub-bands (LH, HL, HH) for different metadata types
- **Semantic Embedding**: Maps metadata structure to image's multi-directional frequency content
- **Quantization Index Modulation (QIM)**: Robust embedding technique
- **Adaptive Step Size**: Based on local variance of sub-bands
- **Quality Metrics**: PSNR, NCC, and SSIM calculations
- **OOP Design**: Well-structured, modular codebase

## System Architecture

### Sub-band Assignments
- **LH (Horizontal Details)**: User metadata
- **HL (Vertical Details)**: Model metadata  
- **HH (Diagonal Details)**: Timestamp metadata

### Embedding Process
1. Perform 1-level DWT to get LL, LH, HL, HH sub-bands
2. Divide each sub-band into blocks
3. Apply DCT to each block
4. Embed bits using QIM on selected DCT coefficient (3,3)
5. Reconstruct image using inverse DWT

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start
```bash
python main.py
```

### Configuration
Modify `config.py` to set your file paths and parameters:

```python
# File paths
COVER_IMAGE_PATH = "your_cover_image.jpg"
WATERMARKED_IMAGE_PATH = "watermarked_image.jpg"

# Parameters
BLOCK_SIZE = 8
ALPHA = 0.1  # Embedding strength

# Metadata
METADATA = {
    'user': 'YourUser',
    'model': 'YourModel',
    'timestamp': '20241201'
}
```

### Individual Module Usage

#### Generate Watermark
```python
from watermark_generator import WatermarkGenerator

generator = WatermarkGenerator()
watermark = generator.generate_sample_watermark()
generator.save_watermark(watermark, "watermark.png")
```

#### Embed Watermark
```python
from watermark_embedder import WatermarkEmbedder

embedder = WatermarkEmbedder()
watermarked = embedder.embed_from_files(
    "cover.jpg", "", "watermarked.jpg"
)
```

#### Extract Watermark
```python
from watermark_extractor import WatermarkExtractor

extractor = WatermarkExtractor()
metadata = extractor.extract_from_file("watermarked.jpg")
```

## File Structure

```
├── main.py                 # Main application
├── config.py              # Configuration settings
├── watermark_generator.py # Watermark generation
├── watermark_embedder.py  # Watermark embedding
├── watermark_extractor.py # Watermark extraction
├── image_metrics.py       # Quality metrics calculation
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## Quality Metrics

The system calculates and displays:
- **PSNR (Peak Signal-to-Noise Ratio)**: Image quality measure
- **NCC (Normalized Cross-Correlation)**: Similarity measure
- **SSIM (Structural Similarity Index)**: Perceptual quality measure

## Robustness Features

- **Multi-directional embedding**: Requires distortion of all sub-bands to destroy watermark
- **QIM embedding**: Inherently robust to common attacks
- **Adaptive quantization**: Adjusts to local image characteristics
- **Semantic structure**: Metadata organization matches frequency content

## Example Output

```
============================================================
Multi-Band Directional Watermarking System
============================================================

Image Quality Metrics:
------------------------------
PSNR: 45.23
NCC: 0.9876
SSIM: 0.9954

Metadata Comparison:
------------------------------
Type         Original        Extracted       Similarity
------------------------------------------------------------
user         User123         User123         1.000
model        Model456        Model456        1.000
timestamp    20241201        20241201        1.000

Overall Assessment:
------------------------------
PSNR: 45.23 dB (Good)
NCC: 0.9876 (Good)
Average Metadata Similarity: 1.000 (Good)

✓ Watermarking process completed successfully!
```

## Technical Details

### DWT Sub-bands
- **LL**: Low-frequency approximation (not used for embedding)
- **LH**: Horizontal details (high frequency in horizontal direction)
- **HL**: Vertical details (high frequency in vertical direction)
- **HH**: Diagonal details (high frequency in both directions)

### QIM Implementation
- Uses mid-band DCT coefficient (3,3) for embedding
- Even multiples of step size represent bit '0'
- Odd multiples of step size represent bit '1'
- Step size adapts based on local block variance

## License

This project is for educational and research purposes.

