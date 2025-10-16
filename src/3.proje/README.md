# DCT-Domain Inter-Block Relationship Watermarking

This project implements a novel digital image watermarking technique based on DCT-domain inter-block relationships. The method embeds watermark information by controlling the magnitude relationships between DCT coefficients of adjacent blocks, making it highly resistant to global attacks.

## Features

- **DCT-Domain Inter-Block Watermarking**: Embeds watermark by controlling relationships between DCT coefficients of adjacent blocks
- **Robust to Global Attacks**: Resistant to contrast enhancement, histogram equalization, and noise addition
- **OOP Design**: Well-structured object-oriented implementation
- **Quality Metrics**: PSNR and NCC calculation for watermark quality assessment
- **Automatic Watermark Generation**: Creates sample watermarks for testing

## Algorithm Overview

1. **Preprocessing**: DWT → LL band → 4x4 blocks → DCT
2. **Embedding**: For each pair of adjacent blocks, select the same mid-frequency DCT coefficient (e.g., (2,3))
3. **Relationship Control**: 
   - If bit = 1: Ensure |D_block1(2,3)| > |D_block2(2,3)| + T
   - If bit = 0: Ensure |D_block1(2,3)| < |D_block2(2,3)| - T
4. **Extraction**: Analyze the magnitude relationships to extract watermark bits

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the complete watermarking system:
```bash
python main.py
```

### File Structure

The system creates the following directories and files:
- `images/` - Cover images
- `watermarks/` - Watermark images
- `output/` - Watermarked images and extracted watermarks

### Customizing File Paths

Edit the file paths in `main.py`:
```python
self.cover_image_path = "path/to/your/cover_image.jpg"
self.watermark_image_path = "path/to/your/watermark.png"
```

### Individual Components

#### Generate Watermarks
```python
from watermark_generator import WatermarkGenerator

generator = WatermarkGenerator()
watermark = generator.generate_logo_watermark("YOUR_TEXT")
generator.save_watermark(watermark, "my_watermark.png")
```

#### Embed Watermark
```python
from watermark_embedder import DCTInterBlockEmbedder
import cv2

embedder = DCTInterBlockEmbedder()
cover_image = cv2.imread("cover.jpg")
watermark_bits = np.array([1, 0, 1, 0, ...])  # Your watermark
watermarked = embedder.embed_watermark(cover_image, watermark_bits)
```

#### Extract Watermark
```python
from watermark_extractor import DCTInterBlockExtractor

extractor = DCTInterBlockExtractor()
extracted = extractor.extract_watermark(watermarked_image, watermark_length)
```

## Parameters

- `block_size`: Size of DCT blocks (default: 4x4)
- `threshold`: Threshold for relationship control (default: 10.0)
- `coefficient_position`: DCT coefficient position (default: (2,3))

## Quality Metrics

- **PSNR**: Peak Signal-to-Noise Ratio (higher is better)
- **NCC**: Normalized Cross-Correlation (closer to 1 is better)
- **BER**: Bit Error Rate (lower is better)
- **Similarity**: Percentage of correctly extracted bits

## Advantages

1. **Robustness**: Highly resistant to global image processing attacks
2. **Invisibility**: Maintains good visual quality of watermarked images
3. **Novelty**: Uses inter-block relationships instead of intra-block modulation
4. **Flexibility**: Configurable parameters for different applications

## Example Output

```
DCT-Domain Inter-Block Relationship Watermarking System
============================================================

=== WATERMARK EMBEDDING ===
Cover image loaded: (512, 512, 3)
Watermark loaded from image: 256 bits
Starting watermark embedding...
LL band shape: (256, 256)
Number of blocks: 4096
DCT applied to all blocks
Embedding 128 watermark bits...
Watermark embedding completed!
Watermarked image saved: output/watermarked_image.png

Quality Metrics:
PSNR: 45.23 dB
NCC: 0.9876

=== WATERMARK EXTRACTION ===
Starting watermark extraction...
LL band shape: (256, 256)
Number of blocks: 4096
DCT applied to all blocks
Extracting 128 watermark bits...
Watermark extraction completed!
Extracted watermark saved: output/extracted_watermark.txt

Extraction Metrics:
Bit Error Rate (BER): 0.0156
Similarity: 0.9844
Extracted watermark length: 128

=== FINAL RESULTS ===
Original watermark: [1 0 1 0 1 1 0 0 1 0]...
Extracted watermark: [1 0 1 0 1 1 0 0 1 0]...

Final Extraction Quality:
Bit Error Rate: 0.0156
Similarity: 0.9844
✓ Watermark extraction successful!
```

## License

This project is for educational and research purposes.

