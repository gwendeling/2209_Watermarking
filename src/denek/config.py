"""
Configuration file for Multi-Band Directional Watermarking System.
Modify these paths according to your file locations.
"""

# File paths - Modify these according to your setup
COVER_IMAGE_PATH = "cover_image.jpg"  # Path to your cover image
WATERMARK_IMAGE_PATH = "sample_watermark.png"  # Path to watermark image
WATERMARKED_IMAGE_PATH = "watermarked_image.jpg"  # Output path for watermarked image

# Watermarking parameters
BLOCK_SIZE = 8  # DCT block size
ALPHA = 0.1  # Embedding strength (0.05-0.2 recommended)

# Metadata for embedding
METADATA = {
    'user': 'User123',
    'model': 'Model456',
    'timestamp': '20241201'
}

# Expected bit lengths for extraction
EXPECTED_BIT_LENGTHS = {
    'user': 64,      # 8 characters * 8 bits
    'model': 64,     # 8 characters * 8 bits
    'timestamp': 64  # 8 characters * 8 bits
}

# Image processing parameters
IMAGE_SIZE = (512, 512)  # Resize images to this size if needed
WATERMARK_SIZE = (64, 64)  # Size of generated watermark

