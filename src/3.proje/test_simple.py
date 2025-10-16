import numpy as np
import cv2
from watermark_embedder import DCTInterBlockEmbedder
from watermark_extractor import DCTInterBlockExtractor

def test_simple_watermarking():
    """
    Test with a simple, smaller watermark for better results
    """
    print("Simple Watermarking Test")
    print("=" * 30)
    
    # Create a simple cover image
    cover_image = np.random.randint(100, 200, (256, 256, 3), dtype=np.uint8)
    
    # Create a simple watermark (32 bits)
    watermark_bits = np.array([1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0,
                              1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0])
    
    print(f"Cover image shape: {cover_image.shape}")
    print(f"Watermark length: {len(watermark_bits)}")
    print(f"Watermark: {watermark_bits}")
    
    # Initialize embedder and extractor
    embedder = DCTInterBlockEmbedder(threshold=1.0)  # Lower threshold
    extractor = DCTInterBlockExtractor(threshold=1.0)
    
    # Embed watermark
    print("\nEmbedding watermark...")
    watermarked = embedder.embed_watermark(cover_image, watermark_bits)
    
    # Calculate PSNR
    mse = np.mean((cover_image.astype(np.float64) - watermarked.astype(np.float64)) ** 2)
    psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float('inf')
    print(f"PSNR: {psnr:.2f} dB")
    
    # Extract watermark
    print("\nExtracting watermark...")
    extracted = extractor.extract_watermark(watermarked, len(watermark_bits))
    
    # Calculate similarity
    similarity = np.sum(watermark_bits == extracted) / len(watermark_bits)
    print(f"Extracted watermark: {extracted}")
    print(f"Similarity: {similarity:.4f}")
    
    # Save results
    cv2.imwrite("test_cover.png", cover_image)
    cv2.imwrite("test_watermarked.png", watermarked)
    
    print(f"\nTest completed!")
    print(f"Files saved: test_cover.png, test_watermarked.png")
    
    return similarity > 0.8

if __name__ == "__main__":
    success = test_simple_watermarking()
    if success:
        print("+ Test passed!")
    else:
        print("- Test failed!")
