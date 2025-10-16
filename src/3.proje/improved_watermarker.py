import numpy as np
import cv2
from scipy.fft import dct, idct
import pywt

class ImprovedDCTWatermarker:
    """
    Improved DCT-Domain Inter-Block Relationship Watermarking
    """
    
    def __init__(self, block_size=8, threshold=5.0, coefficient_position=(3, 4)):
        self.block_size = block_size
        self.threshold = threshold
        self.coefficient_position = coefficient_position
    
    def preprocess_image(self, image):
        """Convert to grayscale and apply DWT"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply DWT to get LL band
        coeffs = pywt.dwt2(gray, 'haar')
        ll_band = coeffs[0]
        
        return ll_band.astype(np.float32)
    
    def divide_into_blocks(self, image):
        """Divide image into blocks"""
        h, w = image.shape
        blocks = []
        
        for i in range(0, h - self.block_size + 1, self.block_size):
            for j in range(0, w - self.block_size + 1, self.block_size):
                block = image[i:i+self.block_size, j:j+self.block_size]
                blocks.append(block)
        
        return blocks
    
    def apply_dct_to_blocks(self, blocks):
        """Apply DCT to each block"""
        dct_blocks = []
        for block in blocks:
            dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
            dct_blocks.append(dct_block)
        return dct_blocks
    
    def embed_watermark_bit(self, dct_block1, dct_block2, bit):
        """Embed watermark bit using inter-block relationship"""
        row, col = self.coefficient_position
        
        coeff1 = dct_block1[row, col]
        coeff2 = dct_block2[row, col]
        
        mag1 = abs(coeff1)
        mag2 = abs(coeff2)
        
        new_dct_block1 = dct_block1.copy()
        new_dct_block2 = dct_block2.copy()
        
        if bit == 1:
            # Ensure |D_block1| > |D_block2| + T
            if mag1 <= mag2 + self.threshold:
                # Increase coeff1
                new_dct_block1[row, col] = coeff1 + self.threshold + 1
        else:
            # Ensure |D_block1| < |D_block2| - T
            if mag1 >= mag2 - self.threshold:
                # Decrease coeff1
                new_dct_block1[row, col] = coeff1 - self.threshold - 1
        
        return new_dct_block1, new_dct_block2
    
    def extract_watermark_bit(self, dct_block1, dct_block2):
        """Extract watermark bit from inter-block relationship"""
        row, col = self.coefficient_position
        
        coeff1 = dct_block1[row, col]
        coeff2 = dct_block2[row, col]
        
        mag1 = abs(coeff1)
        mag2 = abs(coeff2)
        
        # Extract bit based on relationship
        if mag1 > mag2 + self.threshold / 2:
            return 1
        else:
            return 0
    
    def reconstruct_image_from_dct(self, dct_blocks, original_shape):
        """Reconstruct image from DCT blocks"""
        h, w = original_shape
        reconstructed = np.zeros((h, w), dtype=np.float32)
        
        block_idx = 0
        for i in range(0, h - self.block_size + 1, self.block_size):
            for j in range(0, w - self.block_size + 1, self.block_size):
                idct_block = idct(idct(dct_blocks[block_idx].T, norm='ortho').T, norm='ortho')
                reconstructed[i:i+self.block_size, j:j+self.block_size] = idct_block
                block_idx += 1
        
        return reconstructed
    
    def reconstruct_full_image(self, ll_band, original_image):
        """Reconstruct full image from LL band"""
        ll_upsampled = cv2.resize(ll_band, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_CUBIC)
        
        if len(original_image.shape) == 3:
            result = np.zeros_like(original_image, dtype=np.uint8)
            for i in range(3):
                result[:, :, i] = np.clip(ll_upsampled, 0, 255).astype(np.uint8)
        else:
            result = np.clip(ll_upsampled, 0, 255).astype(np.uint8)
        
        return result
    
    def embed_watermark(self, cover_image, watermark_bits):
        """Embed watermark into cover image"""
        print("Embedding watermark...")
        
        ll_band = self.preprocess_image(cover_image)
        blocks = self.divide_into_blocks(ll_band)
        dct_blocks = self.apply_dct_to_blocks(blocks)
        
        num_pairs = len(dct_blocks) // 2
        watermark_length = min(len(watermark_bits), num_pairs)
        
        for i in range(watermark_length):
            block1_idx = i * 2
            block2_idx = i * 2 + 1
            
            if block2_idx < len(dct_blocks):
                dct_blocks[block1_idx], dct_blocks[block2_idx] = self.embed_watermark_bit(
                    dct_blocks[block1_idx], 
                    dct_blocks[block2_idx], 
                    watermark_bits[i]
                )
        
        watermarked_ll = self.reconstruct_image_from_dct(dct_blocks, ll_band.shape)
        watermarked_image = self.reconstruct_full_image(watermarked_ll, cover_image)
        
        return watermarked_image
    
    def extract_watermark(self, watermarked_image, watermark_length):
        """Extract watermark from watermarked image"""
        print("Extracting watermark...")
        
        ll_band = self.preprocess_image(watermarked_image)
        blocks = self.divide_into_blocks(ll_band)
        dct_blocks = self.apply_dct_to_blocks(blocks)
        
        num_pairs = len(dct_blocks) // 2
        actual_length = min(watermark_length, num_pairs)
        
        extracted_bits = []
        for i in range(actual_length):
            block1_idx = i * 2
            block2_idx = i * 2 + 1
            
            if block2_idx < len(dct_blocks):
                bit = self.extract_watermark_bit(
                    dct_blocks[block1_idx], 
                    dct_blocks[block2_idx]
                )
                extracted_bits.append(bit)
        
        return np.array(extracted_bits)

def test_improved_watermarking():
    """Test the improved watermarking system"""
    print("Improved Watermarking Test")
    print("=" * 30)
    
    # Create test image
    cover_image = np.random.randint(100, 200, (256, 256, 3), dtype=np.uint8)
    
    # Create simple watermark
    watermark_bits = np.array([1, 0, 1, 0, 1, 1, 0, 0, 1, 0])
    
    print(f"Cover image shape: {cover_image.shape}")
    print(f"Watermark: {watermark_bits}")
    
    # Initialize watermarker
    watermarker = ImprovedDCTWatermarker()
    
    # Embed watermark
    watermarked = watermarker.embed_watermark(cover_image, watermark_bits)
    
    # Calculate PSNR
    mse = np.mean((cover_image.astype(np.float64) - watermarked.astype(np.float64)) ** 2)
    psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float('inf')
    print(f"PSNR: {psnr:.2f} dB")
    
    # Extract watermark
    extracted = watermarker.extract_watermark(watermarked, len(watermark_bits))
    
    # Calculate similarity
    similarity = np.sum(watermark_bits == extracted) / len(watermark_bits)
    print(f"Extracted: {extracted}")
    print(f"Similarity: {similarity:.4f}")
    
    # Save results
    cv2.imwrite("improved_cover.png", cover_image)
    cv2.imwrite("improved_watermarked.png", watermarked)
    
    return similarity > 0.8

if __name__ == "__main__":
    success = test_improved_watermarking()
    if success:
        print("+ Test passed!")
    else:
        print("- Test failed!")

