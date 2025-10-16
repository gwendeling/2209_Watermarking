import numpy as np
import cv2
from scipy.fft import dct, idct
import pywt

class DCTInterBlockEmbedder:
    """
    DCT-Domain Inter-Block Relationship Watermarking
    Embeds watermark by controlling magnitude relationships between DCT coefficients of adjacent blocks
    """
    
    def __init__(self, block_size=4, threshold=2.0, coefficient_position=(2, 3)):
        self.block_size = block_size
        self.threshold = threshold
        self.coefficient_position = coefficient_position  # (row, col) position in DCT block
        
    def preprocess_image(self, image):
        """
        Preprocess image: DWT -> LL band -> convert to grayscale if needed
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            np.ndarray: Preprocessed image (LL band)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply DWT to get LL band
        coeffs = pywt.dwt2(gray, 'haar')
        ll_band = coeffs[0]
        
        return ll_band.astype(np.float32)
    
    def divide_into_blocks(self, image):
        """
        Divide image into 4x4 blocks
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            list: List of 4x4 blocks
        """
        h, w = image.shape
        blocks = []
        
        for i in range(0, h - self.block_size + 1, self.block_size):
            for j in range(0, w - self.block_size + 1, self.block_size):
                block = image[i:i+self.block_size, j:j+self.block_size]
                blocks.append(block)
        
        return blocks
    
    def apply_dct_to_blocks(self, blocks):
        """
        Apply DCT to each block
        
        Args:
            blocks (list): List of image blocks
            
        Returns:
            list: List of DCT coefficients for each block
        """
        dct_blocks = []
        for block in blocks:
            dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
            dct_blocks.append(dct_block)
        
        return dct_blocks
    
    def embed_watermark_bit(self, dct_block1, dct_block2, bit):
        """
        Embed a single watermark bit using inter-block relationship
        
        Args:
            dct_block1 (np.ndarray): First DCT block
            dct_block2 (np.ndarray): Second DCT block
            bit (int): Watermark bit (0 or 1)
            
        Returns:
            tuple: Modified DCT blocks
        """
        row, col = self.coefficient_position
        
        # Get the coefficients at the specified position
        coeff1 = dct_block1[row, col]
        coeff2 = dct_block2[row, col]
        
        # Get magnitudes
        mag1 = abs(coeff1)
        mag2 = abs(coeff2)
        
        # Create copies to modify
        new_dct_block1 = dct_block1.copy()
        new_dct_block2 = dct_block2.copy()
        
        if bit == 1:
            # Ensure |D_block1(2,3)| > |D_block2(2,3)| + T
            if mag1 <= mag2 + self.threshold:
                # Adjust coefficients to satisfy the condition
                if mag1 > 0:
                    # Scale up coeff1 slightly
                    scale_factor = (mag2 + self.threshold + 0.5) / mag1
                    new_dct_block1[row, col] = coeff1 * scale_factor
                else:
                    # Set coeff1 to a small positive value
                    new_dct_block1[row, col] = mag2 + self.threshold + 0.5
        else:
            # Ensure |D_block1(2,3)| < |D_block2(2,3)| - T
            if mag1 >= mag2 - self.threshold:
                # Adjust coefficients to satisfy the condition
                if mag2 > self.threshold:
                    # Scale down coeff1 slightly
                    scale_factor = max(0.1, (mag2 - self.threshold - 0.5) / max(mag1, 0.1))
                    new_dct_block1[row, col] = coeff1 * scale_factor
                else:
                    # Set coeff1 to a very small value
                    new_dct_block1[row, col] = 0.01
        
        return new_dct_block1, new_dct_block2
    
    def reconstruct_image_from_dct(self, dct_blocks, original_shape):
        """
        Reconstruct image from DCT blocks
        
        Args:
            dct_blocks (list): List of DCT blocks
            original_shape (tuple): Original image shape
            
        Returns:
            np.ndarray: Reconstructed image
        """
        h, w = original_shape
        reconstructed = np.zeros((h, w), dtype=np.float32)
        
        block_idx = 0
        for i in range(0, h - self.block_size + 1, self.block_size):
            for j in range(0, w - self.block_size + 1, self.block_size):
                # Apply inverse DCT
                idct_block = idct(idct(dct_blocks[block_idx].T, norm='ortho').T, norm='ortho')
                reconstructed[i:i+self.block_size, j:j+self.block_size] = idct_block
                block_idx += 1
        
        return reconstructed
    
    def reconstruct_full_image(self, ll_band, original_image):
        """
        Reconstruct full image from LL band
        
        Args:
            ll_band (np.ndarray): LL band of DWT
            original_image (np.ndarray): Original image for reference
            
        Returns:
            np.ndarray: Reconstructed full image
        """
        # Upsample LL band to match original image size
        ll_upsampled = cv2.resize(ll_band, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_CUBIC)
        
        if len(original_image.shape) == 3:
            # Color image
            result = np.zeros_like(original_image, dtype=np.uint8)
            for i in range(3):
                result[:, :, i] = np.clip(ll_upsampled, 0, 255).astype(np.uint8)
        else:
            # Grayscale image
            result = np.clip(ll_upsampled, 0, 255).astype(np.uint8)
        
        return result
    
    def embed_watermark(self, cover_image, watermark_bits):
        """
        Embed watermark into cover image
        
        Args:
            cover_image (np.ndarray): Cover image
            watermark_bits (np.ndarray): Binary watermark
            
        Returns:
            np.ndarray: Watermarked image
        """
        print("Starting watermark embedding...")
        
        # Preprocess image
        ll_band = self.preprocess_image(cover_image)
        print(f"LL band shape: {ll_band.shape}")
        
        # Divide into blocks
        blocks = self.divide_into_blocks(ll_band)
        print(f"Number of blocks: {len(blocks)}")
        
        # Apply DCT to blocks
        dct_blocks = self.apply_dct_to_blocks(blocks)
        print("DCT applied to all blocks")
        
        # Embed watermark bits
        num_pairs = len(dct_blocks) // 2
        watermark_length = min(len(watermark_bits), num_pairs)
        
        print(f"Embedding {watermark_length} watermark bits...")
        
        for i in range(watermark_length):
            block1_idx = i * 2
            block2_idx = i * 2 + 1
            
            if block2_idx < len(dct_blocks):
                dct_blocks[block1_idx], dct_blocks[block2_idx] = self.embed_watermark_bit(
                    dct_blocks[block1_idx], 
                    dct_blocks[block2_idx], 
                    watermark_bits[i]
                )
        
        # Reconstruct image
        watermarked_ll = self.reconstruct_image_from_dct(dct_blocks, ll_band.shape)
        watermarked_image = self.reconstruct_full_image(watermarked_ll, cover_image)
        
        print("Watermark embedding completed!")
        return watermarked_image

def main():
    """
    Test the embedder with a sample image
    """
    # Create a sample cover image
    cover_image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    
    # Create sample watermark
    watermark_bits = np.random.randint(0, 2, 100)
    
    # Initialize embedder
    embedder = DCTInterBlockEmbedder()
    
    # Embed watermark
    watermarked = embedder.embed_watermark(cover_image, watermark_bits)
    
    # Save results
    cv2.imwrite("sample_cover.png", cover_image)
    cv2.imwrite("sample_watermarked.png", watermarked)
    
    print("Sample embedding completed!")

if __name__ == "__main__":
    main()
