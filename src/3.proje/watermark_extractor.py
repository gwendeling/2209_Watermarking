import numpy as np
import cv2
from scipy.fft import dct
import pywt

class DCTInterBlockExtractor:
    """
    DCT-Domain Inter-Block Relationship Watermark Extraction
    Extracts watermark by analyzing magnitude relationships between DCT coefficients of adjacent blocks
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
    
    def extract_watermark_bit(self, dct_block1, dct_block2):
        """
        Extract a single watermark bit using inter-block relationship
        
        Args:
            dct_block1 (np.ndarray): First DCT block
            dct_block2 (np.ndarray): Second DCT block
            
        Returns:
            int: Extracted watermark bit (0 or 1)
        """
        row, col = self.coefficient_position
        
        # Get the coefficients at the specified position
        coeff1 = dct_block1[row, col]
        coeff2 = dct_block2[row, col]
        
        # Get magnitudes
        mag1 = abs(coeff1)
        mag2 = abs(coeff2)
        
        # Extract bit based on relationship
        if mag1 > mag2 + self.threshold / 2:
            return 1
        else:
            return 0
    
    def extract_watermark(self, watermarked_image, watermark_length):
        """
        Extract watermark from watermarked image
        
        Args:
            watermarked_image (np.ndarray): Watermarked image
            watermark_length (int): Expected length of watermark
            
        Returns:
            np.ndarray: Extracted binary watermark
        """
        print("Starting watermark extraction...")
        
        # Preprocess image
        ll_band = self.preprocess_image(watermarked_image)
        print(f"LL band shape: {ll_band.shape}")
        
        # Divide into blocks
        blocks = self.divide_into_blocks(ll_band)
        print(f"Number of blocks: {len(blocks)}")
        
        # Apply DCT to blocks
        dct_blocks = self.apply_dct_to_blocks(blocks)
        print("DCT applied to all blocks")
        
        # Extract watermark bits
        num_pairs = len(dct_blocks) // 2
        actual_length = min(watermark_length, num_pairs)
        
        print(f"Extracting {actual_length} watermark bits...")
        
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
        
        extracted_watermark = np.array(extracted_bits)
        print("Watermark extraction completed!")
        
        return extracted_watermark
    
    def calculate_ber(self, original_watermark, extracted_watermark):
        """
        Calculate Bit Error Rate (BER)
        
        Args:
            original_watermark (np.ndarray): Original watermark
            extracted_watermark (np.ndarray): Extracted watermark
            
        Returns:
            float: Bit Error Rate
        """
        min_length = min(len(original_watermark), len(extracted_watermark))
        if min_length == 0:
            return 1.0
        
        original_truncated = original_watermark[:min_length]
        extracted_truncated = extracted_watermark[:min_length]
        
        errors = np.sum(original_truncated != extracted_truncated)
        ber = errors / min_length
        
        return ber
    
    def calculate_similarity(self, original_watermark, extracted_watermark):
        """
        Calculate similarity between original and extracted watermarks
        
        Args:
            original_watermark (np.ndarray): Original watermark
            extracted_watermark (np.ndarray): Extracted watermark
            
        Returns:
            float: Similarity score (0-1, higher is better)
        """
        min_length = min(len(original_watermark), len(extracted_watermark))
        if min_length == 0:
            return 0.0
        
        original_truncated = original_watermark[:min_length]
        extracted_truncated = extracted_watermark[:min_length]
        
        matches = np.sum(original_truncated == extracted_truncated)
        similarity = matches / min_length
        
        return similarity

def main():
    """
    Test the extractor with sample data
    """
    # Create a sample watermarked image
    watermarked_image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    
    # Initialize extractor
    extractor = DCTInterBlockExtractor()
    
    # Extract watermark
    watermark_length = 100
    extracted_watermark = extractor.extract_watermark(watermarked_image, watermark_length)
    
    print(f"Extracted watermark: {extracted_watermark[:10]}...")
    print(f"Watermark length: {len(extracted_watermark)}")

if __name__ == "__main__":
    main()
