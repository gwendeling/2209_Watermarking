import numpy as np
import cv2
import pywt
from typing import Tuple, Dict, List
import os

class MultiBandDirectionalExtractor:
    """
    Extract watermarks using multi-band directional approach.
    Extracts different metadata from different DWT sub-bands.
    """
    
    def __init__(self, block_size: int = 8, alpha: float = 0.1):
        self.block_size = block_size
        self.alpha = alpha
        self.dct_coeff_pos = (3, 3)
        
        # Sub-band assignments (same as embedder)
        self.subband_assignments = {
            'LH': 'user',
            'HL': 'model',
            'HH': 'timestamp'
        }
    
    def extract_watermark(self, watermarked_image: np.ndarray, 
                         expected_bit_lengths: Dict[str, int]) -> Dict[str, np.ndarray]:
        """
        Extract watermark from watermarked image.
        
        Args:
            watermarked_image: Watermarked grayscale image
            expected_bit_lengths: Expected number of bits for each metadata type
            
        Returns:
            Dictionary containing extracted bits for each metadata type
        """
        # Ensure image is grayscale
        if len(watermarked_image.shape) == 3:
            watermarked_image = cv2.cvtColor(watermarked_image, cv2.COLOR_BGR2GRAY)
        
        # Perform 1-level DWT
        coeffs = pywt.dwt2(watermarked_image, 'haar')
        LL, (LH, HL, HH) = coeffs
        
        # Extract from each sub-band
        extracted_bits = {}
        extracted_bits['user'] = self._extract_from_subband(LH, expected_bit_lengths['user'])
        extracted_bits['model'] = self._extract_from_subband(HL, expected_bit_lengths['model'])
        extracted_bits['timestamp'] = self._extract_from_subband(HH, expected_bit_lengths['timestamp'])
        
        return extracted_bits
    
    def _extract_from_subband(self, subband: np.ndarray, num_bits: int) -> np.ndarray:
        """
        Extract bits from a specific sub-band using DCT and QIM.
        """
        extracted_bits = []
        bit_count = 0
        
        # Divide sub-band into blocks
        rows, cols = subband.shape
        for i in range(0, rows - self.block_size + 1, self.block_size):
            for j in range(0, cols - self.block_size + 1, self.block_size):
                if bit_count >= num_bits:
                    break
                
                # Extract block
                block = subband[i:i+self.block_size, j:j+self.block_size]
                
                # Perform DCT
                dct_block = cv2.dct(block.astype(np.float32))
                
                # Calculate adaptive step size (same as embedding)
                local_variance = np.var(block)
                step_size = self.alpha * (1 + local_variance / 1000)
                
                # Extract bit using QIM
                extracted_bit = self._qim_extract(dct_block, step_size)
                extracted_bits.append(extracted_bit)
                
                bit_count += 1
            
            if bit_count >= num_bits:
                break
        
        return np.array(extracted_bits)
    
    def _qim_extract(self, dct_block: np.ndarray, step_size: float) -> int:
        """
        Quantization Index Modulation (QIM) extraction.
        """
        i, j = self.dct_coeff_pos
        coeff_value = dct_block[i, j]
        
        # Determine which quantization step is closer
        quantized_value = np.round(coeff_value / step_size)
        
        # If closer to odd multiple, bit is 1; if even, bit is 0
        if quantized_value % 2 == 1:
            return 1
        else:
            return 0
    
    def bits_to_string(self, bits: np.ndarray) -> str:
        """
        Convert binary array back to string.
        """
        # Pad bits to make length multiple of 8
        padded_bits = np.pad(bits, (0, (8 - len(bits) % 8) % 8), 'constant')
        
        # Convert to string
        binary_str = ''.join(map(str, padded_bits))
        
        # Convert binary string to text
        text = ''
        for i in range(0, len(binary_str), 8):
            byte = binary_str[i:i+8]
            if len(byte) == 8:
                char = chr(int(byte, 2))
                if char.isprintable():  # Only add printable characters
                    text += char
        
        return text.rstrip('\x00')  # Remove null characters

class WatermarkExtractor:
    """
    Main class for watermark extraction operations.
    """
    
    def __init__(self, block_size: int = 8, alpha: float = 0.1):
        self.extractor = MultiBandDirectionalExtractor(block_size, alpha)
    
    def extract_from_file(self, watermarked_image_path: str, 
                         expected_bit_lengths: Dict[str, int] = None) -> Dict[str, str]:
        """
        Extract watermark from file.
        
        Args:
            watermarked_image_path: Path to watermarked image
            expected_bit_lengths: Expected number of bits for each metadata type
            
        Returns:
            Dictionary containing extracted metadata strings
        """
        # Load watermarked image
        if not os.path.exists(watermarked_image_path):
            raise FileNotFoundError(f"Watermarked image not found: {watermarked_image_path}")
        
        watermarked_image = cv2.imread(watermarked_image_path, cv2.IMREAD_GRAYSCALE)
        if watermarked_image is None:
            raise ValueError(f"Could not load watermarked image from: {watermarked_image_path}")
        
        # Default expected bit lengths
        if expected_bit_lengths is None:
            expected_bit_lengths = {
                'user': 64,      # 8 characters * 8 bits
                'model': 64,     # 8 characters * 8 bits
                'timestamp': 64  # 8 characters * 8 bits
            }
        
        # Extract bits
        extracted_bits = self.extractor.extract_watermark(watermarked_image, expected_bit_lengths)
        
        # Convert bits to strings
        extracted_metadata = {}
        for key, bits in extracted_bits.items():
            extracted_metadata[key] = self.extractor.bits_to_string(bits)
        
        return extracted_metadata
    
    def compare_with_original(self, extracted_metadata: Dict[str, str], 
                            original_metadata: Dict[str, str]) -> Dict[str, float]:
        """
        Compare extracted metadata with original metadata.
        
        Returns:
            Dictionary with similarity scores for each metadata type
        """
        similarities = {}
        
        for key in original_metadata.keys():
            if key in extracted_metadata:
                # Calculate character-wise similarity
                original = original_metadata[key]
                extracted = extracted_metadata[key]
                
                # Pad shorter string with spaces
                max_len = max(len(original), len(extracted))
                original_padded = original.ljust(max_len)
                extracted_padded = extracted.ljust(max_len)
                
                # Calculate similarity
                matches = sum(1 for a, b in zip(original_padded, extracted_padded) if a == b)
                similarity = matches / max_len if max_len > 0 else 0
                similarities[key] = similarity
            else:
                similarities[key] = 0.0
        
        return similarities

if __name__ == "__main__":
    # Example usage
    extractor = WatermarkExtractor()
    
    # Example file path
    watermarked_path = "watermarked_image.jpg"
    
    try:
        extracted_metadata = extractor.extract_from_file(watermarked_path)
        print("Extracted metadata:")
        for key, value in extracted_metadata.items():
            print(f"{key}: {value}")
    except Exception as e:
        print(f"Error during extraction: {e}")

