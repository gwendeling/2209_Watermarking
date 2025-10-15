import numpy as np
import cv2
import os
from typing import Tuple, Dict

class WatermarkGenerator:
    """
    Generates watermark images with metadata structure for multi-band directional embedding.
    """
    
    def __init__(self, width: int = 64, height: int = 64):
        self.width = width
        self.height = height
        
    def generate_sample_watermark(self) -> np.ndarray:
        """
        Generate a sample watermark image with structured metadata.
        Returns a binary watermark image.
        """
        # Create a structured watermark with different regions
        watermark = np.zeros((self.height, self.width), dtype=np.uint8)
        
        # User metadata region (top-left)
        watermark[0:self.height//2, 0:self.width//2] = 255
        
        # Model metadata region (top-right)
        watermark[0:self.height//2, self.width//2:self.width] = 128
        
        # Timestamp metadata region (bottom)
        watermark[self.height//2:self.height, :] = 64
        
        # Add some pattern for better visibility
        for i in range(0, self.height, 8):
            for j in range(0, self.width, 8):
                if (i + j) % 16 == 0:
                    watermark[i:i+4, j:j+4] = 255
                    
        return watermark
    
    def create_metadata_bits(self, user_info: str = "User123", 
                           model_info: str = "Model456", 
                           timestamp: str = "20241201") -> Dict[str, np.ndarray]:
        """
        Convert metadata to binary bits for embedding.
        """
        metadata = {}
        
        # Convert strings to binary
        user_bits = self._string_to_bits(user_info)
        model_bits = self._string_to_bits(model_info)
        timestamp_bits = self._string_to_bits(timestamp)
        
        metadata['user'] = user_bits
        metadata['model'] = model_bits
        metadata['timestamp'] = timestamp_bits
        
        return metadata
    
    def _string_to_bits(self, text: str) -> np.ndarray:
        """Convert string to binary array."""
        binary_str = ''.join(format(ord(char), '08b') for char in text)
        return np.array([int(bit) for bit in binary_str])
    
    def save_watermark(self, watermark: np.ndarray, filepath: str) -> None:
        """Save watermark image to file."""
        cv2.imwrite(filepath, watermark)
        print(f"Watermark saved to: {filepath}")
    
    def load_watermark(self, filepath: str) -> np.ndarray:
        """Load watermark image from file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Watermark file not found: {filepath}")
        
        watermark = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if watermark is None:
            raise ValueError(f"Could not load watermark from: {filepath}")
            
        return watermark

if __name__ == "__main__":
    # Generate and save sample watermark
    generator = WatermarkGenerator()
    watermark = generator.generate_sample_watermark()
    generator.save_watermark(watermark, "sample_watermark.png")
    
    # Generate metadata bits
    metadata = generator.create_metadata_bits()
    print("Generated metadata bits:")
    for key, bits in metadata.items():
        print(f"{key}: {bits[:16]}... (length: {len(bits)})")

