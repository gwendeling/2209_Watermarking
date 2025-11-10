import numpy as np
import cv2
import pywt
from typing import Tuple, Dict, List
import os

class MultiBandDirectionalEmbedder:
    """
    Multi-Band Directional Embedding using DWT and DCT with QIM.
    Embeds different metadata in different DWT sub-bands based on directionality.
    """
    
    def __init__(self, block_size: int = 8, alpha: float = 0.1):
        self.block_size = block_size
        self.alpha = alpha  # Embedding strength
        self.dct_coeff_pos = (3, 3)  # Mid-band DCT coefficient position
        
        # Sub-band assignments
        self.subband_assignments = {
            'LL': 'backup',    # Approximation band -> backup of all metadata
            'LH': 'user',      # Horizontal details -> User metadata
            'HL': 'model',     # Vertical details -> Model metadata  
            'HH': 'timestamp'  # Diagonal details -> Timestamp metadata
        }
    
    def embed_watermark(self, cover_image: np.ndarray, metadata_bits: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Embed watermark using multi-band directional approach.
        
        Args:
            cover_image: Original grayscale image
            metadata_bits: Dictionary containing binary bits for each metadata type
            
        Returns:
            Watermarked image
        """
        # Ensure image is grayscale
        if len(cover_image.shape) == 3:
            cover_image = cv2.cvtColor(cover_image, cv2.COLOR_BGR2GRAY)
        
        # Perform 1-level DWT
        coeffs = pywt.dwt2(cover_image, 'haar')
        LL, (LH, HL, HH) = coeffs
        
        # Backup bits: concatenate all metadata in a deterministic order
        backup_bits = np.concatenate([
            metadata_bits.get('user', np.array([], dtype=int)),
            metadata_bits.get('model', np.array([], dtype=int)),
            metadata_bits.get('timestamp', np.array([], dtype=int)),
        ])
        
        # Embed in each sub-band
        LL_embedded = self._embed_in_subband(LL, backup_bits, 'LL') if backup_bits.size > 0 else LL
        LH_embedded = self._embed_in_subband(LH, metadata_bits['user'], 'LH')
        HL_embedded = self._embed_in_subband(HL, metadata_bits['model'], 'HL')
        HH_embedded = self._embed_in_subband(HH, metadata_bits['timestamp'], 'HH')
        
        # Reconstruct the image
        coeffs_embedded = (LL_embedded, (LH_embedded, HL_embedded, HH_embedded))
        watermarked_image = pywt.idwt2(coeffs_embedded, 'haar')
        
        # Ensure proper data type and range
        watermarked_image = np.clip(watermarked_image, 0, 255).astype(np.uint8)
        
        return watermarked_image
    
    def _embed_in_subband(self, subband: np.ndarray, bits: np.ndarray, subband_name: str) -> np.ndarray:
        """
        Embed bits in a specific sub-band using DCT and QIM.
        """
        embedded_subband = subband.copy()
        bit_index = 0
        
        # Divide sub-band into blocks
        rows, cols = subband.shape
        for i in range(0, rows - self.block_size + 1, self.block_size):
            for j in range(0, cols - self.block_size + 1, self.block_size):
                if bit_index >= len(bits):
                    break
                    
                # Extract block
                block = subband[i:i+self.block_size, j:j+self.block_size]
                
                # Perform DCT
                dct_block = cv2.dct(block.astype(np.float32))
                
                # Calculate adaptive step size based on local variance
                local_variance = np.var(block)
                step_size = self.alpha * (1 + local_variance / 1000)
                
                # Embed bit using QIM
                bit_to_embed = bits[bit_index]
                dct_block = self._qim_embed(dct_block, bit_to_embed, step_size)
                
                # Perform inverse DCT
                embedded_block = cv2.idct(dct_block)
                
                # Place back in sub-band
                embedded_subband[i:i+self.block_size, j:j+self.block_size] = embedded_block
                
                bit_index += 1
            
            if bit_index >= len(bits):
                break
        
        return embedded_subband
    
    def _qim_embed(self, dct_block: np.ndarray, bit: int, step_size: float) -> np.ndarray:
        """
        Quantization Index Modulation (QIM) embedding.
        """
        i, j = self.dct_coeff_pos
        coeff_value = dct_block[i, j]
        
        # Define quantization steps
        if bit == 1:
            # Embed '1' - use odd multiples of step_size
            quantized_value = np.round(coeff_value / step_size)
            if quantized_value % 2 == 0:
                quantized_value += 1
            dct_block[i, j] = quantized_value * step_size
        else:
            # Embed '0' - use even multiples of step_size
            quantized_value = np.round(coeff_value / step_size)
            if quantized_value % 2 == 1:
                quantized_value += 1
            dct_block[i, j] = quantized_value * step_size
        
        return dct_block
    
    def save_watermarked_image(self, watermarked_image: np.ndarray, filepath: str) -> None:
        """Save watermarked image to file."""
        cv2.imwrite(filepath, watermarked_image)
        print(f"Watermarked image saved to: {filepath}")

class WatermarkEmbedder:
    """
    Main class for watermark embedding operations.
    """
    
    def __init__(self, block_size: int = 8, alpha: float = 0.1):
        self.embedder = MultiBandDirectionalEmbedder(block_size, alpha)
        self.watermark_generator = None
    
    def embed_from_array(self, cover_image: np.ndarray, metadata: Dict[str, str]) -> np.ndarray:
        """
        Embed watermark into an in-memory RGB/BGR image using YCbCr (YCrCb in OpenCV) color space.
        
        Args:
            cover_image: Input image as numpy array (BGR or grayscale)
            metadata: Dict with keys 'user', 'model', 'timestamp' (YYYY-MM-DD or ISO)
        
        Returns:
            Watermarked image array in BGR (uint8)
        """
        if cover_image is None or not isinstance(cover_image, np.ndarray):
            raise ValueError("cover_image must be a valid numpy array")
        
        # Ensure metadata defaults
        user = metadata.get('user', 'UnknownUser')
        model = metadata.get('model', 'UnknownModel')
        timestamp = metadata.get('timestamp')
        if not timestamp:
            # Use YYYY-MM-DD format by default
            from datetime import datetime
            timestamp = datetime.utcnow().strftime('%Y-%m-%d')
        
        # Convert metadata to bits
        from watermark_generator import WatermarkGenerator
        generator = WatermarkGenerator()
        metadata_bits = generator.create_metadata_bits(user, model, timestamp)
        
        # Convert to YCrCb (OpenCV uses YCrCb order: [Y, Cr, Cb])
        if len(cover_image.shape) == 3 and cover_image.shape[2] == 3:
            ycrcb = cv2.cvtColor(cover_image, cv2.COLOR_BGR2YCrCb)
            y_channel, cr_channel, cb_channel = cv2.split(ycrcb)
            # Embed watermark on Y channel using DWT+DCT+QIM
            watermarked_y = self.embedder.embed_watermark(y_channel, metadata_bits)
            # Merge back and convert to BGR
            merged = cv2.merge([watermarked_y, cr_channel, cb_channel])
            bgr = cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)
            return bgr
        else:
            # If grayscale provided, treat it as Y directly
            watermarked_y = self.embedder.embed_watermark(cover_image, metadata_bits)
            # Return single channel as BGR for consistency
            return cv2.cvtColor(watermarked_y, cv2.COLOR_GRAY2BGR)
    
    def embed_from_files(self, cover_image_path: str, watermark_path: str, 
                        output_path: str, metadata: Dict[str, str] = None) -> np.ndarray:
        """
        Embed watermark from files.
        
        Args:
            cover_image_path: Path to cover image
            watermark_path: Path to watermark image (optional, can generate metadata)
            output_path: Path to save watermarked image
            metadata: Optional metadata dictionary
            
        Returns:
            Watermarked image
        """
        # Load cover image
        if not os.path.exists(cover_image_path):
            raise FileNotFoundError(f"Cover image not found: {cover_image_path}")
        
        cover_image = cv2.imread(cover_image_path, cv2.IMREAD_GRAYSCALE)
        if cover_image is None:
            raise ValueError(f"Could not load cover image from: {cover_image_path}")
        
        # Generate or load metadata bits
        if metadata is None:
            metadata = {
                'user': 'User123',
                'model': 'Model456', 
                'timestamp': '20241201'
            }
        
        # Convert metadata to bits
        from watermark_generator import WatermarkGenerator
        generator = WatermarkGenerator()
        metadata_bits = generator.create_metadata_bits(
            metadata['user'], metadata['model'], metadata['timestamp']
        )
        
        # Embed watermark
        watermarked_image = self.embedder.embed_watermark(cover_image, metadata_bits)
        
        # Save result
        self.embedder.save_watermarked_image(watermarked_image, output_path)
        
        return watermarked_image

if __name__ == "__main__":
    # Example usage
    embedder = WatermarkEmbedder()
    
    # Example file paths (these should be configured)
    cover_path = "cover_image.jpg"
    output_path = "watermarked_image.jpg"
    
    try:
        watermarked = embedder.embed_from_files(cover_path, "", output_path)
        print("Watermark embedding completed successfully!")
    except Exception as e:
        print(f"Error during embedding: {e}")

