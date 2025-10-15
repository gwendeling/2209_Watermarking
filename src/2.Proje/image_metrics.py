import numpy as np
import cv2
from typing import Tuple

class ImageMetrics:
    """
    Calculate image quality metrics for watermarking evaluation.
    """
    
    @staticmethod
    def calculate_psnr(original: np.ndarray, watermarked: np.ndarray) -> float:
        """
        Calculate Peak Signal-to-Noise Ratio (PSNR).
        
        Args:
            original: Original image
            watermarked: Watermarked image
            
        Returns:
            PSNR value in dB
        """
        # Ensure images are the same size
        if original.shape != watermarked.shape:
            raise ValueError("Images must have the same dimensions")
        
        # Convert to float64 for calculation
        original = original.astype(np.float64)
        watermarked = watermarked.astype(np.float64)
        
        # Calculate MSE
        mse = np.mean((original - watermarked) ** 2)
        
        # Avoid division by zero
        if mse == 0:
            return float('inf')
        
        # Calculate PSNR
        max_pixel_value = 255.0
        psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
        
        return psnr
    
    @staticmethod
    def calculate_ncc(original: np.ndarray, watermarked: np.ndarray) -> float:
        """
        Calculate Normalized Cross-Correlation (NCC).
        
        Args:
            original: Original image
            watermarked: Watermarked image
            
        Returns:
            NCC value between -1 and 1
        """
        # Ensure images are the same size
        if original.shape != watermarked.shape:
            raise ValueError("Images must have the same dimensions")
        
        # Convert to float64 for calculation
        original = original.astype(np.float64)
        watermarked = watermarked.astype(np.float64)
        
        # Calculate means
        mean_original = np.mean(original)
        mean_watermarked = np.mean(watermarked)
        
        # Center the images
        original_centered = original - mean_original
        watermarked_centered = watermarked - mean_watermarked
        
        # Calculate numerator
        numerator = np.sum(original_centered * watermarked_centered)
        
        # Calculate denominators
        denom_original = np.sqrt(np.sum(original_centered ** 2))
        denom_watermarked = np.sqrt(np.sum(watermarked_centered ** 2))
        
        # Avoid division by zero
        if denom_original == 0 or denom_watermarked == 0:
            return 0.0
        
        # Calculate NCC
        ncc = numerator / (denom_original * denom_watermarked)
        
        return ncc
    
    @staticmethod
    def calculate_ssim(original: np.ndarray, watermarked: np.ndarray) -> float:
        """
        Calculate Structural Similarity Index (SSIM).
        
        Args:
            original: Original image
            watermarked: Watermarked image
            
        Returns:
            SSIM value between -1 and 1
        """
        # Ensure images are the same size
        if original.shape != watermarked.shape:
            raise ValueError("Images must have the same dimensions")
        
        # Convert to float64 for calculation
        original = original.astype(np.float64)
        watermarked = watermarked.astype(np.float64)
        
        # Constants
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2
        
        # Calculate means
        mu1 = np.mean(original)
        mu2 = np.mean(watermarked)
        
        # Calculate variances and covariance
        sigma1_sq = np.var(original)
        sigma2_sq = np.var(watermarked)
        sigma12 = np.mean((original - mu1) * (watermarked - mu2))
        
        # Calculate SSIM
        numerator = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
        denominator = (mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2)
        
        ssim = numerator / denominator
        
        return ssim
    
    @staticmethod
    def calculate_all_metrics(original: np.ndarray, watermarked: np.ndarray) -> dict:
        """
        Calculate all image quality metrics.
        
        Args:
            original: Original image
            watermarked: Watermarked image
            
        Returns:
            Dictionary containing all metrics
        """
        metrics = {}
        
        try:
            metrics['PSNR'] = ImageMetrics.calculate_psnr(original, watermarked)
        except Exception as e:
            metrics['PSNR'] = f"Error: {e}"
        
        try:
            metrics['NCC'] = ImageMetrics.calculate_ncc(original, watermarked)
        except Exception as e:
            metrics['NCC'] = f"Error: {e}"
        
        try:
            metrics['SSIM'] = ImageMetrics.calculate_ssim(original, watermarked)
        except Exception as e:
            metrics['SSIM'] = f"Error: {e}"
        
        return metrics

if __name__ == "__main__":
    # Test with sample images
    import numpy as np
    
    # Create test images
    original = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    watermarked = original + np.random.normal(0, 5, original.shape).astype(np.uint8)
    watermarked = np.clip(watermarked, 0, 255)
    
    # Calculate metrics
    metrics = ImageMetrics.calculate_all_metrics(original, watermarked)
    
    print("Image Quality Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

