"""
Main application for Multi-Band Directional Watermarking System.
This script demonstrates the complete watermarking process including embedding,
extraction, and quality evaluation.
"""

import cv2
import numpy as np
import os
from typing import Dict, Tuple

# Import our custom modules
from watermark_generator import WatermarkGenerator
from watermark_embedder import WatermarkEmbedder
from watermark_extractor import WatermarkExtractor
from image_metrics import ImageMetrics
from config import *

class MultiBandWatermarkingSystem:
    """
    Complete watermarking system with embedding, extraction, and evaluation.
    """
    
    def __init__(self, block_size: int = BLOCK_SIZE, alpha: float = ALPHA):
        self.block_size = block_size
        self.alpha = alpha
        
        # Initialize components
        self.generator = WatermarkGenerator(WATERMARK_SIZE[0], WATERMARK_SIZE[1])
        self.embedder = WatermarkEmbedder(block_size, alpha)
        self.extractor = WatermarkExtractor(block_size, alpha)
        self.metrics = ImageMetrics()
    
    def create_sample_cover_image(self, filepath: str = COVER_IMAGE_PATH) -> None:
        """
        Create a sample cover image if it doesn't exist.
        """
        if not os.path.exists(filepath):
            print(f"Creating sample cover image: {filepath}")
            
            # Create a sample image with some texture
            image = np.random.randint(0, 256, IMAGE_SIZE, dtype=np.uint8)
            
            # Add some structure to make it more realistic
            for i in range(0, IMAGE_SIZE[0], 50):
                for j in range(0, IMAGE_SIZE[1], 50):
                    image[i:i+25, j:j+25] = np.random.randint(100, 200)
            
            # Add some noise
            noise = np.random.normal(0, 10, IMAGE_SIZE).astype(np.int16)
            image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            cv2.imwrite(filepath, image)
            print(f"Sample cover image created: {filepath}")
    
    def generate_watermark(self, filepath: str = WATERMARK_IMAGE_PATH) -> None:
        """
        Generate and save a sample watermark.
        """
        print("Generating sample watermark...")
        watermark = self.generator.generate_sample_watermark()
        self.generator.save_watermark(watermark, filepath)
    
    def embed_watermark(self, cover_path: str = COVER_IMAGE_PATH, 
                       output_path: str = WATERMARKED_IMAGE_PATH) -> Tuple[np.ndarray, np.ndarray]:
        """
        Embed watermark into cover image.
        
        Returns:
            Tuple of (original_image, watermarked_image)
        """
        print("Embedding watermark...")
        
        # Load original image
        original_image = cv2.imread(cover_path, cv2.IMREAD_GRAYSCALE)
        if original_image is None:
            raise ValueError(f"Could not load cover image from: {cover_path}")
        
        # Resize if needed
        if original_image.shape != IMAGE_SIZE:
            original_image = cv2.resize(original_image, IMAGE_SIZE)
        
        # Embed watermark
        watermarked_image = self.embedder.embed_from_files(
            cover_path, "", output_path, METADATA
        )
        
        return original_image, watermarked_image
    
    def extract_watermark(self, watermarked_path: str = WATERMARKED_IMAGE_PATH) -> Dict[str, str]:
        """
        Extract watermark from watermarked image.
        
        Returns:
            Dictionary containing extracted metadata
        """
        print("Extracting watermark...")
        
        extracted_metadata = self.extractor.extract_from_file(
            watermarked_path, EXPECTED_BIT_LENGTHS
        )
        
        return extracted_metadata
    
    def evaluate_quality(self, original: np.ndarray, watermarked: np.ndarray) -> Dict[str, float]:
        """
        Evaluate image quality metrics.
        
        Returns:
            Dictionary containing quality metrics
        """
        print("Calculating quality metrics...")
        
        metrics = self.metrics.calculate_all_metrics(original, watermarked)
        
        return metrics
    
    def compare_metadata(self, extracted: Dict[str, str], original: Dict[str, str]) -> Dict[str, float]:
        """
        Compare extracted metadata with original metadata.
        
        Returns:
            Dictionary containing similarity scores
        """
        print("Comparing extracted metadata with original...")
        
        similarities = self.extractor.compare_with_original(extracted, original)
        
        return similarities
    
    def run_complete_process(self) -> None:
        """
        Run the complete watermarking process.
        """
        print("=" * 60)
        print("Multi-Band Directional Watermarking System")
        print("=" * 60)
        
        try:
            # Step 1: Create sample cover image if it doesn't exist
            self.create_sample_cover_image()
            
            # Step 2: Generate watermark
            self.generate_watermark()
            
            # Step 3: Embed watermark
            original_image, watermarked_image = self.embed_watermark()
            
            # Step 4: Extract watermark
            extracted_metadata = self.extract_watermark()
            
            # Step 5: Evaluate quality
            quality_metrics = self.evaluate_quality(original_image, watermarked_image)
            
            # Step 6: Compare metadata
            metadata_similarities = self.compare_metadata(extracted_metadata, METADATA)
            
            # Display results
            self.display_results(quality_metrics, extracted_metadata, metadata_similarities)
            
        except Exception as e:
            print(f"Error during watermarking process: {e}")
            import traceback
            traceback.print_exc()
    
    def display_results(self, quality_metrics: Dict[str, float], 
                       extracted_metadata: Dict[str, str], 
                       metadata_similarities: Dict[str, float]) -> None:
        """
        Display the results of the watermarking process.
        """
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        
        # Display quality metrics
        print("\nImage Quality Metrics:")
        print("-" * 30)
        for metric, value in quality_metrics.items():
            if isinstance(value, (int, float)):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value}")
        
        # Display original vs extracted metadata
        print("\nMetadata Comparison:")
        print("-" * 30)
        print(f"{'Type':<12} {'Original':<15} {'Extracted':<15} {'Similarity':<10}")
        print("-" * 60)
        
        for key in METADATA.keys():
            original = METADATA[key]
            extracted = extracted_metadata.get(key, "N/A")
            similarity = metadata_similarities.get(key, 0.0)
            
            print(f"{key:<12} {original:<15} {extracted:<15} {similarity:<10.3f}")
        
        # Overall assessment
        print("\nOverall Assessment:")
        print("-" * 30)
        
        psnr = quality_metrics.get('PSNR', 0)
        ncc = quality_metrics.get('NCC', 0)
        avg_similarity = np.mean(list(metadata_similarities.values()))
        
        print(f"PSNR: {psnr:.2f} dB {'(Good)' if psnr > 30 else '(Poor)'}")
        print(f"NCC: {ncc:.4f} {'(Good)' if ncc > 0.9 else '(Poor)'}")
        print(f"Average Metadata Similarity: {avg_similarity:.3f} {'(Good)' if avg_similarity > 0.8 else '(Poor)'}")
        
        if psnr > 30 and ncc > 0.9 and avg_similarity > 0.8:
            print("\n✓ Watermarking process completed successfully!")
        else:
            print("\n⚠ Watermarking process completed with some quality issues.")

def main():
    """
    Main function to run the watermarking system.
    """
    # Create and run the watermarking system
    system = MultiBandWatermarkingSystem()
    system.run_complete_process()

if __name__ == "__main__":
    main()

