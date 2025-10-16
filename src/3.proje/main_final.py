import numpy as np
import cv2
import os
from watermark_generator import WatermarkGenerator
from final_watermarker import FinalDCTWatermarker

class FinalWatermarkingSystem:
    """
    Final watermarking system with improved DCT inter-block relationship method
    """
    
    def __init__(self):
        self.generator = WatermarkGenerator()
        self.watermarker = FinalDCTWatermarker()
        
        # File paths - modify these as needed
        self.cover_image_path = "images/cover_image.jpg"
        self.watermark_image_path = "watermarks/logo_watermark.png"
        self.watermarked_output_path = "output/watermarked_image.png"
        self.extracted_watermark_path = "output/extracted_watermark.txt"
        
        # Create output directory
        os.makedirs("output", exist_ok=True)
        os.makedirs("images", exist_ok=True)
    
    def calculate_psnr(self, original, watermarked):
        """Calculate Peak Signal-to-Noise Ratio (PSNR)"""
        original = original.astype(np.float64)
        watermarked = watermarked.astype(np.float64)
        
        mse = np.mean((original - watermarked) ** 2)
        
        if mse == 0:
            return float('inf')
        
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        
        return psnr
    
    def calculate_ncc(self, original, watermarked):
        """Calculate Normalized Cross-Correlation (NCC)"""
        orig_flat = original.flatten().astype(np.float64)
        water_flat = watermarked.flatten().astype(np.float64)
        
        mean_orig = np.mean(orig_flat)
        mean_water = np.mean(water_flat)
        
        orig_centered = orig_flat - mean_orig
        water_centered = water_flat - mean_water
        
        numerator = np.sum(orig_centered * water_centered)
        denominator = np.sqrt(np.sum(orig_centered ** 2) * np.sum(water_centered ** 2))
        
        if denominator == 0:
            return 0.0
        
        ncc = numerator / denominator
        return ncc
    
    def create_sample_cover_image(self):
        """Create a sample cover image if it doesn't exist"""
        if not os.path.exists(self.cover_image_path):
            print("Creating sample cover image...")
            
            # Create a sample image with some texture
            height, width = 512, 512
            cover_image = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Add some patterns
            for i in range(0, height, 32):
                for j in range(0, width, 32):
                    color = np.random.randint(50, 200, 3)
                    cover_image[i:i+32, j:j+32] = color
            
            # Add some noise
            noise = np.random.randint(-20, 20, (height, width, 3))
            cover_image = np.clip(cover_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            # Save the image
            cv2.imwrite(self.cover_image_path, cover_image)
            print(f"Sample cover image created: {self.cover_image_path}")
    
    def load_cover_image(self):
        """Load cover image from file"""
        if not os.path.exists(self.cover_image_path):
            self.create_sample_cover_image()
        
        cover_image = cv2.imread(self.cover_image_path)
        if cover_image is None:
            raise ValueError(f"Could not load cover image from {self.cover_image_path}")
        
        print(f"Cover image loaded: {cover_image.shape}")
        return cover_image
    
    def load_watermark(self):
        """Load watermark from file or generate new one"""
        if os.path.exists(self.watermark_image_path):
            # Load watermark image and convert to binary
            watermark_image = cv2.imread(self.watermark_image_path, cv2.IMREAD_GRAYSCALE)
            if watermark_image is not None:
                # Convert image to binary watermark
                watermark_bits = (watermark_image.flatten() > 127).astype(int)
                print(f"Watermark loaded from image: {len(watermark_bits)} bits")
                return watermark_bits
        
        # Generate new watermark
        print("Generating new watermark...")
        watermark_bits = self.generator.generate_binary_watermark(64)  # Smaller watermark for better results
        
        # Save watermark
        self.generator.save_watermark(
            self.generator.generate_logo_watermark("DCT_WM"), 
            "logo_watermark.png"
        )
        
        print(f"New watermark generated: {len(watermark_bits)} bits")
        return watermark_bits
    
    def embed_watermark(self):
        """Embed watermark into cover image"""
        print("\n=== WATERMARK EMBEDDING ===")
        
        # Load cover image and watermark
        cover_image = self.load_cover_image()
        watermark_bits = self.load_watermark()
        
        # Embed watermark
        watermarked_image = self.watermarker.embed_watermark(cover_image, watermark_bits)
        
        # Save watermarked image
        cv2.imwrite(self.watermarked_output_path, watermarked_image)
        print(f"Watermarked image saved: {self.watermarked_output_path}")
        
        # Calculate quality metrics
        psnr = self.calculate_psnr(cover_image, watermarked_image)
        ncc = self.calculate_ncc(cover_image, watermarked_image)
        
        print(f"\nQuality Metrics:")
        print(f"PSNR: {psnr:.2f} dB")
        print(f"NCC: {ncc:.4f}")
        
        return watermarked_image, watermark_bits
    
    def extract_watermark(self, watermarked_image, original_watermark):
        """Extract watermark from watermarked image"""
        print("\n=== WATERMARK EXTRACTION ===")
        
        # Extract watermark
        extracted_watermark = self.watermarker.extract_watermark(
            watermarked_image, 
            len(original_watermark)
        )
        
        # Save extracted watermark
        with open(self.extracted_watermark_path, "w") as f:
            f.write(" ".join(map(str, extracted_watermark)))
        print(f"Extracted watermark saved: {self.extracted_watermark_path}")
        
        # Calculate extraction metrics
        min_length = min(len(original_watermark), len(extracted_watermark))
        if min_length > 0:
            original_truncated = original_watermark[:min_length]
            extracted_truncated = extracted_watermark[:min_length]
            
            errors = np.sum(original_truncated != extracted_truncated)
            ber = errors / min_length
            similarity = (min_length - errors) / min_length
        else:
            ber = 1.0
            similarity = 0.0
        
        print(f"\nExtraction Metrics:")
        print(f"Bit Error Rate (BER): {ber:.4f}")
        print(f"Similarity: {similarity:.4f}")
        print(f"Extracted watermark length: {len(extracted_watermark)}")
        
        return extracted_watermark
    
    def run_complete_test(self):
        """Run complete watermarking test"""
        print("DCT-Domain Inter-Block Relationship Watermarking System")
        print("=" * 60)
        
        try:
            # Embed watermark
            watermarked_image, original_watermark = self.embed_watermark()
            
            # Extract watermark
            extracted_watermark = self.extract_watermark(watermarked_image, original_watermark)
            
            # Display results
            print("\n=== FINAL RESULTS ===")
            print(f"Original watermark: {original_watermark[:10]}...")
            print(f"Extracted watermark: {extracted_watermark[:10]}...")
            
            # Calculate final metrics
            min_length = min(len(original_watermark), len(extracted_watermark))
            if min_length > 0:
                original_truncated = original_watermark[:min_length]
                extracted_truncated = extracted_watermark[:min_length]
                
                errors = np.sum(original_truncated != extracted_truncated)
                final_ber = errors / min_length
                final_similarity = (min_length - errors) / min_length
            else:
                final_ber = 1.0
                final_similarity = 0.0
            
            print(f"\nFinal Extraction Quality:")
            print(f"Bit Error Rate: {final_ber:.4f}")
            print(f"Similarity: {final_similarity:.4f}")
            
            if final_similarity > 0.7:
                print("+ Watermark extraction successful!")
            else:
                print("- Watermark extraction quality is low")
            
            print(f"\nFiles created:")
            print(f"- Cover image: {self.cover_image_path}")
            print(f"- Watermarked image: {self.watermarked_output_path}")
            print(f"- Extracted watermark: {self.extracted_watermark_path}")
            print(f"- Watermark image: {self.watermark_image_path}")
            
        except Exception as e:
            print(f"Error during watermarking process: {str(e)}")
            raise

def main():
    """Main function to run the watermarking system"""
    # Initialize the system
    system = FinalWatermarkingSystem()
    
    # Run complete test
    system.run_complete_test()

if __name__ == "__main__":
    main()

