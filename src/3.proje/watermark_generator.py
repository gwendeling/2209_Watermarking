import numpy as np
import cv2
import os

class WatermarkGenerator:
    """
    Generates watermark images for testing purposes
    """
    
    def __init__(self, width=64, height=64):
        self.width = width
        self.height = height
    
    def generate_binary_watermark(self, num_bits=256):
        """
        Generate a random binary watermark
        
        Args:
            num_bits (int): Number of bits in the watermark
            
        Returns:
            np.ndarray: Binary watermark array
        """
        return np.random.randint(0, 2, num_bits)
    
    def generate_logo_watermark(self, text="WATERMARK"):
        """
        Generate a text-based watermark image
        
        Args:
            text (str): Text to display in watermark
            
        Returns:
            np.ndarray: Watermark image
        """
        # Create a white background
        watermark = np.ones((self.height, self.width), dtype=np.uint8) * 255
        
        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (0, 0, 0)  # Black text
        thickness = 2
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Center the text
        x = (self.width - text_width) // 2
        y = (self.height + text_height) // 2
        
        cv2.putText(watermark, text, (x, y), font, font_scale, color, thickness)
        
        return watermark
    
    def generate_pattern_watermark(self):
        """
        Generate a geometric pattern watermark
        
        Returns:
            np.ndarray: Watermark image
        """
        watermark = np.zeros((self.height, self.width), dtype=np.uint8)
        
        # Create a checkerboard pattern
        block_size = 8
        for i in range(0, self.height, block_size):
            for j in range(0, self.width, block_size):
                if (i // block_size + j // block_size) % 2 == 0:
                    watermark[i:i+block_size, j:j+block_size] = 255
        
        return watermark
    
    def save_watermark(self, watermark, filename="watermark.png"):
        """
        Save watermark to file
        
        Args:
            watermark (np.ndarray): Watermark image
            filename (str): Output filename
        """
        if not os.path.exists("watermarks"):
            os.makedirs("watermarks")
        
        filepath = os.path.join("watermarks", filename)
        cv2.imwrite(filepath, watermark)
        print(f"Watermark saved to: {filepath}")
        return filepath

def main():
    """
    Generate and save sample watermarks
    """
    generator = WatermarkGenerator()
    
    # Generate different types of watermarks
    binary_watermark = generator.generate_binary_watermark(256)
    logo_watermark = generator.generate_logo_watermark("DCT_WM")
    pattern_watermark = generator.generate_pattern_watermark()
    
    # Save watermarks
    generator.save_watermark(logo_watermark, "logo_watermark.png")
    generator.save_watermark(pattern_watermark, "pattern_watermark.png")
    
    # Save binary watermark as text
    with open("watermarks/binary_watermark.txt", "w") as f:
        f.write(" ".join(map(str, binary_watermark)))
    
    print("Sample watermarks generated successfully!")
    return binary_watermark, logo_watermark, pattern_watermark

if __name__ == "__main__":
    main()

