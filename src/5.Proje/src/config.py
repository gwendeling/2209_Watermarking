import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# Users can change these two variables directly
COVER_IMAGE_PATH = os.path.join(DATA_DIR, "cover_grayscale.png")
WATERMARK_IMAGE_PATH = os.path.join(DATA_DIR, "watermark.png")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
