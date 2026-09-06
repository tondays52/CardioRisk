"""
CardioRisk AI - Retinal Feature Extraction
Processes STARE retinal images for vessel analysis
"""
import os
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
import cv2
from skimage import filters, morphology, measure
import warnings
warnings.filterwarnings('ignore')

def find_retinal_data():
    """Find retinal dataset in various locations."""
    paths = [
        r"C:\Users\tonda\Desktop\dataset\Retinals",
        r"C:\Users\tonda\Desktop\dataset\retinal",
        r"C:\Users\tonda\Desktop\data\retinal",
    ]
    
    for path in paths:
        if os.path.exists(path):
            print(f"✅ Found retinal data at: {path}")
            return path
    
    # Search recursively
    for root, dirs, files in os.walk(r"C:\Users\tonda\Desktop"):
        for dir_name in dirs:
            if dir_name.lower() in ['retinals', 'retinal']:
                full_path = os.path.join(root, dir_name)
                print(f"✅ Found retinal data at: {full_path}")
                return full_path
    
    print("❌ Retinal dataset not found")
    return None

def process_retinal_image(img_path: str) -> Optional[Dict[str, Any]]:
    """Extract features from retinal image."""
    try:
        # Read PPM file
        if img_path.endswith('.ppm'):
            # Read PPM header and data
            with open(img_path, 'rb') as f:
                # Read header
                header = f.readline().decode('ascii').strip()
                while header.startswith('#'):
                    header = f.readline().decode('ascii').strip()
                if header == 'P6':
                    # Get dimensions
                    dims = f.readline().decode('ascii').strip()
                    while dims.startswith('#'):
                        dims = f.readline().decode('ascii').strip()
                    width, height = map(int, dims.split())
                    # Get max value
                    max_val = int(f.readline().decode('ascii').strip())
                    # Read pixel data
                    img_data = f.read()
                    # Convert to numpy array
                    img = np.frombuffer(img_data, dtype=np.uint8).reshape((height, width, 3))
                    # Convert to grayscale
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                else:
                    return None
        else:
            # Handle regular image files
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            return None
        
        # Resize to standard size
        img = cv2.resize(img, (256, 256))
        
        # Enhance contrast
        img = cv2.equalizeHist(img)
        img = np.asarray(img, dtype=np.float32) / 255.0
        
        # Simple vessel detection using thresholding
        vessel = img > np.percentile(img, 85)
        
        # Calculate features
        features: Dict[str, Any] = {
            'vessel_density': np.mean(vessel),
            'vessel_area': np.sum(vessel) / (256 * 256),
            'mean_intensity': np.mean(img),
            'std_intensity': np.std(img),
            'max_intensity': np.max(img),
            'min_intensity': np.min(img)
        }
        
        return features
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

def process_retinal():
    """Process all retinal images."""
    print("=" * 60)
    print("CardioRisk AI - Retinal Feature Extraction")
    print("=" * 60)
    
    data_dir = find_retinal_data()
    if data_dir is None:
        return
    
    # Only process original images (not the label files)
    image_files = [f for f in os.listdir(data_dir) 
                   if f.endswith('.ppm') and not '.ah.' in f and not '.vk.' in f]
    print(f"Found {len(image_files)} original retinal images")
    
    if len(image_files) == 0:
        print("⚠️ No retinal images found!")
        print("Looking for all .ppm files...")
        image_files = [f for f in os.listdir(data_dir) if f.endswith('.ppm')]
        print(f"Found {len(image_files)} total .ppm files (including labels)")
        if len(image_files) > 0:
            # Use all files
            pass
        else:
            return
    
    features_list = []
    
    for i, img_file in enumerate(image_files):
        img_path = os.path.join(data_dir, img_file)
        print(f"Processing {i+1}/{len(image_files)}: {img_file}...")
        features = process_retinal_image(img_path)
        
        if features:
            features['filename'] = img_file
            features_list.append(features)
    
    if features_list:
        df = pd.DataFrame(features_list)
        
        # Create output directory
        os.makedirs("models/retinal", exist_ok=True)
        
        # Save features
        df.to_csv("models/retinal/retinal_features.csv", index=False)
        print(f"\n✅ Processed {len(df)} retinal images")
        print("\n📊 Feature Summary:")
        print(df.describe())
        
        # Show sample
        print("\n📋 Sample Features:")
        print(df.head())
        
        return df
    else:
        print("⚠️ No retinal features extracted")
        return None

if __name__ == "__main__":
    process_retinal()