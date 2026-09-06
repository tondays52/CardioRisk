"""
CardioRisk AI - Retinal Image Processing
Analyzes fundus images for cardiovascular risk assessment.
"""
import numpy as np
import cv2
from scipy import ndimage
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

def load_retinal_image(file_path):
    """Load and preprocess retinal fundus image."""
    image = cv2.imread(file_path)
    if image is None:
        raise ValueError(f"Could not load image: {file_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def preprocess_retinal(image):
    """
    Preprocess retinal image: resize, enhance contrast, normalize.
    """
    # Resize to standard dimensions
    image_resized = cv2.resize(image, (224, 224))
    
    # Convert to grayscale for vessel analysis
    gray = cv2.cvtColor(image_resized, cv2.COLOR_RGB2GRAY)
    
    # CLAHE contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Normalize
    normalized = (enhanced - enhanced.mean()) / (enhanced.std() + 1e-8)
    
    return image_resized, enhanced, normalized

def segment_vessels(enhanced_image):
    """
    Segment blood vessels from enhanced retinal image.
    Uses adaptive thresholding and morphological operations.
    """
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(enhanced_image, (5, 5), 0)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Morphological cleaning
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    
    # Remove small objects
    min_size = 50
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_size:
            cleaned[labels == i] = 0
    
    return cleaned

def extract_vascular_features(vessel_mask):
    """
    Extract vascular features from vessel segmentation.
    """
    # Vessel density
    total_pixels = vessel_mask.shape[0] * vessel_mask.shape[1]
    vessel_pixels = np.sum(vessel_mask > 0)
    vessel_density = vessel_pixels / total_pixels
    
    # Vessel width estimation (average thickness)
    # Use distance transform
    dist = ndimage.distance_transform_edt(vessel_mask > 0)
    assert dist is not None
    vessel_widths = dist[vessel_mask > 0]
    mean_vessel_width = np.mean(vessel_widths) if len(vessel_widths) > 0 else 0
    
    # Tortuosity estimation (ratio of actual path length to straight-line)
    # Simplified: count branching points
    skeleton = vessel_mask > 0
    # Count endpoints and branchpoints using convolution
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.float32)
    filtered = cv2.filter2D(skeleton.astype(np.float32), -1, kernel)
    neighbors = np.asarray(filtered, dtype=np.float32)
    branch_points = np.sum(neighbors > 12)
    endpoints = np.sum(neighbors == 11)
    
    # Tortuosity proxy
    tortuosity = branch_points / (vessel_pixels + 1e-8)
    
    return {
        'vessel_density': round(vessel_density, 4),
        'mean_vessel_width_px': round(mean_vessel_width, 3),
        'branch_points': int(branch_points),
        'endpoints': int(endpoints),
        'tortuosity_index': round(tortuosity, 6),
        'vessel_pixel_ratio': round(vessel_pixels / total_pixels, 4)
    }

def calculate_retinal_risk(features):
    """
    Calculate cardiovascular risk from retinal features.
    Rule-based for now; will be replaced with CNN in 499B.
    """
    risk = 0.4
    
    # Abnormal vessel density
    if features['vessel_density'] < 0.05:
        risk += 0.15
    elif features['vessel_density'] > 0.2:
        risk += 0.1
    
    # Abnormal tortuosity
    if features['tortuosity_index'] > 0.02:
        risk += 0.15
    elif features['tortuosity_index'] < 0.005:
        risk += 0.05
    
    # Vessel width abnormalities
    if features['mean_vessel_width_px'] < 1.5:
        risk += 0.1
    elif features['mean_vessel_width_px'] > 4.0:
        risk += 0.1
    
    # Branching abnormalities
    if features['branch_points'] > 200:
        risk += 0.1
    
    return np.clip(risk, 0.1, 0.9)

def process_retinal_image(file_path):
    """
    Complete retinal image processing pipeline.
    """
    # Load
    image = load_retinal_image(file_path)
    
    # Preprocess
    image_rgb, enhanced, normalized = preprocess_retinal(image)
    
    # Segment vessels
    vessel_mask = segment_vessels(enhanced)
    
    # Extract features
    features = extract_vascular_features(vessel_mask)
    
    # Calculate risk
    risk = calculate_retinal_risk(features)
    
    return {
        'features': features,
        'retinal_risk': round(risk, 4),
        'assessment': assess_retinal(features, risk)
    }

def assess_retinal(features, risk):
    """Provide clinical assessment of retinal analysis."""
    if risk > 0.6:
        return "Retinal vascular abnormalities detected - elevated cardiovascular risk"
    elif risk > 0.4:
        return "Moderate retinal vascular changes - monitor cardiovascular health"
    else:
        return "Retinal vasculature appears within normal range"

def generate_synthetic_retina(size=224):
    """
    Generate synthetic retinal image for testing.
    """
    image = np.zeros((size, size), dtype=np.uint8)
    
    # Create main vessels
    center = size // 2
    
    # Optic disc
    cv2.circle(image, (center, center), 20, (200,), -1)
    
    # Main vessels radiating outward
    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    for angle in angles:
        x_end = int(center + size * 0.45 * np.cos(angle))
        y_end = int(center + size * 0.45 * np.sin(angle))
        cv2.line(image, (center, center), (x_end, y_end), (255,), 3)
        
        # Branches
        for offset in [-0.3, 0.3]:
            branch_angle = angle + offset
            bx_end = int(center + size * 0.3 * np.cos(branch_angle))
            by_end = int(center + size * 0.3 * np.sin(branch_angle))
            mid_x = int(center + size * 0.2 * np.cos(angle))
            mid_y = int(center + size * 0.2 * np.sin(angle))
            cv2.line(image, (mid_x, mid_y), (bx_end, by_end), (200,), 2)
    
    # Add noise
    noise = np.random.normal(0, 10, image.shape)
    image = np.clip(image + noise, 0, 255).astype(np.uint8)
    
    return image

if __name__ == "__main__":
    print("Retinal Processing Module")
    print("=" * 40)
    
    # Generate synthetic retina
    print("Generating synthetic retinal image...")
    synthetic = generate_synthetic_retina()
    
    # Process
    print("Processing...")
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(synthetic)
    vessel_mask = segment_vessels(enhanced)
    features = extract_vascular_features(vessel_mask)
    risk = calculate_retinal_risk(features)
    
    print(f"\nRetinal Features:")
    for key, value in features.items():
        print(f"  {key}: {value}")
    
    print(f"\nRetinal Risk: {risk:.4f}")
    print(f"Assessment: {assess_retinal(features, risk)}")
    print("\nModule ready for real retinal images.")