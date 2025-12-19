"""
Feature extraction utilities for MSI System.
Used by both the notebook and Flask app.
MUST MATCH the notebook's extract_features_pipeline function!
"""

import numpy as np
from PIL import Image
from skimage.feature import local_binary_pattern, hog

IMG_SIZE = (128, 128)

CLASS_NAMES = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash', 'unknown']


def extract_features_pipeline(img):
    """
    Extract features from a PIL Image.
    MUST MATCH the notebook's feature extraction exactly!
    Returns: HOG + Color + LBP = ~1878 features
    """
    
    img = img.resize(IMG_SIZE)
    
    
    hist = []
    for c in range(3):
        h, _ = np.histogram(np.array(img)[:,:,c], bins=32, range=(0, 256))
        hist.extend(h)
    color_feat = np.array(hist) / (np.sum(hist) + 1e-7)
    

    gray = np.array(img.convert("L"))
    
 
    lbp = local_binary_pattern(gray, P=16, R=2, method="uniform")
    n_bins = int(lbp.max() + 1)
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    lbp_feat = lbp_hist.astype("float") / (lbp_hist.sum() + 1e-7)
    

    hog_feat = hog(gray, orientations=9, pixels_per_cell=(16, 16),
                   cells_per_block=(2, 2), block_norm="L2-Hys")
    
  
    return np.concatenate([hog_feat, color_feat, lbp_feat])


def preprocess_image(img):
    """
    Preprocess image for feature extraction.
    Accepts numpy array (BGR from cv2) or PIL Image.
    Returns PIL Image in RGB.
    """
    
    if isinstance(img, np.ndarray):
        import cv2
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img_rgb)
    
    return img
