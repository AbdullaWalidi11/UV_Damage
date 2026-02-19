import cv2
import numpy as np

class RednessVascularAnalyzer:
    def __init__(self):
        """
        Initialize the Unified Redness & Vascular Analyzer.
        
        This module uses Color Space Analysis (LAB/HSV) to detect both:
        1. Diffuse Erythema (Sunburn/Rosacea)
        2. Vascular Signs (Spider Veins/Telangiectasia)
        """
        pass

    def detect_erythema(self, image_path, skin_mask=None):
        """
        Detect diffuse redness (Erythema) using LAB/HSV color spaces.
        """
        image = cv2.imread(image_path)
        if image is None: 
            return None
            
        # Convert to LAB (L=Lightness, A=Green-to-Red, B=Blue-to-Yellow)
        # The 'A' channel is perfect for redness detection.
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        _, a_channel, _ = cv2.split(lab)
        
        # Thresholding for "Abnormal" Redness
        # Note: Skin is naturally red. We need to find *excess* red.
        # Tuned: 145 -> 152 (More selective, less false positives on normal skin)
        redness_mask = (a_channel > 152).astype(np.uint8) 

        # If skin_mask is provided, filter out non-skin areas (lips, background, eyes)
        if skin_mask is not None:
             # Resize skin_mask provided to match image
            if skin_mask.shape[:2] != redness_mask.shape[:2]:
                 skin_mask = cv2.resize(skin_mask.astype(np.uint8), (redness_mask.shape[1], redness_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            redness_mask = cv2.bitwise_and(redness_mask, redness_mask, mask=skin_mask.astype(np.uint8))

        return redness_mask

    def compute_severity(self, image_path, redness_mask, skin_mask=None):
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 1. Skin Area
        if skin_mask is not None:
             facial_area = np.sum(skin_mask > 0)
        else:
            facial_area = gray.shape[0] * gray.shape[1] 

        if facial_area == 0: return 0.0

        # 2. Redness Area
        erythema_area = np.sum(redness_mask > 0)
        area_ratio = erythema_area / facial_area

        # 3. Intensity (How red is it?)
        # We look at the A-channel mean of the masked area
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        _, a_channel, _ = cv2.split(lab)
        
        # Ensure mask matches image dimensions (Crucial fix for passed-in resized masks)
        if redness_mask.shape[:2] != a_channel.shape[:2]:
            redness_mask = cv2.resize(redness_mask.astype(np.uint8), (a_channel.shape[1], a_channel.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        red_pixels = a_channel[redness_mask == 1]
        
        if len(red_pixels) == 0:
            return 0.0
            
        # Normalize intensity: 
        # 152 (thresh) -> 0.0 severity
        # 200 (very red) -> 1.0 severity
        avg_a_val = np.mean(red_pixels)
        intensity_score = np.clip((avg_a_val - 152) / (200 - 152), 0, 1)

        # 4. Final Score
        # Calibration:
        # Area: 30% coverage should be significant. (multiplier ~3.3)
        # Intensity: We keep as is.
        
        area_score = min(area_ratio * 3.5, 1.0)
        
        # Weighted Average:
        # Intensity matters more for "how bad it looks", Area matters for "how widespread"
        # 50/50 Split seems balanced for general evaluation
        final_severity = (area_score * 0.5) + (intensity_score * 0.5)
        
        return min(final_severity, 1.0)
