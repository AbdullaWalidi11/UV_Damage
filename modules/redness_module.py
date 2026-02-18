import cv2
import numpy as np

class RednessAnalyzer:
    def __init__(self):
        """
        Initialize the Redness Analyzer.
        This module focuses purely on diffuse Erythema (general redness/inflammation)
        and primarily uses Color Space Analysis since it's a color property, not a shape.
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
        # A good heuristic is values > 140-145 in the A-channel.
        redness_mask = (a_channel > 145).astype(np.uint8) 

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
        
        red_pixels = a_channel[redness_mask == 1]
        
        if len(red_pixels) == 0:
            return 0.0
            
        # Normalize intensity: 
        # 145 (thresh) -> 0.0 severity
        # 200 (very red) -> 1.0 severity
        avg_a_val = np.mean(red_pixels)
        intensity_score = np.clip((avg_a_val - 145) / (200 - 145), 0, 1)

        # 4. Final Score
        # Diffuse redness is often widespread, so area matters less than intensity
        final_severity = (area_ratio * 0.4) + (intensity_score * 0.6)
        
        return min(final_severity, 1.0)
