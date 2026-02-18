from ultralytics import YOLO
import numpy as np
import cv2

class HyperpigmentationAnalyzer:

    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def predict_mask(self, image_path, conf=0.10):
        results = self.model.predict(image_path, conf=conf)

        if results[0].masks is None:
            return None

        masks = results[0].masks.data.cpu().numpy()

        # Combine all detected patches
        combined_mask = np.sum(masks, axis=0)
        combined_mask = np.clip(combined_mask, 0, 1)

        return combined_mask

    def compute_severity(self, image_path, mask, skin_mask=None):
        if mask is None:
            return 0.0

        image = cv2.imread(image_path)
        
        # ---------------------------------------------------------
        # 1. ROBUST SKIN DETECTION (The Denominator Fix)
        # ---------------------------------------------------------
        if skin_mask is None:
            # Convert to HSV to easily isolate skin tones
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Standard range for skin color in HSV
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            
            # Create a mask of what counts as "Face"
            detected_skin = cv2.inRange(hsv, lower_skin, upper_skin)
            
            # Cleanup: Fill small holes in the skin mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            detected_skin = cv2.morphologyEx(detected_skin, cv2.MORPH_CLOSE, kernel)
            
            # Calculate the actual skin area
            facial_area = np.sum(detected_skin > 0)
            
            # Fallback: If skin detection fails (e.g. bad lighting), use 90% of image
            if facial_area < (image.shape[0] * image.shape[1] * 0.1): 
                facial_area = image.shape[0] * image.shape[1] * 0.9
        else:
            facial_area = np.sum(skin_mask)

        # ---------------------------------------------------------
        # 2. SEVERITY CALCULATION
        # ---------------------------------------------------------
        # Resize mask to original image size
        mask_resized = cv2.resize(
            mask.astype(np.uint8),
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

        pigmented_area = np.sum(mask_resized)
        
        # True Ratio: (Damaged Skin) / (Total Skin)
        area_ratio = pigmented_area / (facial_area + 1e-6)

        # Sanity Clamp: You can't have >100% damage
        area_ratio = min(area_ratio, 1.0)

        # ---------------------------------------------------------
        # 3. SCALING CURVE (Adjusted for "Hallucination")
        # ---------------------------------------------------------
        # We lowered the multiplier from 8 to 4 to stop it from jumping to 80% too fast.
        # Now: 10% coverage -> ~40% Severity (High but realistic)
        #      25% coverage -> 100% Severity (Max)
        
        area_score = min(area_ratio * 4.0, 1.0) 

        # Intensity (Darkness) calculation
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        pigmented_pixels = gray[mask_resized == 1]
        
        if len(pigmented_pixels) == 0:
            return 0.0

        avg_pixel_val = np.mean(pigmented_pixels)
        intensity_score = 1.0 - (avg_pixel_val / 255.0)

        # Final Weighted Score
        final_severity = (area_score * 0.7) + (intensity_score * 0.3)

        return final_severity