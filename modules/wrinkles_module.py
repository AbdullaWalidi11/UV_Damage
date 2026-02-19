import cv2
import numpy as np
from ultralytics import YOLO

class WrinklesAnalyzer:
    def __init__(self, model_path):
        """
        Initialize the Wrinkles Analyzer.
        
        Args:
            model_path (str): Path to the trained YOLO model (segmentation or detection).
        """
        try:
            self.model = YOLO(model_path)
            self.model_loaded = True
        except Exception as e:
            # More descriptive error
            print(f"CRITICAL WARNING: Could not load Wrinkles model at '{model_path}'. Check if file exists. Error: {e}")
            self.model = None
            self.model_loaded = False

    def predict_mask(self, image_path, conf=0.15):
        """
        Detect wrinkles and generate a binary mask.
        """
        if not self.model_loaded:
            return None

        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image at {image_path}")
            return None

        # Run inference
        results = self.model.predict(image_path, conf=conf, verbose=False)

        if not results or results[0].masks is None:
            return None

        # Filter out "Face" vs "Wrinkle" confusion (Dataset Error Fix)
        # Real wrinkles are thin lines. A mask covering the whole face is an error.
        valid_masks = []
        img_area = image.shape[0] * image.shape[1]
        
        for mask in results[0].masks.data:
            mask_np = mask.cpu().numpy()
            mask_area = np.sum(mask_np > 0)
            ratio = mask_area / img_area
            
            # If a single mask covers > 5% of the image, it's likely a Face, not a Wrinkle.
            if ratio < 0.05: 
                valid_masks.append(mask_np)
            else:
                print(f"  > Ignored large mask (Ratio: {ratio:.4f}), likely a Face/Head detection.")

        if not valid_masks:
            return None

        # Combine valid masks
        combined_mask = np.sum(valid_masks, axis=0)
        combined_mask = np.clip(combined_mask, 0, 1)

        return combined_mask

    def compute_severity(self, image_path, mask, skin_mask=None):
        """
        Compute a 0-100 severity score for wrinkles.
        
        Logic:
        1. **Density**: (Wrinkle Area / Face Area)
        2. **Depth/Contrast**: Average intensity of the wrinkle lines vs surrounding skin.
        3. **Location Weighting**: (Optional) Forehead/Eyes might be weighted differently.
        """
        if mask is None:
            return 0.0

        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize mask to match image dimensions
        if mask.shape[:2] != gray.shape[:2]:
            mask = cv2.resize(mask.astype(np.uint8), (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)

        # 1. Calculate Skin Area (Denominator)
        if skin_mask is not None:
             facial_area = np.sum(skin_mask > 0)
        else:
            # Fallback: Simple approximate skin detection or full image
            # Ideally, pass the shared skin_mask from the main pipeline here
            facial_area = gray.shape[0] * gray.shape[1] 

        # 2. Wrinkle Area (Numerator)
        wrinkle_area = np.sum(mask > 0)
        
        if facial_area == 0:
            return 0.0

        density_ratio = wrinkle_area / facial_area
        
        # Normalize Density: 
        # For wrinkles, even 3-5% coverage is significant. 
        # Let's say 5% coverage = 100% severity as a base scaling factor.
        density_score = min(density_ratio * 20.0, 1.0) 

        # 3. Intensity/Depth (Average darkness of the wrinkle lines)
        wrinkle_pixels = gray[mask == 1]
        
        if len(wrinkle_pixels) == 0:
            return 0.0
            
        # Darker pixels = deeper wrinkles. 
        # We invert it: 0 (black) -> 1.0 (severe), 255 (white) -> 0.0 (mild)
        avg_intensity = np.mean(wrinkle_pixels)
        depth_score = 1.0 - (avg_intensity / 255.0)

        # 4. Final Weighted Score
        # We might weight density higher than depth for wrinkles
        final_severity = (density_score * 0.6) + (depth_score * 0.4)

        return final_severity
