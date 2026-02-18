import cv2
import numpy as np
from ultralytics import YOLO

class VascularAnalyzer:
    def __init__(self, model_path):
        """
        Initialize the Vascular/Spider Vein Analyzer.
        
        Args:
            model_path (str): Path to the trained YOLO model (specifically for thread veins/telangiectasia).
        """
        try:
            self.model = YOLO(model_path)
            self.model_loaded = True
        except Exception as e:
            print(f"Warning: Could not load Vascular model at {model_path}. Error: {e}")
            self.model = None
            self.model_loaded = False

    def predict_mask(self, image_path, conf=0.15):
        """
        Detect spider veins/broken capillaries using YOLO.
        """
        if not self.model_loaded:
            return None

        # Run inference
        results = self.model.predict(image_path, conf=conf, verbose=False)

        if not results or results[0].masks is None:
            return None

        masks = results[0].masks.data.cpu().numpy()
        combined_mask = np.sum(masks, axis=0)
        combined_mask = np.clip(combined_mask, 0, 1)

        return combined_mask

    def compute_severity(self, image_path, mask, skin_mask=None):
        """
        Compute severity for Vascular signs (Spider Veins).
        These are distinct structural defects, unlike general redness.
        """
        if mask is None:
            return 0.0

        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 1. Skin Area
        if skin_mask is not None:
             facial_area = np.sum(skin_mask > 0)
        else:
            facial_area = gray.shape[0] * gray.shape[1] 

        if facial_area == 0: return 0.0

        # 2. Vascular Area
        vascular_area = np.sum(mask > 0)
        
        # Spider veins are usually very thin lines, so they take up tiny area.
        # Even 1% coverage is actually quite severe for spider veins.
        area_ratio = vascular_area / facial_area

        # Severity Curve: 
        # 0.1% coverage -> Mild
        # 1.0% coverage -> Severe (10x sensitivity compared to pigment)
        severity = min(area_ratio * 100.0, 1.0)

        return severity
