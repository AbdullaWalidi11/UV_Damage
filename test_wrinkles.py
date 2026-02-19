import cv2
import numpy as np
from modules.wrinkles_module import WrinklesAnalyzer

def test_wrinkles_module(image_path):
    print(f"Testing Wrinkles Module on: {image_path}")
    
    # 1. Initialize Analyzer
    # Ensure this path matches where you put the new model file!
    model_path = "models/wrinkles_best.pt" 
    analyzer = WrinklesAnalyzer(model_path)
    
    if not analyzer.model_loaded:
        print("Error: Wrinkles model failed to load. Please check the file path.")
        return

    # 2. Load Image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return

    # 3. Predict Mask
    # We use a low confidence threshold because wrinkles can be subtle
    wrinkles_mask = analyzer.predict_mask(image_path, conf=0.15)
    
    if wrinkles_mask is None:
        print("No wrinkles detected.")
        # Create empty mask for visualization
        wrinkles_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    # 4. Compute Severity
    # We simulate a full-face skin mask for now (using the whole image as skin area)
    severity = analyzer.compute_severity(image_path, wrinkles_mask)
    
    print(f"Wrinkle Severity: {severity:.4f} (0-1.0)")
    print(f"Wrinkle Severity: {severity*100:.2f}%")

    # 5. Visualization
    def show_resized(name, img, scale=0.5):
        h, w = img.shape[:2]
        new_dim = (int(w * scale), int(h * scale))
        resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_AREA)
        cv2.imshow(name, resized)

    # A. Draw Green Lines on Original
    overlay = image.copy()
    # Resize mask to match image if needed
    if wrinkles_mask.shape[:2] != image.shape[:2]:
        wrinkles_mask = cv2.resize(wrinkles_mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        
    overlay[wrinkles_mask == 1] = [0, 255, 0] # Green lines
    blended = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
    
    show_resized("1. Original", image)
    show_resized("2. Detected Wrinkles (Green)", blended)
    
    print("Press any key to close windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Test on the same image
    test_image = "assets/test8.webp" 
    test_wrinkles_module(test_image)
