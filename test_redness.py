import cv2
import numpy as np
from modules.redness_module import RednessAnalyzer

def test_redness_module(image_path):
    print(f"Testing Redness Module on: {image_path}")
    
    # 1. Initialize Analyzer
    analyzer = RednessAnalyzer()
    
    # 2. Load Image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return

    # 3. Simulate a Skin Mask (Critical for your test)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Adjusted Thresholds:
    # Lower: Very light skin to dark skin
    # Upper: We increase Saturation (S) and Value (V) range to capture inflamed/red skin
    lower_skin = np.array([0, 15, 50], dtype=np.uint8)
    upper_skin = np.array([179, 255, 255], dtype=np.uint8) # Broad detection
    
    # Initial mask
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # 4. Refinement: Fill Holes (Morphological Closing)
    # This is crucial for filling in the "red holes" on cheeks
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)) # Larger kernel
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    
    # Remove small noise
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    # 4. Run Redness Detection
    # Pass the skin_mask to ensure we only look at skin
    redness_mask = analyzer.detect_erythema(image_path, skin_mask=skin_mask)
    
    if redness_mask is None:
        print("Error: Redness detection failed.")
        return

    # 5. Compute Severity
    severity = analyzer.compute_severity(image_path, redness_mask, skin_mask=skin_mask)
    print(f"Redness Severity: {severity:.4f} (0-1.0)")
    print(f"Redness Severity: {severity*100:.2f}%")

    # 6. Visualization
    # Create a nice visual stack
    
    # A. Skin Mask (Gray)
    skin_visual = cv2.cvtColor(skin_mask, cv2.COLOR_GRAY2BGR)
    
    # B. Redness Mask (Blue for high contrast)
    redness_visual = np.zeros_like(image)
    redness_visual[redness_mask == 1] = [255, 0, 0] # Blue
    
    # C. Overlay on Original
    # Highlight redness in Blue on the original image
    overlay = image.copy()
    overlay[redness_mask == 1] = [255, 0, 0] # Pint redness blue
    blended = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

    # Show results (Resized for viewing)
    def show_resized(name, img, scale=0.5):
        h, w = img.shape[:2]
        new_dim = (int(w * scale), int(h * scale))
        resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_AREA)
        cv2.imshow(name, resized)

    show_resized("1. Original", image)
    show_resized("2. Skin Mask (Simulated)", skin_mask)
    show_resized("3. Detected Redness (Mask)", redness_visual)
    show_resized("4. Overlay", blended)
    
    print("Press any key to close windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Use one of your existing test images
    # Make sure this file exists!
    test_image = "assets/test1.jpg" 
    test_redness_module(test_image)
