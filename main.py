import cv2
import numpy as np

from modules.hyperpigmentation_module import HyperpigmentationAnalyzer

analyzer = HyperpigmentationAnalyzer(
    "models/hyperpigmentation_best.pt"
)

mask = analyzer.predict_mask("assets/test2.jpg")
if mask is None:
    print("No hyperpigmentation detected.")
else:
    severity = analyzer.compute_severity("assets/test2.jpg", mask)

    print("Normalized Severity (0–1):", severity)
    print("Percentage Severity (0–100):", severity * 100)

    # ------------------------------------------------------------------
# 2. Load the original image
# ------------------------------------------------------------------
image_path = "assets/test2.jpg"
image = cv2.imread(image_path)

# ------------------------------------------------------------------
# 3. Resize mask to match image
# ------------------------------------------------------------------
mask_resized = cv2.resize(
    mask,
    (image.shape[1], image.shape[0]),
    interpolation=cv2.INTER_NEAREST
)

# ------------------------------------------------------------------
# 4. Create colored mask for visualization
# ------------------------------------------------------------------
colored_mask = np.zeros_like(image)

# Red color for hyperpigmentation
colored_mask[mask_resized == 1] = [0, 0, 255]  # BGR format

# ------------------------------------------------------------------
# 5. Blend the mask with the original image
# ------------------------------------------------------------------
alpha = 0.5  # Transparency factor
blended = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)

# ------------------------------------------------------------------
# 2. Wrinkle Analysis (Boilerplate integration)
# ------------------------------------------------------------------
from modules.wrinkles_module import WrinklesAnalyzer

print("\n--- Wrinkle Analysis ---")
wrinkle_analyzer = WrinklesAnalyzer("models/wrinkles_best.pt") # Placeholder path

# Assuming we want to reuse the image and skin logic later
# For now, just a standalone test
wrinkle_mask = wrinkle_analyzer.predict_mask(image_path)

if wrinkle_mask is None:
    print("No wrinkles detected (or model not found).")
else:
    wrinkle_severity = wrinkle_analyzer.compute_severity(image_path, wrinkle_mask)
    print("Normalized Severity (0–1):", wrinkle_severity)
    print("Percentage Severity (0–100):", wrinkle_severity * 100)
    
    # Visualization for Wrinkles (Green)
    wrinkle_colored = np.zeros_like(image)
    wrinkle_colored[cv2.resize(wrinkle_mask, (image.shape[1], image.shape[0])) == 1] = [0, 255, 0] # Green
    
    # Combine with previous results
    # (Just adding to the visual stack)
    blended = cv2.addWeighted(blended, 1.0, wrinkle_colored, 0.5, 0)

# ------------------------------------------------------------------
# 6. Display the results
# ------------------------------------------------------------------
cv2.imshow("Original Image", image)
cv2.imshow("Combined Analysis Result", blended)
cv2.waitKey(0)
cv2.destroyAllWindows()