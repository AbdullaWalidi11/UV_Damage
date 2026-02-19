import cv2
import numpy as np

# Import our modules
from modules.hyperpigmentation_module import HyperpigmentationAnalyzer
from modules.wrinkles_module import WrinklesAnalyzer
from modules.redness_module import RednessVascularAnalyzer
from modules.face_segmentation_module import FaceSegmenter

def main():
    # --- CONFIGURATION ---
    image_path = "assets/test7.jpg" # Change to your test image
    
    # Model Paths
    pigment_model = "models/hyperpigmentation_best.pt"
    wrinkle_model = "models/wrinkles_best.pt"
    
    # Initialize Analyzers
    print("Loading models...")
    pigment_analyzer = HyperpigmentationAnalyzer(pigment_model)
    wrinkle_analyzer = WrinklesAnalyzer(wrinkle_model)
    redness_analyzer = RednessVascularAnalyzer()
    face_segmenter = FaceSegmenter()
    
    # Load Image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image at {image_path}")
        return

    print(f"Processing {image_path}...")

    # --- 0. PREPROCESSING (Face Segmentation) ---
    # Using MediaPipe Face Mesh for precise skin/face masking.
    print("Segmenting Face (MediaPipe)...")
    skin_mask = face_segmenter.get_skin_mask(image)
    
    if skin_mask is None or cv2.countNonZero(skin_mask) == 0:
         print("Warning: No face detected. Falling back to simple full-image analysis (or maybe check image).")
         skin_mask = np.ones(image.shape[:2], dtype=np.uint8) * 255 # Fallback to all-white if fails? Or maybe error?
    else:
         # Optional: Erode slightly to avoid boundary artifacts
         kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
         skin_mask = cv2.erode(skin_mask, kernel, iterations=1)

    # DEBUG: visual check for user
    cv2.imwrite("debug_skin_mask.jpg", skin_mask)
    print("  > Saved debug_skin_mask.jpg (MediaPipe Face Mask).")


    with open("results.log", "w") as log: # logging to console and file
      # --- 1. HYPERPIGMENTATION ANALYSIS ---
      print("Running Hyperpigmentation Analysis...")
      # User requested modification: conf=0.15. (High conf=0.8 misses subtle spots)
      pigment_mask = pigment_analyzer.predict_mask(image_path, conf=0.25)

      # --- STRICT SKIN MASKING (Anti-Hallucination) ---
      # Models can "hallucinate" on hair/background/neck. We MUST force them to respect the Skin Mask.
      
      # Resize skin_mask to match if needed (crucial for safety)
      if skin_mask.shape[:2] != (image.shape[0], image.shape[1]):
           skin_mask = cv2.resize(skin_mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

      if pigment_mask is not None:
           if pigment_mask.shape[:2] != skin_mask.shape[:2]:
                pigment_mask = cv2.resize(pigment_mask.astype(np.uint8), (skin_mask.shape[1], skin_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
           pigment_mask = cv2.bitwise_and(pigment_mask, pigment_mask, mask=skin_mask)

      pigment_score = pigment_analyzer.compute_severity(image_path, pigment_mask, skin_mask)
      print(f"  > Score: {pigment_score:.4f} ({pigment_score*100:.1f}%)")

      # --- 2. WRINKLE ANALYSIS ---
      print("Running Wrinkle Analysis...")
      # Tuned: 0.50 (High confidence to avoid texture/hair)
      wrinkle_mask = wrinkle_analyzer.predict_mask(image_path, conf=0.20)
      
      if wrinkle_mask is not None:
           if wrinkle_mask.shape[:2] != skin_mask.shape[:2]:
                wrinkle_mask = cv2.resize(wrinkle_mask.astype(np.uint8), (skin_mask.shape[1], skin_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
           wrinkle_mask = cv2.bitwise_and(wrinkle_mask, wrinkle_mask, mask=skin_mask)

      wrinkle_score = wrinkle_analyzer.compute_severity(image_path, wrinkle_mask, skin_mask)
      print(f"  > Score: {wrinkle_score:.4f} ({wrinkle_score*100:.1f}%)")

      # --- 3. REDNESS/VASCULAR ANALYSIS ---
      print("Running Redness/Vascular Analysis...")
      redness_mask = redness_analyzer.detect_erythema(image_path, skin_mask)
      
      # --- CONFLICT RESOLUTION (Smart Logic) ---
      # Issue: Inflamed red skin (rosacea/sunburn) can sometimes be distinct enough 
      # to trigger the pigment model or just look like "dark spots" to computer vision.
      # User Request: "If area is covered by redness, don't override with pigment."
      
      if redness_mask is not None and pigment_mask is not None:
          # Resize redness mask to MATCH pigment mask dimensions exactly
          # OpenCV resize expects (width, height), but shape is (height, width)
          target_h, target_w = pigment_mask.shape[:2]
          
          # Use a temporary mask so we don't degrade the original high-res redness mask
          temp_red_mask = redness_mask.copy()
          if temp_red_mask.shape[:2] != (target_h, target_w):
               temp_red_mask = cv2.resize(temp_red_mask.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST)
          
          # Ensure types match
          pigment_mask = pigment_mask.astype(np.uint8)
          temp_red_mask = temp_red_mask.astype(np.uint8)

          # LOGIC: If it's RED, it cannot be PIGMENT (Melanin).
          overlap = cv2.bitwise_and(pigment_mask, temp_red_mask)
          pigment_mask = cv2.bitwise_xor(pigment_mask, overlap)
          
          print("  > Applied Logic: Removed Hyperpigmentation detection from Redness areas.")

      # Re-compute pigment score with the cleaned mask
      pigment_score = pigment_analyzer.compute_severity(image_path, pigment_mask, skin_mask)
      print(f"  > Score (Corrected): {pigment_score:.4f} ({pigment_score*100:.1f}%)")

      redness_score = redness_analyzer.compute_severity(image_path, redness_mask, skin_mask)
      print(f"  > Score: {redness_score:.4f} ({redness_score*100:.1f}%)")

      # --- 4. UNIFIED UV DAMAGE SCORE ---
      # Weighted Average based on dermo-aesthetic impact (User Specified)
      # Pigment: 50% (Primary Indicator of UV)
      # Wrinkles: 30% (Secondary/Structural)
      # Redness: 20% (Least specific/Noisy)
      
      final_score = (pigment_score * 0.50) + (wrinkle_score * 0.30) + (redness_score * 0.20)
      final_score_scaled = min(final_score * 100, 100.0) # Scale to 0-100

      print("\n" + "="*30)
      print(f"FINAL UV DAMAGE SCORE: {final_score_scaled:.1f} / 100")
      print("="*30)

    # --- 5. VISUALIZATION ---
    # Create a composite overlay
    # Red = Hyperpigmentation
    # Green = Wrinkles
    # Blue = Diffuse Redness
    
    overlay = np.zeros_like(image)
    
    # 5a. RESIZE ALL MASKS TO IMAGE SIZE FIRST (For clean visualization)
    if pigment_mask is not None and pigment_mask.shape[:2] != image.shape[:2]:
        pigment_mask = cv2.resize(pigment_mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    if wrinkle_mask is not None and wrinkle_mask.shape[:2] != image.shape[:2]:
        wrinkle_mask = cv2.resize(wrinkle_mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    if redness_mask is not None and redness_mask.shape[:2] != image.shape[:2]:
        redness_mask = cv2.resize(redness_mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        
    # 5b. FINAL CONFLICT RESOLUTION (Pixel-Perfect)
    # Ensure Redness (Blue) strictly overwrites Pigment (Red) at full resolution
    if redness_mask is not None and pigment_mask is not None:
         pigment_mask[redness_mask == 1] = 0

    # 5c. BUILD OVERLAY
    if pigment_mask is not None:
        overlay[pigment_mask == 1] = [0, 0, 255] # Red

    if wrinkle_mask is not None:
        overlay[wrinkle_mask == 1] = [0, 255, 0] # Green

    if redness_mask is not None:
         # Blue (Redness) - No need for overwrite logic since we cleared pigment_mask
         overlay[redness_mask == 1] = [255, 0, 0]

    # Blend
    blended = cv2.addWeighted(image, 0.7, overlay, 0.4, 0)
    
    # --- 6. LEGEND & TEXT ---
    h, w = image.shape[:2]
    
    def put_text_with_outline(img, text, pos, color, scale=0.8, thickness=2):
        x, y = pos
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness + 2) # White outline
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

    # Score (Top Left)
    put_text_with_outline(blended, f"UV Score: {final_score_scaled:.1f}", (20, 50), (0, 0, 0), scale=1.2, thickness=3)

    # Legend (Bottom Left)
    legend_y = h - 100
    put_text_with_outline(blended, "Red: Hyperpigmentation", (20, legend_y), (0, 0, 255))
    put_text_with_outline(blended, "Green: Wrinkles", (20, legend_y + 35), (0, 255, 0))
    put_text_with_outline(blended, "Blue: Redness/Vascular", (20, legend_y + 70), (255, 0, 0))

    # --- 7. DISPLAY ---
    # Concatenate Original and Result Side-by-Side
    combined_result = cv2.hconcat([image, blended])

    # Show (Resized)
    def show_resized(name, img, scale=0.5):
        h, w = img.shape[:2]
        new_dim = (int(w * scale), int(h * scale))
        resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_AREA)
        cv2.imshow(name, resized)

    show_resized("Analysis Result (Original vs Analyzed)", combined_result, scale=0.5)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()