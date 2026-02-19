import cv2
import mediapipe as mp
import numpy as np

# New Tasks API imports
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class FaceSegmenter:
    def __init__(self, model_path="models/face_landmarker.task"):
        """
        Initialize MediaPipe Face Landmarker for precise skin segmentation.
        """
        try:
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.detector = vision.FaceLandmarker.create_from_options(options)
            print(f"FaceSegmenter initialized with model: {model_path}")
        except Exception as e:
            print(f"Error initializing FaceSegmenter: {e}")
            self.detector = None

    def get_skin_mask(self, image):
        """
        Generates a binary mask of the face area (excluding hair/background).
        
        Args:
            image (numpy.ndarray): Input image (BGR).
            
        Returns:
            numpy.ndarray: Binary mask (0 or 255) of the face.
        """
        if image is None or self.detector is None:
            return None

        h, w = image.shape[:2]
        
        # Convert to MediaPipe Image (Tasks API requires mp.Image)
        # It expects RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        
        # Detect
        try:
            detection_result = self.detector.detect(mp_image)
        except Exception as e:
            print(f"Error during face detection: {e}")
            return np.zeros((h, w), dtype=np.uint8)
        
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if detection_result.face_landmarks:
            # We only asked for 1 face
            face_landmarks = detection_result.face_landmarks[0]
            
            points = []
            for lm in face_landmarks:
                x, y = int(lm.x * w), int(lm.y * h)
                points.append([x, y])
            
            points = np.array(points, dtype=np.int32)
            
            # Convex Hull to get the boundary of the face
            hull = cv2.convexHull(points)
            # 1. Fill Face with White (255)
            cv2.fillPoly(mask, [hull], 255)
            
        return mask
