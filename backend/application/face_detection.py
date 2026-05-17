"""
Face Detection Service
Primary: RetinaFace (SCRFD via ONNX Runtime) - High accuracy deep learning detector
Fallback: Haar Cascade - Fast CPU-based classical detector
"""

import cv2
import logging
import numpy as np
from typing import List, Tuple, Optional, Dict
import time

logger = logging.getLogger(__name__)


class FaceDetectionService:
    """
    Face detection using RetinaFace/SCRFD (primary) with Haar Cascade fallback.
    RetinaFace provides landmark detection and higher accuracy.
    """

    def __init__(self, detection_model: str = "retinaface"):
        self.detection_model = detection_model
        self.scrfd_detector = None
        self.face_cascade = None
        self._load_retinaface()

    def _load_retinaface(self):
        """Load RetinaFace (SCRFD) model via direct ONNX Runtime"""
        try:
            from backend.application.onnx_models import get_scrfd_detector

            detector = get_scrfd_detector()
            if detector is not None:
                self.scrfd_detector = detector
                self.detection_model = "retinaface"
                logger.info("RetinaFace (SCRFD/ONNX) loaded successfully as PRIMARY detector")
                return

            logger.warning("SCRFD ONNX model not available, trying insightface package...")
        except Exception as e:
            logger.warning(f"ONNX RetinaFace loading failed: {e}")

        try:
            from insightface.app import FaceAnalysis

            logger.info("Loading RetinaFace via insightface package...")
            self._insightface_model = FaceAnalysis(
                name="buffalo_l", allowed_modules=["detection"]
            )
            self._insightface_model.prepare(ctx_id=-1, det_size=(640, 640))
            self.detection_model = "retinaface"
            logger.info("RetinaFace (insightface) loaded successfully")
            return
        except Exception as e:
            logger.warning(f"insightface RetinaFace loading failed: {e}")

        logger.info("Falling back to Haar Cascade...")
        self.detection_model = "haar"
        self._load_cascade()

    def _load_cascade(self):
        """Load Haar Cascade classifier as fallback"""
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.face_cascade = cv2.CascadeClassifier(cascade_path)

            if self.face_cascade.empty():
                logger.error("Failed to load Haar Cascade")
                return False

            logger.info("Haar Cascade loaded successfully (fallback)")
            return True

        except Exception as e:
            logger.error(f"Cascade loading failed: {e}")
            return False

    @staticmethod
    def is_frontal_face(landmarks: Optional[List[List[float]]]) -> bool:
        """Check if face is frontal based on 5-point landmarks."""
        if not landmarks or len(landmarks) < 3:
            return True # Fallback if no landmarks available
            
        # Insightface/SCRFD landmarks: [left_eye, right_eye, nose, left_mouth, right_mouth]
        left_eye_x = landmarks[0][0]
        right_eye_x = landmarks[1][0]
        nose_x = landmarks[2][0]
        
        # Ensure left eye is actually to the left of right eye (standard representation)
        if left_eye_x > right_eye_x:
            left_eye_x, right_eye_x = right_eye_x, left_eye_x
            
        dist_left = nose_x - left_eye_x
        dist_right = right_eye_x - nose_x
        
        if dist_left <= 0 or dist_right <= 0:
            return False
            
        ratio = dist_left / dist_right
        # Ratio around 1.0 is perfectly frontal. 
        # > 2.5 or < 0.4 means head is turned significantly to the side
        if ratio < 0.4 or ratio > 2.5:
            return False
            
        return True

    def detect_faces(
        self,
        frame: np.ndarray,
        min_confidence: float = 0.5,
        scale_factor: float = 1.05,
        min_neighbors: int = 3,
        frontal_only: bool = True,
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in frame. Uses RetinaFace if available, otherwise Haar Cascade.
        Filters out side profiles if frontal_only is True.
        Returns list of (x, y, w, h) tuples.
        """
        if self.scrfd_detector is not None or (hasattr(self, "_insightface_model") and self._insightface_model is not None):
            # We use landmarks directly to filter
            detections = self.detect_faces_with_landmarks(frame, min_confidence, frontal_only)
            return [d["bbox"] for d in detections]
            
        return self._detect_haar(frame, scale_factor, min_neighbors)

    def detect_faces_with_landmarks(
        self, frame: np.ndarray, min_confidence: float = 0.5, frontal_only: bool = True
    ) -> List[Dict]:
        """
        Detect faces with landmarks.
        Filters out side profiles if frontal_only is True.
        Returns list of dicts with bbox, confidence, and 5-point landmarks.
        """
        detections = []
        if self.scrfd_detector is not None:
            detections = self.scrfd_detector.detect(frame, score_threshold=min_confidence)
        elif hasattr(self, "_insightface_model") and self._insightface_model is not None:
            detections = self._detect_insightface_with_landmarks(frame, min_confidence)
        else:
            faces = self._detect_haar(frame)
            return [
                {"bbox": (x, y, w, h), "confidence": 0.0, "landmarks": None}
                for x, y, w, h in faces
            ]
            
        if not frontal_only:
            return detections
            
        # Filter frontals
        frontal_detections = []
        for d in detections:
            if self.is_frontal_face(d.get("landmarks")):
                frontal_detections.append(d)
            else:
                logger.debug("Filtered out non-frontal face based on landmarks.")
                
        return frontal_detections

    def _detect_scrfd(
        self, frame: np.ndarray, min_confidence: float = 0.5
    ) -> List[Tuple[int, int, int, int]]:
        """Detect faces using SCRFD (RetinaFace) ONNX model"""
        try:
            detections = self.scrfd_detector.detect(frame, score_threshold=min_confidence)
            return [d["bbox"] for d in detections]
        except Exception as e:
            logger.error(f"SCRFD detection failed: {e}")
            if self.face_cascade is None:
                self._load_cascade()
            if self.face_cascade is not None:
                return self._detect_haar(frame)
            return []

    def _detect_insightface(
        self, frame: np.ndarray, min_confidence: float = 0.5
    ) -> List[Tuple[int, int, int, int]]:
        """Detect faces using insightface FaceAnalysis"""
        try:
            results = self._insightface_model.get(frame)
            faces = []
            for face in results:
                if float(face.det_score) < min_confidence:
                    continue
                bbox = face.bbox.astype(int)
                x, y, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                faces.append((int(x), int(y), int(x2 - x), int(y2 - y)))
            return faces
        except Exception as e:
            logger.error(f"InsightFace detection failed: {e}")
            return []

    def _detect_insightface_with_landmarks(
        self, frame: np.ndarray, min_confidence: float = 0.5
    ) -> List[Dict]:
        """Detect faces with landmarks using insightface"""
        try:
            results = self._insightface_model.get(frame)
            detections = []
            for face in results:
                det_score = float(face.det_score)
                if det_score < min_confidence:
                    continue
                bbox = face.bbox.astype(int)
                x, y, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                landmarks = face.kps.tolist() if face.kps is not None else None
                detections.append({
                    "bbox": (int(x), int(y), int(x2 - x), int(y2 - y)),
                    "confidence": det_score,
                    "landmarks": landmarks,
                })
            return detections
        except Exception as e:
            logger.error(f"InsightFace detection with landmarks failed: {e}")
            return []

    def _detect_haar(
        self,
        frame: np.ndarray,
        scale_factor: float = 1.05,
        min_neighbors: int = 3,
    ) -> List[Tuple[int, int, int, int]]:
        """Detect faces using Haar Cascade (fallback)"""
        try:
            if self.face_cascade is None:
                self._load_cascade()
            if self.face_cascade is None:
                return []

            start_time = time.time()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors,
                minSize=(20, 20),
                maxSize=(frame.shape[1], frame.shape[0]),
            )

            processing_time = (time.time() - start_time) * 1000

            if len(faces) > 0:
                logger.info(
                    f"Haar Cascade detected {len(faces)} face(s) in {processing_time:.2f}ms"
                )

            return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]

        except Exception as e:
            logger.error(f"Haar detection failed: {e}")
            return []

    def extract_roi(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        padding: int = 0,
    ) -> Optional[np.ndarray]:
        """Extract face ROI from frame"""
        try:
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(frame.shape[1], x + w + padding)
            y2 = min(frame.shape[0], y + h + padding)

            roi = frame[y1:y2, x1:x2]

            if roi.size == 0:
                return None

            return roi

        except Exception as e:
            logger.error(f"ROI extraction failed: {e}")
            return None

    def is_valid_face(self, face_roi: np.ndarray) -> bool:
        """Validate face quality: size, contrast, brightness"""
        try:
            if face_roi is None or face_roi.size == 0:
                return False

            h, w = face_roi.shape[:2]

            if h < 30 or w < 30:
                logger.warning(f"Face too small: {w}x{h}")
                return False

            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            min_variance = 10 if h > 200 else 15
            if laplacian_var < min_variance:
                logger.warning(f"Face too blurry (variance={laplacian_var:.2f})")
                return False

            brightness = np.mean(gray)
            if brightness < 10 or brightness > 245:
                logger.warning(f"Face brightness out of range: {brightness:.2f}")
                return False

            return True

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False

    def draw_face_boxes(
        self,
        frame: np.ndarray,
        faces: List[Tuple[int, int, int, int]],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """Draw bounding boxes around detected faces"""
        try:
            result = frame.copy()
            for x, y, w, h in faces:
                cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
            return result
        except Exception as e:
            logger.error(f"Drawing failed: {e}")
            return frame

    def get_model_info(self) -> dict:
        return {
            "active_model": self.detection_model,
            "retinaface_available": self.scrfd_detector is not None
            or (hasattr(self, "_insightface_model") and self._insightface_model is not None),
            "haar_available": self.face_cascade is not None,
            "supports_landmarks": self.scrfd_detector is not None
            or (hasattr(self, "_insightface_model") and self._insightface_model is not None),
            "backend": "onnxruntime"
            if self.scrfd_detector is not None
            else ("insightface" if hasattr(self, "_insightface_model") else "opencv"),
        }


_detection_service = None


def get_detection_service() -> FaceDetectionService:
    global _detection_service

    if _detection_service is None:
        _detection_service = FaceDetectionService()

    return _detection_service
