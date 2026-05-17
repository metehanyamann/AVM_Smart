"""
Feature Extraction Service
Primary: ArcFace (ONNX Runtime) - 512D deep learning face embeddings
Fallback: Histogram + LBP - Classical 512D feature vector
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple
import time

logger = logging.getLogger(__name__)


class FeatureExtractionService:
    """
    Extract 512D face embeddings from face images.
    Primary model: ArcFace via direct ONNX Runtime (no insightface package needed).
    Fallback: Histogram + LBP when ArcFace is unavailable.
    """

    def __init__(self, model_type: str = "arcface"):
        self.model_type = model_type
        self.arcface_onnx = None
        self._insightface_model = None
        self._load_arcface()

    def _load_arcface(self):
        """Load ArcFace model - try ONNX direct first, then insightface package"""
        try:
            from backend.application.onnx_models import get_arcface_recognizer

            recognizer = get_arcface_recognizer()
            if recognizer is not None:
                self.arcface_onnx = recognizer
                self.model_type = "arcface"
                logger.info("ArcFace (ONNX) loaded as PRIMARY feature extractor")
                return

            logger.warning("ArcFace ONNX model not available, trying insightface package...")
        except Exception as e:
            logger.warning(f"ONNX ArcFace loading failed: {e}")

        try:
            from insightface.app import FaceAnalysis

            logger.info("Loading ArcFace via insightface package...")
            self._insightface_model = FaceAnalysis(
                name="buffalo_l", allowed_modules=["detection", "recognition"]
            )
            self._insightface_model.prepare(ctx_id=-1)
            self.model_type = "arcface"
            logger.info("ArcFace (insightface) loaded successfully")
            return
        except Exception as e:
            logger.warning(f"insightface ArcFace loading failed: {e}")

        logger.info("Falling back to Histogram+LBP feature extraction...")
        self.model_type = "histogram_lbp"

    def extract_features_arcface(self, face_roi: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract 512D embedding using ArcFace.
        Tries to detect landmarks in the ROI for proper alignment.
        """
        if self.arcface_onnx is not None:
            return self._extract_onnx_arcface(face_roi)

        if self._insightface_model is not None:
            return self._extract_insightface(face_roi)

        logger.warning("No ArcFace model loaded")
        return None

    def _extract_onnx_arcface(self, face_roi: np.ndarray) -> Optional[np.ndarray]:
        """Extract embedding using ONNX ArcFace with center crop (fast, no re-detection)."""
        try:
            if face_roi is None or face_roi.size == 0:
                return None

            embedding = self.arcface_onnx.get_embedding(face_roi, landmarks=None)
            return embedding

        except Exception as e:
            logger.error(f"ONNX ArcFace extraction failed: {e}")
            return None

    def _extract_insightface(self, face_roi: np.ndarray) -> Optional[np.ndarray]:
        """Extract embedding using insightface FaceAnalysis."""
        try:
            if face_roi is None or face_roi.size == 0:
                return None

            start_time = time.time()
            results = self._insightface_model.get(face_roi)

            if not results or len(results) == 0:
                return None

            embedding = results[0].embedding.astype(np.float32)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            elapsed = (time.time() - start_time) * 1000
            logger.info(f"ArcFace (insightface) extraction in {elapsed:.2f}ms")
            return embedding

        except Exception as e:
            logger.error(f"InsightFace extraction failed: {e}")
            return None

    def extract_features_histogram_lbp(
        self, face_roi: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Extract 512D embedding using Histogram (256D) + LBP (256D).
        Fallback method when ArcFace is unavailable.
        """
        try:
            if face_roi is None or face_roi.size == 0:
                logger.error("Invalid face image")
                return None

            start_time = time.time()

            face_resized = cv2.resize(face_roi, (128, 128))
            face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)

            hist = cv2.calcHist([face_gray], [0], None, [256], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()

            lbp_array = self._compute_lbp(face_gray)
            lbp_hist = np.histogram(lbp_array, bins=256, range=(0, 256))[0]
            lbp_hist = lbp_hist.astype(np.float32) / (np.sum(lbp_hist) + 1e-10)

            features = np.concatenate([hist, lbp_hist]).astype(np.float32)
            features = self._normalize_vector(features)

            processing_time = (time.time() - start_time) * 1000
            logger.info(
                f"Histogram+LBP extraction complete in {processing_time:.2f}ms"
            )

            return features

        except Exception as e:
            logger.error(f"Histogram+LBP extraction failed: {e}")
            return None

    def _compute_lbp(self, image: np.ndarray) -> np.ndarray:
        """Compute Local Binary Patterns for texture features"""
        try:
            h, w = image.shape
            lbp = np.zeros((h, w), dtype=np.uint8)

            offsets = [
                (-1, -1), (-1, 0), (-1, 1), (0, -1),
                (0, 1), (1, -1), (1, 0), (1, 1),
            ]

            for i in range(1, h - 1):
                for j in range(1, w - 1):
                    center = image[i, j]
                    pattern = 0
                    for bit_pos, (di, dj) in enumerate(offsets):
                        if image[i + di, j + dj] >= center:
                            pattern |= 1 << bit_pos
                    lbp[i, j] = pattern

            return lbp.flatten()

        except Exception as e:
            logger.error(f"LBP computation failed: {e}")
            return np.zeros((h - 2) * (w - 2), dtype=np.uint8)

    def extract_features(
        self, face_roi: np.ndarray
    ) -> Tuple[Optional[np.ndarray], str]:
        """
        Extract 512D embedding using best available model.
        ArcFace first (primary), then Histogram+LBP (fallback).
        """
        try:
            if self.model_type == "arcface":
                embedding = self.extract_features_arcface(face_roi)
                if embedding is not None:
                    return embedding, "arcface"
                logger.warning("ArcFace failed, trying Histogram+LBP...")

            embedding = self.extract_features_histogram_lbp(face_roi)
            if embedding is not None:
                return embedding, "histogram_lbp"

            logger.error("All extraction methods failed")
            return None, "none"

        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return None, "none"

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        try:
            norm = np.linalg.norm(vector)
            if norm == 0:
                return vector
            return (vector / (norm + 1e-10)).astype(np.float32)
        except Exception as e:
            logger.error(f"Normalization failed: {e}")
            return vector

    def batch_extract_features(self, face_rois: list) -> list:
        results = []
        for roi in face_rois:
            embedding, model_name = self.extract_features(roi)
            results.append((embedding, model_name))
        return results

    def get_model_info(self) -> dict:
        return {
            "model_type": self.model_type,
            "output_dim": 512,
            "metric": "L2 (Euclidean distance)",
            "arcface_available": self.arcface_onnx is not None
            or self._insightface_model is not None,
            "fallback_available": True,
            "primary_model": "ArcFace (InsightFace)",
            "fallback_model": "Histogram + LBP",
            "backend": "onnxruntime"
            if self.arcface_onnx is not None
            else ("insightface" if self._insightface_model is not None else "opencv"),
        }


_feature_service = None


def get_feature_service() -> FeatureExtractionService:
    global _feature_service

    if _feature_service is None:
        _feature_service = FeatureExtractionService()

    return _feature_service
