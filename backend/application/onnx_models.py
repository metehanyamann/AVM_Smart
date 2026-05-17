"""
Direct ONNX model inference for RetinaFace (SCRFD) and ArcFace.
Bypasses insightface Python package compilation issues on Windows.
Uses the same underlying ONNX models from insightface's buffalo_l model pack.
"""

import os
import sys
import zipfile
import logging
import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple
import time

logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "infrastructure",
    "models",
)
MODEL_PACK_URL = (
    "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
)
DET_MODEL_FILE = "det_10g.onnx"
REC_MODEL_FILE = "w600k_r50.onnx"


def download_models(force: bool = False) -> bool:
    """Download InsightFace buffalo_l ONNX models if not already present."""
    det_path = os.path.join(MODELS_DIR, DET_MODEL_FILE)
    rec_path = os.path.join(MODELS_DIR, REC_MODEL_FILE)

    if not force and os.path.exists(det_path) and os.path.exists(rec_path):
        logger.info("ONNX models already present, skipping download")
        return True

    os.makedirs(MODELS_DIR, exist_ok=True)
    zip_path = os.path.join(MODELS_DIR, "buffalo_l.zip")

    try:
        import urllib.request

        logger.info("Downloading InsightFace buffalo_l models (~316 MB)...")
        logger.info(f"URL: {MODEL_PACK_URL}")

        last_pct = [-1]

        def _progress(count, block_size, total_size):
            if total_size > 0:
                pct = int(count * block_size * 100 / total_size)
                pct = min(pct, 100)
                if pct >= last_pct[0] + 10:
                    last_pct[0] = pct
                    logger.info(f"  Download progress: {pct}%")

        urllib.request.urlretrieve(MODEL_PACK_URL, zip_path, reporthook=_progress)
        logger.info("Download complete. Extracting required models...")

        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in zf.namelist():
                basename = os.path.basename(name)
                if basename in (DET_MODEL_FILE, REC_MODEL_FILE):
                    data = zf.read(name)
                    out_path = os.path.join(MODELS_DIR, basename)
                    with open(out_path, "wb") as f:
                        f.write(data)
                    logger.info(f"  Extracted: {basename} ({len(data) / 1024 / 1024:.1f} MB)")

        if os.path.exists(zip_path):
            os.remove(zip_path)
            logger.info("  Cleaned up ZIP file")

        success = os.path.exists(det_path) and os.path.exists(rec_path)
        if success:
            logger.info("RetinaFace + ArcFace ONNX models ready!")
        else:
            logger.error("Model extraction failed - files missing after extraction")
        return success

    except Exception as e:
        logger.error(f"Model download failed: {e}")
        if os.path.exists(zip_path):
            try:
                os.remove(zip_path)
            except OSError:
                pass
        return False


def _distance2bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def _distance2kps(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    num_points = distance.shape[1] // 2
    result = np.zeros((distance.shape[0], num_points, 2), dtype=np.float32)
    for i in range(num_points):
        result[:, i, 0] = points[:, 0] + distance[:, i * 2]
        result[:, i, 1] = points[:, 1] + distance[:, i * 2 + 1]
    return result


def _nms(dets: np.ndarray, threshold: float) -> List[int]:
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))

        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-10)
        inds = np.where(iou <= threshold)[0]
        order = order[inds + 1]

    return keep


class SCRFDDetector:
    """
    SCRFD (RetinaFace) face detector using ONNX Runtime.
    Same underlying model as insightface buffalo_l det_10g.
    """

    def __init__(self, model_path: str = None):
        import onnxruntime as ort

        if model_path is None:
            model_path = os.path.join(MODELS_DIR, DET_MODEL_FILE)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Detection model not found: {model_path}")

        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape

        if isinstance(input_shape[2], int) and isinstance(input_shape[3], int):
            self.input_size = (input_shape[3], input_shape[2])
        else:
            self.input_size = (320, 320)

        self.input_mean = 127.5
        self.input_std = 128.0

        outputs = self.session.get_outputs()
        num_outputs = len(outputs)
        self.output_names = [o.name for o in outputs]

        self.use_kps = False
        self.fmc = 3
        self.feat_stride_fpn = [8, 16, 32]
        self.num_anchors = 2

        if num_outputs == 6:
            pass
        elif num_outputs == 9:
            self.use_kps = True
        elif num_outputs == 10:
            self.fmc = 5
            self.feat_stride_fpn = [8, 16, 32, 64, 128]
            self.num_anchors = 1
        elif num_outputs == 15:
            self.fmc = 5
            self.feat_stride_fpn = [8, 16, 32, 64, 128]
            self.num_anchors = 1
            self.use_kps = True

        self.nms_threshold = 0.4
        self._anchor_cache = {}

        logger.info(
            f"SCRFD (RetinaFace) detector loaded: input={self.input_size}, "
            f"outputs={num_outputs}, use_kps={self.use_kps}"
        )

    def detect(
        self,
        img: np.ndarray,
        score_threshold: float = 0.5,
        max_num: int = 0,
    ) -> List[Dict]:
        """
        Detect faces in image.

        Returns list of dicts with keys:
            bbox: (x, y, w, h)
            confidence: float
            landmarks: [[x,y], ...] (5 points) or None
        """
        start_time = time.time()

        det_img, det_scale = self._preprocess(img)

        blob = cv2.dnn.blobFromImage(
            det_img,
            1.0 / self.input_std,
            tuple(self.input_size),
            (self.input_mean, self.input_mean, self.input_mean),
            swapRB=True,
        )

        net_outs = self.session.run(self.output_names, {self.input_name: blob})

        input_height = blob.shape[2]
        input_width = blob.shape[3]

        scores_list = []
        bboxes_list = []
        kps_list = []

        for idx, stride in enumerate(self.feat_stride_fpn):
            scores = net_outs[idx]
            bbox_preds = net_outs[idx + self.fmc] * stride

            if self.use_kps:
                kps_preds = net_outs[idx + self.fmc * 2] * stride

            height = input_height // stride
            width = input_width // stride

            key = (height, width, stride)
            if key not in self._anchor_cache:
                anchor_centers = np.stack(
                    np.mgrid[:height, :width][::-1], axis=-1
                ).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape(-1, 2)
                if self.num_anchors > 1:
                    anchor_centers = np.stack(
                        [anchor_centers] * self.num_anchors, axis=1
                    ).reshape(-1, 2)
                self._anchor_cache[key] = anchor_centers

            anchor_centers = self._anchor_cache[key]

            bboxes = _distance2bbox(anchor_centers, bbox_preds.reshape(-1, 4))

            scores_flat = scores.reshape(-1)
            pos_inds = np.where(scores_flat >= score_threshold)[0]

            if len(pos_inds) == 0:
                continue

            pos_scores = scores_flat[pos_inds]
            pos_bboxes = bboxes[pos_inds]

            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)

            if self.use_kps:
                kps = _distance2kps(anchor_centers, kps_preds.reshape(-1, 10))
                pos_kps = kps[pos_inds]
                kps_list.append(pos_kps)

        if len(scores_list) == 0:
            return []

        all_scores = np.concatenate(scores_list)
        all_bboxes = np.concatenate(bboxes_list)

        if self.use_kps and len(kps_list) > 0:
            all_kps = np.concatenate(kps_list)
        else:
            all_kps = None

        pre_det = np.hstack((all_bboxes, all_scores.reshape(-1, 1))).astype(
            np.float32
        )
        keep = _nms(pre_det, self.nms_threshold)

        det = pre_det[keep]
        if all_kps is not None:
            all_kps = all_kps[keep]

        order = det[:, 4].argsort()[::-1]
        det = det[order]
        if all_kps is not None:
            all_kps = all_kps[order]

        if max_num > 0 and det.shape[0] > max_num:
            det = det[:max_num]
            if all_kps is not None:
                all_kps = all_kps[:max_num]

        det[:, :4] /= det_scale
        if all_kps is not None:
            all_kps[:, :, :2] /= det_scale

        detections = []
        for i in range(det.shape[0]):
            x1, y1, x2, y2, score = det[i]
            result = {
                "bbox": (int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                "confidence": float(score),
                "landmarks": all_kps[i].tolist() if all_kps is not None else None,
            }
            detections.append(result)

        elapsed = (time.time() - start_time) * 1000
        if detections:
            logger.info(
                f"SCRFD (RetinaFace) detected {len(detections)} face(s) in {elapsed:.1f}ms"
            )

        return detections

    def _preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, float]:
        input_w, input_h = self.input_size
        img_h, img_w = img.shape[:2]

        im_ratio = float(img_h) / img_w
        model_ratio = float(input_h) / input_w

        if im_ratio > model_ratio:
            new_height = input_h
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_w
            new_height = int(new_width * im_ratio)

        det_scale = float(new_height) / img_h

        resized = cv2.resize(img, (new_width, new_height))
        det_img = np.zeros((input_h, input_w, 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized

        return det_img, det_scale


class ArcFaceRecognizer:
    """
    ArcFace face recognizer using ONNX Runtime.
    Same underlying model as insightface buffalo_l w600k_r50.
    Produces 512-dimensional L2-normalized face embeddings.
    """

    ARCFACE_DST = np.array(
        [
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ],
        dtype=np.float32,
    )

    def __init__(self, model_path: str = None):
        import onnxruntime as ort

        if model_path is None:
            model_path = os.path.join(MODELS_DIR, REC_MODEL_FILE)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Recognition model not found: {model_path}")

        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        self.input_size = (input_shape[3], input_shape[2])
        self.output_names = [o.name for o in self.session.get_outputs()]

        self.input_mean = 127.5
        self.input_std = 127.5

        logger.info(
            f"ArcFace recognizer loaded: input={self.input_size}, output_dim=512"
        )

    def get_embedding(
        self,
        img: np.ndarray,
        landmarks: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """
        Extract 512D face embedding.

        Args:
            img: BGR image (full frame or face crop)
            landmarks: 5x2 facial landmarks for alignment (strongly recommended)

        Returns:
            512D L2-normalized float32 embedding, or None on failure
        """
        try:
            start_time = time.time()

            if landmarks is not None:
                lm = np.array(landmarks, dtype=np.float32)
                if lm.shape == (10,):
                    lm = lm.reshape(5, 2)
                aligned = self._align_face(img, lm)
            else:
                aligned = self._center_crop(img)

            if aligned is None:
                return None

            blob = cv2.dnn.blobFromImage(
                aligned,
                1.0 / self.input_std,
                self.input_size,
                (self.input_mean, self.input_mean, self.input_mean),
                swapRB=True,
            )

            embedding = self.session.run(
                self.output_names, {self.input_name: blob}
            )[0][0]

            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            elapsed = (time.time() - start_time) * 1000
            logger.info(f"ArcFace embedding extracted in {elapsed:.1f}ms")

            return embedding.astype(np.float32)

        except Exception as e:
            logger.error(f"ArcFace embedding extraction failed: {e}")
            return None

    def _align_face(self, img: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        dst = self.ARCFACE_DST.copy()
        M = self._umeyama(landmarks, dst)
        aligned = cv2.warpAffine(img, M, (112, 112), borderValue=0.0)
        return aligned

    def _center_crop(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        size = min(h, w)
        y_start = (h - size) // 2
        x_start = (w - size) // 2
        cropped = img[y_start : y_start + size, x_start : x_start + size]
        return cv2.resize(cropped, (112, 112))

    @staticmethod
    def _umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        """Similarity transform estimation (Umeyama algorithm)."""
        num = src.shape[0]
        dim = src.shape[1]

        src_mean = src.mean(axis=0)
        dst_mean = dst.mean(axis=0)

        src_demean = src - src_mean
        dst_demean = dst - dst_mean

        A = dst_demean.T @ src_demean / num

        d = np.ones((dim,), dtype=np.float64)
        if np.linalg.det(A) < 0:
            d[dim - 1] = -1

        T = np.eye(dim + 1, dtype=np.float64)

        U, S, Vt = np.linalg.svd(A)

        rank = np.linalg.matrix_rank(A)
        if rank == 0:
            return np.eye(2, 3, dtype=np.float32)

        if rank == dim - 1:
            if np.linalg.det(U) * np.linalg.det(Vt) > 0:
                T[:dim, :dim] = U @ Vt
            else:
                s = d[dim - 1]
                d[dim - 1] = -1
                T[:dim, :dim] = U @ np.diag(d) @ Vt
                d[dim - 1] = s
        else:
            T[:dim, :dim] = U @ np.diag(d) @ Vt

        src_var = src_demean.var(axis=0).sum()
        scale = (S * d).sum() / (src_var + 1e-14)

        T[:dim, dim] = dst_mean - scale * (T[:dim, :dim] @ src_mean)
        T[:dim, :dim] *= scale

        return T[:2, :].astype(np.float32)


# ---------------------------------------------------------------------------
# Singleton access
# ---------------------------------------------------------------------------

_scrfd_detector: Optional[SCRFDDetector] = None
_arcface_recognizer: Optional[ArcFaceRecognizer] = None


def get_scrfd_detector() -> Optional[SCRFDDetector]:
    global _scrfd_detector
    if _scrfd_detector is None:
        det_path = os.path.join(MODELS_DIR, DET_MODEL_FILE)
        if not os.path.exists(det_path):
            logger.warning("SCRFD model not found. Run download_models() first.")
            return None
        try:
            _scrfd_detector = SCRFDDetector(det_path)
        except Exception as e:
            logger.error(f"Failed to load SCRFD: {e}")
            return None
    return _scrfd_detector


def get_arcface_recognizer() -> Optional[ArcFaceRecognizer]:
    global _arcface_recognizer
    if _arcface_recognizer is None:
        rec_path = os.path.join(MODELS_DIR, REC_MODEL_FILE)
        if not os.path.exists(rec_path):
            logger.warning("ArcFace model not found. Run download_models() first.")
            return None
        try:
            _arcface_recognizer = ArcFaceRecognizer(rec_path)
        except Exception as e:
            logger.error(f"Failed to load ArcFace: {e}")
            return None
    return _arcface_recognizer


def models_available() -> bool:
    det_path = os.path.join(MODELS_DIR, DET_MODEL_FILE)
    rec_path = os.path.join(MODELS_DIR, REC_MODEL_FILE)
    return os.path.exists(det_path) and os.path.exists(rec_path)
