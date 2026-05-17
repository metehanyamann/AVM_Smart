"""
Real-Time Multi-Camera Face Tracking Service
Supports cross-camera person re-identification using cosine similarity
on 512D ArcFace embeddings with embedding-derived hash IDs.

Key features:
- Cosine similarity matching (not L2 distance) for normalized embeddings
- Deterministic hash ID from embedding vectors for persistent tracking
- Floor-based traffic analysis with per-person duration tracking
- Cross-camera handoff with camera history
"""

import hashlib
import logging
import time
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)

# ---------- Configuration Constants ----------
# Number of consecutive frames a face must be observed
# with consistent embeddings before it gets a confirmed identity.
CONFIRM_FRAME_THRESHOLD = 10


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.
    Returns value in [-1, 1] where 1 = identical, 0 = orthogonal, -1 = opposite.
    ArcFace embeddings are L2-normalized so this is equivalent to dot product.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

def calculate_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """Calculate Intersection over Union (IoU) for moving objects.
    Assumes bounding boxes are in (x, y, w, h) format.
    """
    x1_a, y1_a, w_a, h_a = box1
    x2_a, y2_a = x1_a + w_a, y1_a + h_a

    x1_b, y1_b, w_b, h_b = box2
    x2_b, y2_b = x1_b + w_b, y1_b + h_b

    x_left = max(x1_a, x1_b)
    y_top = max(y1_a, y1_b)
    x_right = min(x2_a, x2_b)
    y_bottom = min(y2_a, y2_b)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    inter_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = w_a * h_a
    box2_area = w_b * h_b

    return inter_area / float(box1_area + box2_area - inter_area + 1e-6)


def embedding_to_hash_id(embedding: np.ndarray) -> str:
    """Generate a deterministic hash ID from a 512D embedding vector.
    
    Uses a quantized version of the embedding to create a stable hash
    that is resilient to minor variations between frames.
    The embedding is quantized to reduce noise, then SHA-256 hashed.
    
    Returns a short hex string (first 12 characters) as the person's ID.
    """
    # Quantize to reduce noise from minor frame variations
    quantized = np.round(embedding * 10).astype(np.int16)
    raw_bytes = quantized.tobytes()
    hash_digest = hashlib.sha256(raw_bytes).hexdigest()
    return f"ID-{hash_digest[:10].upper()}"


@dataclass
class FloorVisit:
    """Records a person's visit to a specific floor"""
    floor: int
    camera_id: str
    enter_time: float
    exit_time: Optional[float] = None
    
    @property
    def duration_seconds(self) -> float:
        end = self.exit_time or time.time()
        return round(end - self.enter_time, 2)
    
    def to_dict(self) -> dict:
        return {
            "floor": self.floor,
            "camera_id": self.camera_id,
            "enter_time": datetime.fromtimestamp(self.enter_time).isoformat(),
            "exit_time": datetime.fromtimestamp(self.exit_time).isoformat() if self.exit_time else None,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class TrackedFace:
    """Represents a tracked face across frames and cameras"""

    track_id: str           # UUID-based internal track ID
    hash_id: str            # Embedding-derived hash ID (persistent across cameras)
    name: Optional[str]     # Recognized name (if registered)
    embedding: np.ndarray
    bbox: Tuple[int, int, int, int]
    camera_id: str
    floor: int
    first_seen: float
    last_seen: float
    confidence: float
    frame_count: int = 1
    status: str = "PENDING" # "PENDING" or "CONFIRMED"
    embedding_history: List[np.ndarray] = field(default_factory=list)  # last N embeddings for consistency check
    camera_history: List[Dict] = field(default_factory=list)
    floor_visits: List[FloorVisit] = field(default_factory=list)
    is_active: bool = True

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "hash_id": self.hash_id,
            "name": self.name,
            "bbox": {
                "x": self.bbox[0],
                "y": self.bbox[1],
                "width": self.bbox[2],
                "height": self.bbox[3],
            },
            "camera_id": self.camera_id,
            "floor": self.floor,
            "first_seen": datetime.fromtimestamp(self.first_seen).isoformat(),
            "last_seen": datetime.fromtimestamp(self.last_seen).isoformat(),
            "confidence": round(self.confidence, 4),
            "frame_count": self.frame_count,
            "status": self.status,
            "camera_history": self.camera_history,
            "floor_visits": [fv.to_dict() for fv in self.floor_visits],
            "is_active": self.is_active,
            "duration_seconds": round(self.last_seen - self.first_seen, 2),
        }


class RealTimeTracker:
    """
    Multi-camera face tracking with cross-camera re-identification.
    
    Uses cosine similarity on 512D ArcFace embeddings for matching.
    Generates deterministic hash IDs from embeddings for persistent tracking.
    Tracks floor-based traffic patterns for analytics.
    """

    def __init__(
        self,
        match_threshold: float = 0.55,
        track_timeout: float = 60.0,
        max_tracks: int = 1000,
    ):
        """
        Args:
            match_threshold: Cosine similarity threshold for re-identification (0-1)
                            Higher = stricter matching. 0.55 is good for ArcFace.
            track_timeout: Seconds before an inactive track expires
            max_tracks: Maximum concurrent tracks
        """
        self.match_threshold = match_threshold
        self.track_timeout = track_timeout
        self.max_tracks = max_tracks

        self.global_identities: Dict[str, np.ndarray] = {}  # person_id -> embedding (for persistent tracking)
        self.global_match_threshold = 0.55  # Changed from 0.75 to 0.55 to allow cross-camera stability

        self.active_tracks: Dict[str, TrackedFace] = {}
        self.expired_tracks: List[TrackedFace] = []
        self.cameras: Dict[str, Dict] = {}
        
        # Floor traffic counters
        self.floor_visit_log: List[Dict] = []  # All floor visits for reporting
        
        # Hash ID to track mapping for cross-camera persistence
        self._hash_id_map: Dict[str, str] = {}  # hash_id -> track_id

    def register_camera(
        self, camera_id: str, location: str = "", floor: int = 0
    ) -> bool:
        """Register a camera source for tracking with floor assignment"""
        if camera_id in self.cameras:
            logger.warning(f"Camera {camera_id} already registered")
            return False

        self.cameras[camera_id] = {
            "camera_id": camera_id,
            "location": location,
            "floor": floor,
            "registered_at": time.time(),
            "last_frame_at": None,
            "total_frames": 0,
            "active_tracks": 0,
        }
        logger.info(f"Camera registered: {camera_id} (location={location}, floor={floor})")
        return True

    def unregister_camera(self, camera_id: str) -> bool:
        """Remove a camera from tracking"""
        if camera_id not in self.cameras:
            return False

        for track in list(self.active_tracks.values()):
            if track.camera_id == camera_id:
                # Close current floor visit
                if track.floor_visits:
                    track.floor_visits[-1].exit_time = time.time()
                track.is_active = False
                self.expired_tracks.append(track)
                del self.active_tracks[track.track_id]

        del self.cameras[camera_id]
        logger.info(f"Camera unregistered: {camera_id}")
        return True

    def _get_camera_floor(self, camera_id: str) -> int:
        """Get the floor number for a camera"""
        if camera_id in self.cameras:
            return self.cameras[camera_id].get("floor", 0)
        return 0

    def update(
        self,
        camera_id: str,
        detections: List[Dict],
    ) -> List[TrackedFace]:
        """
        Update tracker with new detections from a camera frame.

        Args:
            camera_id: Source camera identifier
            detections: List of dicts with 'bbox', 'embedding', 'name' (optional), 'confidence'

        Returns:
            List of active tracked faces after update
        """
        now = time.time()
        floor = self._get_camera_floor(camera_id)

        if camera_id in self.cameras:
            self.cameras[camera_id]["last_frame_at"] = now
            self.cameras[camera_id]["total_frames"] += 1

        self._expire_old_tracks(now)

        updated_tracks = []

        for detection in detections:
            bbox = detection.get("bbox", (0, 0, 0, 0))
            embedding = detection.get("embedding")
            name = detection.get("name")
            confidence = detection.get("confidence", 0.0)

            if embedding is None:
                continue

            if isinstance(embedding, list):
                embedding = np.array(embedding, dtype=np.float32)

            # L2 normalize the embedding for cosine similarity
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            # 1) Try to find matching active track FIRST (using tracking threshold ~0.55 or spatial IoU)
            # This allows the person to turn their head and drop similarity slightly
            # without getting a completely new database ID!
            matched_track = self._find_matching_track(embedding, bbox, camera_id, hash_id=None)

            if matched_track:
                # Use their existing tracked identity!
                hash_id = matched_track.hash_id

                # Alpha blending: Gradually learn the new angle (moving average)
                alpha = 0.15
                blended_emb = (1.0 - alpha) * matched_track.embedding + alpha * embedding
                blended_emb = blended_emb / np.linalg.norm(blended_emb)
                
                # Update track and system memory with the combined angle embedding
                embedding = blended_emb
                if hash_id in self.global_identities:
                    self.global_identities[hash_id] = blended_emb

            if matched_track:
                old_camera = matched_track.camera_id
                old_floor = matched_track.floor
                
                matched_track.bbox = bbox
                matched_track.last_seen = now
                matched_track.confidence = confidence
                matched_track.frame_count += 1
                matched_track.embedding = embedding
                matched_track.embedding_history.append(embedding.copy())
                if len(matched_track.embedding_history) > 5:
                    matched_track.embedding_history.pop(0)
                
                # Check status transition: PENDING → CONFIRMED
                if matched_track.status == "PENDING" and matched_track.frame_count >= CONFIRM_FRAME_THRESHOLD:
                    # Consistency gate: the rolling average embedding must be
                    # similar enough to the latest observation.
                    if len(matched_track.embedding_history) >= 3:
                        avg_emb = np.mean(matched_track.embedding_history[-5:], axis=0)
                        avg_emb = avg_emb / (np.linalg.norm(avg_emb) + 1e-8)
                        consistency = cosine_similarity(avg_emb, embedding)
                        if consistency < 0.50:
                            logger.debug(
                                f"Track {matched_track.track_id} consistency too low "
                                f"({consistency:.3f}), staying PENDING."
                            )
                            # Stay PENDING – don't promote yet
                            updated_tracks.append(matched_track)
                            continue

                    old_hash = matched_track.hash_id
                    new_hash_id = self._get_or_create_identity(embedding)
                    
                    matched_track.hash_id = new_hash_id
                    matched_track.status = "CONFIRMED"
                    hash_id = new_hash_id
                    
                    # Update hash map
                    if old_hash in self._hash_id_map:
                        del self._hash_id_map[old_hash]
                        
                    logger.info(
                        f"Track {matched_track.track_id} CONFIRMED as hash: {new_hash_id} "
                        f"after {CONFIRM_FRAME_THRESHOLD} frames."
                    )
                
                matched_track.hash_id = hash_id  # Update hash (may refine)
                
                if name and matched_track.status == "CONFIRMED":
                    matched_track.name = name

                # Cross-camera handoff
                if old_camera != camera_id:
                    # Close previous floor visit (only log if CONFIRMED)
                    if matched_track.floor_visits:
                        matched_track.floor_visits[-1].exit_time = now
                        if matched_track.status == "CONFIRMED":
                            self.floor_visit_log.append({
                                "hash_id": matched_track.hash_id,
                                "name": matched_track.name,
                                "floor": old_floor,
                                "camera_id": old_camera,
                                "duration_seconds": matched_track.floor_visits[-1].duration_seconds,
                                "timestamp": datetime.fromtimestamp(now).isoformat(),
                            })
                    
                    # Start new floor visit
                    matched_track.floor_visits.append(
                        FloorVisit(floor=floor, camera_id=camera_id, enter_time=now)
                    )
                    
                    matched_track.camera_history.append(
                        {
                            "from_camera": old_camera,
                            "to_camera": camera_id,
                            "from_floor": old_floor,
                            "to_floor": floor,
                            "timestamp": datetime.fromtimestamp(now).isoformat(),
                        }
                    )
                    matched_track.camera_id = camera_id
                    matched_track.floor = floor
                    
                    logger.info(
                        f"Cross-camera handoff: {matched_track.hash_id} "
                        f"({matched_track.name or 'unknown'}) "
                        f"floor {old_floor} -> floor {floor} "
                        f"({old_camera} -> {camera_id})"
                    )

                self._hash_id_map[hash_id] = matched_track.track_id
                updated_tracks.append(matched_track)
            else:
                # 2) Genuinely New Track: WAIT before querying the database (Verify-Before-Assign)
                import uuid
                track_id = str(uuid.uuid4())[:8]
                temp_hash_id = f"PENDING-{track_id}"

                if len(self.active_tracks) >= self.max_tracks:
                    self._evict_oldest_track()
                
                new_track = TrackedFace(
                    track_id=track_id,
                    hash_id=temp_hash_id,
                    name=name,
                    embedding=embedding,
                    bbox=bbox,
                    camera_id=camera_id,
                    floor=floor,
                    first_seen=now,
                    last_seen=now,
                    confidence=confidence,
                    status="PENDING",
                    embedding_history=[embedding.copy()],
                    camera_history=[
                        {
                            "from_camera": None,
                            "to_camera": camera_id,
                            "from_floor": None,
                            "to_floor": floor,
                            "timestamp": datetime.fromtimestamp(now).isoformat(),
                        }
                    ],
                    floor_visits=[
                        FloorVisit(floor=floor, camera_id=camera_id, enter_time=now)
                    ],
                )
                self.active_tracks[track_id] = new_track
                self._hash_id_map[temp_hash_id] = track_id
                updated_tracks.append(new_track)
                logger.debug(
                    f"New track PENDING: {track_id} (camera={camera_id})"
                )

        if camera_id in self.cameras:
            cam_tracks = sum(
                1 for t in self.active_tracks.values() if t.camera_id == camera_id
            )
            self.cameras[camera_id]["active_tracks"] = cam_tracks

        # ─── Alert System Hook ─────────────────────────────
        # Check each confirmed track against the wanted persons list
        try:
            from backend.application.alert_service import get_alert_service
            alert_svc = get_alert_service()
            for track in updated_tracks:
                if track.status == "CONFIRMED" or track.frame_count >= 2:
                    alert_svc.check_embedding(
                        embedding=track.embedding,
                        camera_id=camera_id,
                        floor=floor,
                        bbox=track.bbox,
                    )
        except Exception as e:
            logger.debug(f"Alert check skipped: {e}")
        # ──────────────────────────────────────────────────

        return updated_tracks

    def _find_matching_track(
        self, embedding: np.ndarray, bbox: Tuple[int, int, int, int], camera_id: str, hash_id: Optional[str] = None
    ) -> Optional[TrackedFace]:
        """Find existing track using cosine similarity and Spatial IoU fallback.
        
        First checks hash ID map for quick lookup, then falls back to
        cosine similarity and spatial bounding box comparison against active tracks.
        """
        # Quick lookup by hash ID
        if hash_id and hash_id in self._hash_id_map:
            track_id = self._hash_id_map[hash_id]
            if track_id in self.active_tracks:
                track = self.active_tracks[track_id]
                sim = cosine_similarity(embedding, track.embedding)
                if sim >= self.match_threshold:
                    return track
        
        # Scan with cosine similarity and Spatial Fallback (IoU)
        best_match = None
        best_similarity = -1.0
        best_iou = -1.0

        for track in self.active_tracks.values():
            similarity = cosine_similarity(embedding, track.embedding)
            
            # Allow spatial fallback ONLY if they are in the exact same camera and box hasn't jumped
            iou = 0.0
            if track.camera_id == camera_id:
                iou = calculate_iou(bbox, track.bbox)

            # High confidence face match OR Spatial tracking match (head turned so appearance drops, but location stayed same)
            if similarity >= self.match_threshold or iou >= 0.4:
                if similarity > best_similarity or iou > best_iou:
                    best_similarity = similarity
                    best_iou = iou
                    best_match = track

        if best_match:
            logger.debug(
                f"Matched track {best_match.track_id} "
                f"(hash={best_match.hash_id}) with similarity={best_similarity:.4f}"
            )

        return best_match

    def _get_or_create_identity(self, embedding: np.ndarray) -> str:
        """
        1) Check against all known persistent identities in the session.
        2) If not in session, query the global SQLite/Milvus vector database 
           for previously registered faces.
        3) Otherwise, create a new SESSION-ONLY ID (no auto-save to DB).
        
        IMPORTANT: This method does NOT auto-register unknown faces into the
        database. Only manually registered faces (via frontend "Kişiyi Kaydet")
        will persist. Unknown faces get a session-local numeric ID like "000001".
        """
        best_id = None
        best_sim = -1.0

        for pid, ref_embedding in self.global_identities.items():
            sim = cosine_similarity(embedding, ref_embedding)
            if sim > best_sim:
                best_sim = sim
                best_id = pid

        if best_id and best_sim >= self.global_match_threshold:
            return best_id

        # Query Database for Registered Users (manual registrations only)
        from backend.application.face_search import get_search_service
        try:
            svc = get_search_service()
            db_name, db_dist, is_confident = svc.identify_face(embedding)
            if db_name:
                logger.debug(f"Tracker linked registered user: {db_name} (L2 dist={db_dist:.3f})")
                self.global_identities[db_name] = embedding
                return db_name
        except Exception as e:
            logger.error(f"Failed to query face database in tracker: {e}")

        # No match found — create session-local ID (NOT saved to database)
        self._session_id_counter = getattr(self, '_session_id_counter', 0) + 1
        new_id = f"{self._session_id_counter:06d}"

        self.global_identities[new_id] = embedding
        logger.info(f"Yeni oturum kimligi olusturuldu: {new_id} (veritabanina KAYDEDILMEDI)")
        return new_id

    def _expire_old_tracks(self, now: float):
        """Move expired tracks to history and close their floor visits.
        Only CONFIRMED tracks are logged to floor_visit_log — PENDING tracks
        are silently discarded."""
        expired_ids = []
        for track_id, track in self.active_tracks.items():
            if now - track.last_seen > self.track_timeout:
                track.is_active = False
                
                # Only log floor visits for CONFIRMED tracks
                if track.status == "CONFIRMED" and track.floor_visits:
                    last_visit = track.floor_visits[-1]
                    if last_visit.exit_time is None:
                        last_visit.exit_time = track.last_seen
                    # Log the visit
                    self.floor_visit_log.append({
                        "hash_id": track.hash_id,
                        "name": track.name,
                        "floor": last_visit.floor,
                        "camera_id": last_visit.camera_id,
                        "duration_seconds": last_visit.duration_seconds,
                        "timestamp": datetime.fromtimestamp(now).isoformat(),
                    })
                
                # Only keep CONFIRMED tracks in history
                if track.status == "CONFIRMED":
                    self.expired_tracks.append(track)
                    
                expired_ids.append(track_id)
                
                # Clean hash map
                if track.hash_id in self._hash_id_map:
                    del self._hash_id_map[track.hash_id]

        for track_id in expired_ids:
            del self.active_tracks[track_id]

        if self.expired_tracks and len(self.expired_tracks) > 500:
            self.expired_tracks = self.expired_tracks[-500:]

    def _evict_oldest_track(self):
        """Remove the oldest inactive track to make room"""
        if not self.active_tracks:
            return

        oldest_id = min(
            self.active_tracks, key=lambda k: self.active_tracks[k].last_seen
        )
        track = self.active_tracks.pop(oldest_id)
        track.is_active = False
        if track.hash_id in self._hash_id_map:
            del self._hash_id_map[track.hash_id]
        self.expired_tracks.append(track)

    def get_active_tracks(
        self, camera_id: Optional[str] = None
    ) -> List[TrackedFace]:
        """Get all active CONFIRMED tracks, optionally filtered by camera.
        PENDING tracks are excluded from the public API."""
        tracks = [t for t in self.active_tracks.values() if t.status == "CONFIRMED"]
        if camera_id:
            tracks = [t for t in tracks if t.camera_id == camera_id]
        return tracks

    def get_track(self, track_id: str) -> Optional[TrackedFace]:
        """Get a specific track by ID"""
        return self.active_tracks.get(track_id)

    def get_person_trail(self, name: str) -> List[Dict]:
        """Get movement trail for a person across all cameras"""
        trail = []

        all_tracks = list(self.active_tracks.values()) + self.expired_tracks
        person_tracks = [t for t in all_tracks if t.name == name]
        person_tracks.sort(key=lambda t: t.first_seen)

        for track in person_tracks:
            trail.append(
                {
                    "track_id": track.track_id,
                    "hash_id": track.hash_id,
                    "camera_id": track.camera_id,
                    "floor": track.floor,
                    "first_seen": datetime.fromtimestamp(track.first_seen).isoformat(),
                    "last_seen": datetime.fromtimestamp(track.last_seen).isoformat(),
                    "duration_seconds": round(track.last_seen - track.first_seen, 2),
                    "camera_history": track.camera_history,
                    "floor_visits": [fv.to_dict() for fv in track.floor_visits],
                }
            )

        return trail

    def get_person_trail_by_hash(self, hash_id: str) -> List[Dict]:
        """Get movement trail for a person by their embedding hash ID"""
        trail = []

        all_tracks = list(self.active_tracks.values()) + self.expired_tracks
        person_tracks = [t for t in all_tracks if t.hash_id == hash_id]
        person_tracks.sort(key=lambda t: t.first_seen)

        for track in person_tracks:
            trail.append(
                {
                    "track_id": track.track_id,
                    "hash_id": track.hash_id,
                    "name": track.name,
                    "camera_id": track.camera_id,
                    "floor": track.floor,
                    "first_seen": datetime.fromtimestamp(track.first_seen).isoformat(),
                    "last_seen": datetime.fromtimestamp(track.last_seen).isoformat(),
                    "duration_seconds": round(track.last_seen - track.first_seen, 2),
                    "camera_history": track.camera_history,
                    "floor_visits": [fv.to_dict() for fv in track.floor_visits],
                }
            )

        return trail

    def get_floor_traffic_report(self) -> Dict:
        """
        Generate comprehensive floor traffic analysis report.
        
        Returns:
            Dict with floor-by-floor traffic data including:
            - visitor counts per floor
            - average duration per floor
            - busiest floor
            - person movement summaries
        """
        # Collect all floor visits from CONFIRMED active + expired tracks only
        # PENDING tracks are excluded from reports
        all_tracks = [
            t for t in list(self.active_tracks.values()) + self.expired_tracks
            if t.status == "CONFIRMED"
        ]
        
        floor_stats: Dict[int, Dict] = defaultdict(lambda: {
            "total_visitors": set(),
            "total_visits": 0,
            "total_duration_seconds": 0.0,
            "visit_durations": [],
        })
        
        person_movements: Dict[str, Dict] = {}  # hash_id -> movement info
        
        for track in all_tracks:
            hash_id = track.hash_id
            display_name = track.name or hash_id
            
            if hash_id not in person_movements:
                person_movements[hash_id] = {
                    "hash_id": hash_id,
                    "name": track.name,
                    "display_name": display_name,
                    "floors_visited": [],
                    "floor_durations": {},
                    "total_duration_seconds": 0.0,
                    "camera_transitions": len(track.camera_history),
                    "first_seen": datetime.fromtimestamp(track.first_seen).isoformat(),
                    "last_seen": datetime.fromtimestamp(track.last_seen).isoformat(),
                }
            
            pm = person_movements[hash_id]
            
            for fv in track.floor_visits:
                floor = fv.floor
                dur = fv.duration_seconds
                
                floor_stats[floor]["total_visitors"].add(hash_id)
                floor_stats[floor]["total_visits"] += 1
                floor_stats[floor]["total_duration_seconds"] += dur
                floor_stats[floor]["visit_durations"].append(dur)
                
                if floor not in pm["floors_visited"]:
                    pm["floors_visited"].append(floor)
                
                pm["floor_durations"][floor] = pm["floor_durations"].get(floor, 0) + dur
                pm["total_duration_seconds"] += dur
        
        # Build floor summary
        floor_summary = {}
        busiest_floor = None
        max_visitors = 0
        
        for floor, stats in sorted(floor_stats.items()):
            visitor_count = len(stats["total_visitors"])
            avg_duration = (
                stats["total_duration_seconds"] / stats["total_visits"]
                if stats["total_visits"] > 0 else 0
            )
            
            floor_summary[floor] = {
                "floor": floor,
                "unique_visitors": visitor_count,
                "total_visits": stats["total_visits"],
                "total_duration_seconds": round(stats["total_duration_seconds"], 2),
                "average_duration_seconds": round(avg_duration, 2),
                "average_duration_minutes": round(avg_duration / 60, 2),
            }
            
            if visitor_count > max_visitors:
                max_visitors = visitor_count
                busiest_floor = floor
        
        # Clean up person movements for JSON serialization
        person_list = []
        for pm in person_movements.values():
            if pm["floor_durations"]:
                most_visited = max(pm["floor_durations"], key=pm["floor_durations"].get)
                pm["most_visited_floor"] = most_visited
            else:
                pm["most_visited_floor"] = None

            pm["floor_durations"] = {
                str(k): round(v, 2) for k, v in pm["floor_durations"].items()
            }
            pm["total_duration_seconds"] = round(pm["total_duration_seconds"], 2)
            pm["total_duration_minutes"] = round(pm["total_duration_seconds"] / 60, 2)
            person_list.append(pm)
        
        return {
            "report_generated_at": datetime.now().isoformat(),
            "summary": {
                "total_unique_visitors": len(person_movements),
                "total_floors_monitored": len(floor_summary),
                "busiest_floor": busiest_floor,
                "busiest_floor_visitors": max_visitors,
            },
            "floor_details": floor_summary,
            "person_movements": person_list,
            "active_tracks_count": len(self.active_tracks),
            "expired_tracks_count": len(self.expired_tracks),
        }

    def get_cameras(self) -> List[Dict]:
        """Get all registered cameras"""
        return list(self.cameras.values())

    def get_statistics(self) -> Dict:
        """Get tracker statistics"""
        # Count per-floor active tracks
        floor_active = defaultdict(int)
        for track in self.active_tracks.values():
            floor_active[track.floor] += 1
        
        return {
            "active_tracks": len(self.active_tracks),
            "expired_tracks": len(self.expired_tracks),
            "registered_cameras": len(self.cameras),
            "cameras": {
                cam_id: cam_info for cam_id, cam_info in self.cameras.items()
            },
            "floor_active_counts": dict(floor_active),
            "match_threshold": self.match_threshold,
            "track_timeout_seconds": self.track_timeout,
            "max_tracks": self.max_tracks,
            "total_floor_visits_logged": len(self.floor_visit_log),
        }

    def clear_all(self):
        """Clear all tracks and cameras"""
        self.active_tracks.clear()
        self.expired_tracks.clear()
        self.cameras.clear()
        self.floor_visit_log.clear()
        self._hash_id_map.clear()
        logger.info("All tracking data cleared")


_tracker = None


def get_tracker(
    match_threshold: float = 0.55,
    track_timeout: float = 60.0,
) -> RealTimeTracker:
    """Get or create tracker instance (singleton)"""
    global _tracker

    if _tracker is None:
        _tracker = RealTimeTracker(
            match_threshold=match_threshold,
            track_timeout=track_timeout,
        )

    return _tracker
