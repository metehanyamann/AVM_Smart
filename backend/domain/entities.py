"""
Domain Entities
Core business objects for AvmSmart face tracking & alert system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Tuple
import time
import numpy as np

from backend.domain.enums import TrackStatus, AlertLevel, AlertStatus


# ─────────────────────────────────────────────
# Floor Visit
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────
# Tracked Face
# ─────────────────────────────────────────────

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
    status: str = TrackStatus.PENDING.value
    embedding_history: List[np.ndarray] = field(default_factory=list)
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


# ─────────────────────────────────────────────
# Wanted Person (NEW — Alert System)
# ─────────────────────────────────────────────

@dataclass
class WantedPerson:
    """A person flagged by AVM management for alert monitoring"""
    wanted_id: str          # Unique identifier
    name: str               # Person name / alias
    description: str        # Reason / notes
    embedding: np.ndarray   # 512D face embedding
    alert_level: str = AlertLevel.HIGH.value
    photo_base64: Optional[str] = None  # Original photo (base64)
    added_by: str = "admin"
    added_at: float = field(default_factory=time.time)
    is_active: bool = True

    def to_dict(self) -> dict:
        return {
            "wanted_id": self.wanted_id,
            "name": self.name,
            "description": self.description,
            "alert_level": self.alert_level,
            "has_photo": self.photo_base64 is not None,
            "added_by": self.added_by,
            "added_at": datetime.fromtimestamp(self.added_at).isoformat(),
            "is_active": self.is_active,
        }


# ─────────────────────────────────────────────
# Alert Event (NEW — Alert System)
# ─────────────────────────────────────────────

@dataclass
class AlertEvent:
    """An alert triggered when a wanted person is detected"""
    alert_id: str
    wanted_id: str
    wanted_name: str
    alert_level: str
    camera_id: str
    floor: int
    similarity_score: float
    bbox: Tuple[int, int, int, int]
    timestamp: float = field(default_factory=time.time)
    wanted_description: str = ""
    status: str = AlertStatus.ACTIVE.value
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[float] = None
    snapshot_base64: Optional[str] = None

    def to_dict(self) -> dict:
        detected_at = datetime.fromtimestamp(self.timestamp)
        return {
            "alert_id": self.alert_id,
            "wanted_id": self.wanted_id,
            "wanted_name": self.wanted_name,
            "wanted_description": self.wanted_description,
            "alert_level": self.alert_level,
            "camera_id": self.camera_id,
            "floor": self.floor,
            "similarity_score": round(self.similarity_score, 4),
            "bbox": {
                "x": self.bbox[0],
                "y": self.bbox[1],
                "width": self.bbox[2],
                "height": self.bbox[3],
            },
            "timestamp": detected_at.isoformat(),
            "timestamp_epoch": self.timestamp,
            "detected_at": detected_at.strftime("%Y-%m-%d %H:%M:%S"),
            "status": self.status,
            "acknowledged_by": self.acknowledged_by,
            "acknowledged_at": (
                datetime.fromtimestamp(self.acknowledged_at).isoformat()
                if self.acknowledged_at else None
            ),
            "has_snapshot": self.snapshot_base64 is not None,
        }
