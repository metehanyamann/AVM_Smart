"""
Wanted Person Alert Service
Manages the wanted_faces list and triggers alerts when a match is found
during real-time tracking.

Flow:
1. Admin uploads wanted person photo → embedding extracted → stored in memory
2. Every tracking frame, each detected embedding is compared against wanted list
3. Match above threshold → AlertEvent created → pushed to active alerts
"""

import logging
import time
import uuid
import numpy as np
from typing import List, Optional, Dict
from datetime import datetime

from backend.domain.entities import WantedPerson, AlertEvent
from backend.domain.enums import AlertLevel, AlertStatus
from backend.infrastructure.milvus_client import get_milvus_client
from backend.infrastructure.database import SessionLocal, WantedFace

logger = logging.getLogger(__name__)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class AlertService:
    """
    Manages wanted persons list and triggers alerts on detection.
    Uses in-memory store (could be backed by Milvus wanted_faces collection).
    """

    def __init__(
        self,
        similarity_threshold: float = 0.55,
        cooldown_seconds: int = 60,
    ):
        self.similarity_threshold = similarity_threshold
        self.cooldown_seconds = cooldown_seconds
        
        from backend.infrastructure.config import settings
        self.milvus_client = get_milvus_client(host=settings.milvus_host, port=settings.milvus_port)

        # In-memory stores for active alerts and history
        self.active_alerts: Dict[str, AlertEvent] = {}
        self.alert_history: List[AlertEvent] = []

        # Cooldown map: (wanted_id, camera_id) → last_alert_time
        self._cooldown_map: Dict[str, float] = {}

    def _save_wanted_metadata(self, person: WantedPerson) -> None:
        """Persist wanted-person display metadata in SQL."""
        if SessionLocal is None:
            logger.warning("SQL database unavailable; wanted metadata was not saved")
            return

        db = SessionLocal()
        try:
            existing = (
                db.query(WantedFace)
                .filter(WantedFace.wanted_id == person.wanted_id)
                .first()
            )
            if existing is None:
                existing = WantedFace(wanted_id=person.wanted_id)
                db.add(existing)

            existing.name = person.name
            existing.description = person.description
            existing.alert_level = person.alert_level
            existing.embedding = person.embedding.tolist()
            existing.timestamp = int(person.added_at)
            existing.photo_base64 = person.photo_base64
            existing.added_by = person.added_by
            existing.created_at = datetime.fromtimestamp(person.added_at)
            db.commit()
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to save wanted metadata: {e}")
        finally:
            db.close()

    def _get_wanted_metadata(self) -> Dict[str, WantedFace]:
        """Return SQL metadata keyed by wanted_id."""
        if SessionLocal is None:
            return {}

        db = SessionLocal()
        try:
            rows = db.query(WantedFace).all()
            return {row.wanted_id: row for row in rows if row.wanted_id}
        except Exception as e:
            logger.error(f"Failed to load wanted metadata: {e}")
            return {}
        finally:
            db.close()

    # ─── Wanted Person CRUD ───────────────────────────

    def add_wanted_person(
        self,
        name: str,
        description: str,
        embedding: np.ndarray,
        alert_level: str = AlertLevel.HIGH.value,
        photo_base64: Optional[str] = None,
        added_by: str = "admin",
    ) -> WantedPerson:
        """Add a wanted person to the watch list in Milvus."""
        wanted_id = f"W-{uuid.uuid4().hex[:8].upper()}"

        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = (embedding / norm).astype(np.float32)

        person = WantedPerson(
            wanted_id=wanted_id,
            name=name,
            description=description,
            embedding=embedding,
            alert_level=alert_level,
            photo_base64=photo_base64,
            added_by=added_by,
            added_at=time.time(),
        )

        milvus_id = self.milvus_client.insert_vector(
            embedding=embedding,
            name=name,
            timestamp=int(time.time()),
            collection_name="wanted_faces",
            extra_fields={"wanted_id": wanted_id, "alert_level": alert_level}
        )

        if milvus_id is None:
            logger.error("Failed to insert wanted person into Milvus")
            raise Exception("Database error")

        self._save_wanted_metadata(person)

        logger.info(f"Wanted person added: {name} (ID={wanted_id})")
        return person

    def remove_wanted_person(self, wanted_id: str) -> bool:
        """Remove a wanted person from Milvus and SQL metadata."""
        milvus_success = self.milvus_client.delete_by_wanted_id(wanted_id)
        db_success = False
        if SessionLocal is not None:
            db = SessionLocal()
            try:
                deleted = (
                    db.query(WantedFace)
                    .filter(WantedFace.wanted_id == wanted_id)
                    .delete()
                )
                db.commit()
                db_success = deleted > 0
            except Exception as e:
                db.rollback()
                logger.error(f"Failed to delete wanted metadata: {e}")
            finally:
                db.close()

        success = milvus_success or db_success
        if success:
            logger.info(f"Wanted person removed: {wanted_id}")
        return success



    def get_wanted_list(self) -> List[WantedPerson]:
        """Get all wanted persons from Milvus with SQL metadata."""
        results = self.milvus_client.get_all_vectors(collection_name="wanted_faces")
        metadata = self._get_wanted_metadata()
        persons = []
        for r in results:
            wanted_id = r.get("wanted_id", "UNKNOWN")
            meta = metadata.get(wanted_id)
            added_at = r.get("timestamp", 0)
            if meta and meta.timestamp:
                added_at = meta.timestamp
            persons.append(WantedPerson(
                wanted_id=wanted_id,
                name=(meta.name if meta and meta.name else r.get("name", "Unknown")),
                description=(meta.description if meta and meta.description else ""),
                embedding=np.zeros(512),
                alert_level=(meta.alert_level if meta and meta.alert_level else r.get("alert_level", AlertLevel.HIGH.value)),
                photo_base64=(meta.photo_base64 if meta else None),
                added_by=(meta.added_by if meta and meta.added_by else "admin"),
                added_at=added_at,
            ))

        milvus_ids = {p.wanted_id for p in persons}
        for wanted_id, meta in metadata.items():
            if wanted_id in milvus_ids:
                continue
            persons.append(WantedPerson(
                wanted_id=wanted_id,
                name=meta.name or "Unknown",
                description=meta.description or "",
                embedding=np.zeros(512),
                alert_level=meta.alert_level or AlertLevel.HIGH.value,
                photo_base64=meta.photo_base64,
                added_by=meta.added_by or "admin",
                added_at=meta.timestamp or time.time(),
            ))
        return persons

    def get_wanted_person(self, wanted_id: str) -> Optional[WantedPerson]:
        """Get a specific wanted person."""
        results = self.milvus_client.get_all_vectors(collection_name="wanted_faces")
        metadata = self._get_wanted_metadata()
        meta = metadata.get(wanted_id)
        for r in results:
            if r.get("wanted_id") == wanted_id:
                return WantedPerson(
                    wanted_id=wanted_id,
                    name=(meta.name if meta and meta.name else r.get("name", "Unknown")),
                    description=(meta.description if meta and meta.description else ""),
                    embedding=np.zeros(512),
                    alert_level=(meta.alert_level if meta and meta.alert_level else r.get("alert_level", AlertLevel.HIGH.value)),
                    photo_base64=(meta.photo_base64 if meta else None),
                    added_by=(meta.added_by if meta and meta.added_by else "admin"),
                    added_at=(meta.timestamp if meta and meta.timestamp else r.get("timestamp", 0)),
                )
        if meta:
            return WantedPerson(
                wanted_id=wanted_id,
                name=meta.name or "Unknown",
                description=meta.description or "",
                embedding=np.zeros(512),
                alert_level=meta.alert_level or AlertLevel.HIGH.value,
                photo_base64=meta.photo_base64,
                added_by=meta.added_by or "admin",
                added_at=meta.timestamp or time.time(),
            )
        return None

    # ─── Alert Matching ───────────────────────────────

    def check_embedding(
        self,
        embedding: np.ndarray,
        camera_id: str,
        floor: int,
        bbox: tuple,
    ) -> Optional[AlertEvent]:
        """
        Check an embedding against Milvus wanted_faces collection.
        Returns an AlertEvent if a match is found, None otherwise.
        """
        # L2 normalize query
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        # Distance is L2. Lower is more similar.
        # Cosine similarity roughly corresponds to 1 - (L2^2)/2 for normalized vectors.
        # But we'll let Milvus return the raw distance and handle it.
        # Wait, milvus returns distance. We passed threshold, but our Milvus threshold is actually L2 distance.
        # If we expect similarity_threshold = 0.55 (cosine), we should convert threshold or use Milvus search.
        # For L2 normalized vectors, L2_dist^2 = 2 - 2*cos_sim => cos_sim = 1 - (L2_dist^2)/2
        # Let's use the threshold from config as cosine similarity and convert to L2 distance threshold.
        # cos_sim >= 0.55 means L2_dist^2 <= 2 - 2*0.55 = 0.9.
        # So L2_dist <= sqrt(0.9) approx 0.948.
        l2_threshold = float(np.sqrt(2 - 2 * self.similarity_threshold))

        # Fetch top-3: guards against ANN approximation misses.
        # All returned results are already below l2_threshold.
        # Best match (lowest distance) is first.
        matches = self.milvus_client.search_vector(
            embedding=embedding,
            limit=3,
            threshold=l2_threshold,
            collection_name="wanted_faces"
        )

        if not matches:
            return None

        # matches format for wanted_faces: (milvus_id, name, distance, wanted_id, alert_level)
        match = matches[0]
        milvus_id, name, distance, wanted_id, alert_level = match
        wanted_person = self.get_wanted_person(wanted_id)
        if wanted_person:
            name = wanted_person.name
            alert_level = wanted_person.alert_level

        cos_sim = 1.0 - (distance ** 2) / 2.0

        # Check cooldown
        cooldown_key = f"{wanted_id}:{camera_id}"
        last_alert = self._cooldown_map.get(cooldown_key, 0)
        if time.time() - last_alert < self.cooldown_seconds:
            return None

        # Create alert
        alert = AlertEvent(
            alert_id=f"A-{uuid.uuid4().hex[:8].upper()}",
            wanted_id=wanted_id,
            wanted_name=name,
            alert_level=alert_level,
            camera_id=camera_id,
            floor=floor,
            similarity_score=cos_sim,
            bbox=bbox,
            timestamp=time.time(),
            wanted_description=wanted_person.description if wanted_person else "",
        )

        self.active_alerts[alert.alert_id] = alert
        self._cooldown_map[cooldown_key] = time.time()

        logger.warning(
            f"🚨 ALERT: Wanted person '{name}' detected! "
            f"Camera={camera_id}, Floor={floor}, Similarity={cos_sim:.3f}"
        )

        return alert

    # ─── Alert Management ─────────────────────────────

    def get_active_alerts(self) -> List[AlertEvent]:
        """Get all active (unresolved) alerts."""
        return [
            a for a in self.active_alerts.values()
            if a.status == AlertStatus.ACTIVE.value
        ]

    def get_alert_history(self, limit: int = 100) -> List[AlertEvent]:
        """Get alert history."""
        all_alerts = list(self.active_alerts.values()) + self.alert_history
        all_alerts.sort(key=lambda a: a.timestamp, reverse=True)
        return all_alerts[:limit]

    def acknowledge_alert(
        self, alert_id: str, acknowledged_by: str = "admin"
    ) -> Optional[AlertEvent]:
        """Acknowledge an alert."""
        alert = self.active_alerts.get(alert_id)
        if alert is None:
            return None

        alert.status = AlertStatus.ACKNOWLEDGED.value
        alert.acknowledged_by = acknowledged_by
        alert.acknowledged_at = time.time()

        logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
        return alert

    def resolve_alert(self, alert_id: str) -> Optional[AlertEvent]:
        """Resolve and archive an alert."""
        alert = self.active_alerts.pop(alert_id, None)
        if alert is None:
            return None

        alert.status = AlertStatus.RESOLVED.value
        self.alert_history.append(alert)

        # Trim history
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]

        logger.info(f"Alert resolved: {alert_id}")
        return alert

    def get_statistics(self) -> Dict:
        """Get alert system statistics."""
        active_count = len(self.get_active_alerts())
        wanted_count = len(self.get_wanted_list())
        return {
            "wanted_persons_count": wanted_count,
            "active_wanted": wanted_count,
            "active_alerts": active_count,
            "total_alerts_history": len(self.alert_history),
            "similarity_threshold": self.similarity_threshold,
            "cooldown_seconds": self.cooldown_seconds,
        }


# ─── Singleton ────────────────────────────────────

_alert_service: Optional[AlertService] = None


def get_alert_service() -> AlertService:
    """Get or create alert service singleton."""
    global _alert_service
    if _alert_service is None:
        from backend.infrastructure.config import settings
        _alert_service = AlertService(
            similarity_threshold=settings.alert_similarity_threshold,
            cooldown_seconds=settings.alert_cooldown_seconds,
        )
    return _alert_service
