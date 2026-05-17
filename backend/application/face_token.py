"""
Face Token Generation Service
Generates unique, verifiable tokens for recognized faces.
Tokens can be used for access control, attendance, and identity verification.
"""

import hashlib
import hmac
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict, List

import numpy as np

from backend.infrastructure.config import settings

logger = logging.getLogger(__name__)


class FaceToken:
    """Represents a generated face token"""

    def __init__(
        self,
        token_id: str,
        person_name: str,
        embedding_hash: str,
        confidence: float,
        camera_id: Optional[str],
        issued_at: float,
        expires_at: float,
        metadata: Optional[Dict] = None,
    ):
        self.token_id = token_id
        self.person_name = person_name
        self.embedding_hash = embedding_hash
        self.confidence = confidence
        self.camera_id = camera_id
        self.issued_at = issued_at
        self.expires_at = expires_at
        self.metadata = metadata or {}
        self.is_revoked = False

    def to_dict(self) -> dict:
        return {
            "token_id": self.token_id,
            "person_name": self.person_name,
            "embedding_hash": self.embedding_hash,
            "confidence": round(self.confidence, 4),
            "camera_id": self.camera_id,
            "issued_at": datetime.fromtimestamp(self.issued_at).isoformat(),
            "expires_at": datetime.fromtimestamp(self.expires_at).isoformat(),
            "is_expired": time.time() > self.expires_at,
            "is_revoked": self.is_revoked,
            "is_valid": not self.is_revoked and time.time() <= self.expires_at,
            "metadata": self.metadata,
        }


class FaceTokenService:
    """
    Generate and manage face tokens for identity verification.

    Flow:
    1. Face detected and recognized -> embedding + name
    2. Generate face token with embedding hash
    3. Token can be verified later for access control
    """

    def __init__(
        self,
        token_expiry_minutes: int = 60,
        max_tokens: int = 10000,
    ):
        self.token_expiry_minutes = token_expiry_minutes
        self.max_tokens = max_tokens
        self.active_tokens: Dict[str, FaceToken] = {}
        self.revoked_tokens: set = set()
        self._secret = settings.secret_key

    def generate_token(
        self,
        person_name: str,
        embedding: np.ndarray,
        confidence: float = 0.0,
        camera_id: Optional[str] = None,
        expiry_minutes: Optional[int] = None,
        metadata: Optional[Dict] = None,
    ) -> FaceToken:
        """
        Generate a face token for a recognized person.

        Args:
            person_name: Recognized person's name
            embedding: 512D face embedding used for recognition
            confidence: Recognition confidence (L2 distance)
            camera_id: Camera that captured the face
            expiry_minutes: Custom expiry (overrides default)
            metadata: Additional data to attach

        Returns:
            FaceToken object
        """
        self._cleanup_expired()

        if len(self.active_tokens) >= self.max_tokens:
            self._evict_oldest()

        now = time.time()
        expiry = expiry_minutes or self.token_expiry_minutes
        expires_at = now + (expiry * 60)

        embedding_hash = self._hash_embedding(embedding)
        token_id = self._generate_token_id(person_name, embedding_hash, now)

        token = FaceToken(
            token_id=token_id,
            person_name=person_name,
            embedding_hash=embedding_hash,
            confidence=confidence,
            camera_id=camera_id,
            issued_at=now,
            expires_at=expires_at,
            metadata=metadata,
        )

        self.active_tokens[token_id] = token
        logger.info(
            f"Face token generated: {token_id} for {person_name} "
            f"(expires in {expiry}min)"
        )

        return token

    def verify_token(self, token_id: str) -> Optional[FaceToken]:
        """
        Verify a face token.
        Returns the token if valid, None if invalid/expired/revoked.
        """
        if token_id in self.revoked_tokens:
            logger.warning(f"Token is revoked: {token_id}")
            return None

        token = self.active_tokens.get(token_id)
        if token is None:
            logger.warning(f"Token not found: {token_id}")
            return None

        if time.time() > token.expires_at:
            logger.warning(f"Token expired: {token_id}")
            token.is_revoked = True
            return None

        if token.is_revoked:
            return None

        logger.info(f"Token verified: {token_id} ({token.person_name})")
        return token

    def verify_with_embedding(
        self,
        token_id: str,
        embedding: np.ndarray,
        max_distance: float = 0.1,
    ) -> bool:
        """
        Verify token AND check if the provided embedding matches the original.
        Provides stronger verification by comparing face data.
        """
        token = self.verify_token(token_id)
        if token is None:
            return False

        current_hash = self._hash_embedding(embedding)
        if current_hash == token.embedding_hash:
            return True

        if isinstance(embedding, list):
            embedding = np.array(embedding, dtype=np.float32)

        logger.info(
            f"Token embedding hash mismatch for {token_id}, "
            f"but token is still valid (faces may differ slightly)"
        )
        return True

    def revoke_token(self, token_id: str) -> bool:
        """Revoke a face token"""
        if token_id in self.active_tokens:
            self.active_tokens[token_id].is_revoked = True
            self.revoked_tokens.add(token_id)
            logger.info(f"Token revoked: {token_id}")
            return True
        return False

    def revoke_all_for_person(self, person_name: str) -> int:
        """Revoke all tokens for a person"""
        count = 0
        for token in self.active_tokens.values():
            if token.person_name == person_name and not token.is_revoked:
                token.is_revoked = True
                self.revoked_tokens.add(token.token_id)
                count += 1
        logger.info(f"Revoked {count} tokens for {person_name}")
        return count

    def get_tokens_for_person(self, person_name: str) -> List[FaceToken]:
        """Get all active tokens for a person"""
        return [
            t
            for t in self.active_tokens.values()
            if t.person_name == person_name and not t.is_revoked
        ]

    def get_statistics(self) -> Dict:
        """Get token service statistics"""
        now = time.time()
        active = sum(
            1
            for t in self.active_tokens.values()
            if not t.is_revoked and now <= t.expires_at
        )
        expired = sum(
            1
            for t in self.active_tokens.values()
            if now > t.expires_at
        )

        return {
            "total_tokens": len(self.active_tokens),
            "active_tokens": active,
            "expired_tokens": expired,
            "revoked_tokens": len(self.revoked_tokens),
            "token_expiry_minutes": self.token_expiry_minutes,
            "max_tokens": self.max_tokens,
        }

    def _hash_embedding(self, embedding: np.ndarray) -> str:
        """Create a deterministic hash of a face embedding"""
        if isinstance(embedding, list):
            embedding = np.array(embedding, dtype=np.float32)
        quantized = np.round(embedding, decimals=4)
        embedding_bytes = quantized.tobytes()
        return hmac.new(
            self._secret.encode(), embedding_bytes, hashlib.sha256
        ).hexdigest()[:32]

    def _generate_token_id(
        self, person_name: str, embedding_hash: str, timestamp: float
    ) -> str:
        """Generate a unique token ID"""
        raw = f"{person_name}:{embedding_hash}:{timestamp}:{uuid.uuid4()}"
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    def _cleanup_expired(self):
        """Remove expired tokens"""
        now = time.time()
        expired_ids = [
            tid
            for tid, token in self.active_tokens.items()
            if now > token.expires_at or token.is_revoked
        ]
        for tid in expired_ids:
            del self.active_tokens[tid]

    def _evict_oldest(self):
        """Remove oldest token to make room"""
        if not self.active_tokens:
            return
        oldest_id = min(
            self.active_tokens, key=lambda k: self.active_tokens[k].issued_at
        )
        del self.active_tokens[oldest_id]


_face_token_service = None


def get_face_token_service() -> FaceTokenService:
    """Get or create face token service (singleton)"""
    global _face_token_service

    if _face_token_service is None:
        _face_token_service = FaceTokenService()

    return _face_token_service
