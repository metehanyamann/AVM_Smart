"""
Face Search & Matching Service
Searches for similar faces and performs matching
"""

import logging
import numpy as np
from typing import List, Tuple, Optional
from backend.infrastructure.milvus_client import get_milvus_client

logger = logging.getLogger(__name__)


class FaceSearchService:
    """
    Search and match faces using Milvus vector database
    Uses L2 distance (Euclidean) for similarity matching
    """

    def __init__(self, threshold: float = None):
        """
        Initialize search service

        Args:
            threshold: L2 distance threshold for matching
                      < threshold = same person (match)
                      >= threshold = different person (no match)

        Recommended thresholds (L2 on L2-normalized 512D embeddings):
            ArcFace:        ~1.0  (high discriminative power)
            Histogram+LBP:  ~0.55 (lower discriminative power)
        """
        self.milvus_client = get_milvus_client()

        if threshold is not None:
            self.threshold = threshold
        else:
            self.threshold = self._auto_threshold()
    
    def search_face(
        self,
        embedding: np.ndarray,
        top_k: int = 3,
        threshold: Optional[float] = None
    ) -> List[Tuple[int, str, float]]:
        """
        Search for similar faces in database
        
        Args:
            embedding: Query 512D face embedding
            top_k: Number of top results to return
            threshold: L2 distance threshold (overrides default)
            
        Returns:
            List of (milvus_id, name, distance) tuples
            Sorted by distance (smallest first = best match)
        """
        try:
            if threshold is None:
                threshold = self.threshold
            
            # Validate embedding
            if embedding is None or len(embedding) != 512:
                logger.error(f"❌ Invalid embedding dimension: {len(embedding) if embedding is not None else 0}")
                return []
            
            # Normalize embedding
            embedding = self._normalize_vector(embedding)
            
            # Search in Milvus
            results = self.milvus_client.search_vector(
                embedding=embedding,
                limit=top_k,
                threshold=threshold
            )
            
            if not results:
                logger.info("⚠️ No matches found in database")
                return []
            
            logger.info(f"✅ Found {len(results)} match(es)")
            return results
            
        except Exception as e:
            logger.error(f"❌ Search failed: {str(e)}")
            return []
    
    def identify_face(
        self,
        embedding: np.ndarray,
        confidence_threshold: float = 0.05
    ) -> Tuple[Optional[str], float, bool]:
        """
        Identify a single face with confidence level
        
        Returns most likely match if confident enough
        
        Logic:
        1. Get top 3 matches
        2. Check primary: if distance < main_threshold → MATCH
        3. Check secondary: if gap between 1st and 2nd < confidence_threshold
           → HIGH CONFIDENCE, otherwise MEDIUM CONFIDENCE
        
        Args:
            embedding: Query 512D face embedding
            confidence_threshold: Gap threshold between 1st and 2nd match
                                 (for confidence assessment)
        
        Returns:
            Tuple of (name, distance, is_confident)
            - name: Person name or None if no match
            - distance: L2 distance to best match
            - is_confident: True if high confidence, False if ambiguous
        """
        try:
            # Get top 3 matches
            matches = self.search_face(embedding, top_k=3)
            
            if not matches:
                logger.info("ℹ️ No matches found")
                return None, float('inf'), False
            
            # Best match
            best_id, best_name, best_distance = matches[0]
            
            # Check if match passes threshold
            if best_distance >= self.threshold:
                logger.info(f"❌ Best match rejected (distance={best_distance:.4f} >= {self.threshold})")
                return None, best_distance, False
            
            # Confidence assessment
            is_confident = True
            
            if len(matches) > 1:
                second_distance = matches[1][2]
                confidence_gap = second_distance - best_distance
                
                if confidence_gap < confidence_threshold:
                    logger.warning(f"⚠️ Low confidence: gap only {confidence_gap:.4f}")
                    is_confident = False
                else:
                    logger.info(f"✅ High confidence: gap {confidence_gap:.4f}")
            
            logger.info(f"✅ Identified: {best_name} (distance={best_distance:.4f}, confident={is_confident})")
            return best_name, best_distance, is_confident
            
        except Exception as e:
            logger.error(f"❌ Identification failed: {str(e)}")
            return None, float('inf'), False
    
    def batch_search(
        self,
        embeddings: List[np.ndarray],
        top_k: int = 3
    ) -> List[List[Tuple[int, str, float]]]:
        """
        Search multiple embeddings
        
        Args:
            embeddings: List of 512D embeddings
            top_k: Top results per query
            
        Returns:
            List of match lists (one per embedding)
        """
        try:
            results = []
            for embedding in embeddings:
                matches = self.search_face(embedding, top_k=top_k)
                results.append(matches)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Batch search failed: {str(e)}")
            return []
    
    def _auto_threshold(self) -> float:
        """Pick threshold based on which feature extraction model is active."""
        try:
            from backend.application.feature_extraction import get_feature_service

            svc = get_feature_service()
            if svc.model_type == "arcface":
                logger.info("Auto-threshold: ArcFace active -> threshold=1.0")
                return 1.0
            logger.info("Auto-threshold: Histogram+LBP active -> threshold=0.55")
            return 0.55
        except Exception:
            return 0.55

    def set_threshold(self, threshold: float):
        """
        Set L2 distance threshold for matching

        Args:
            threshold: New threshold value (ArcFace: ~1.0, Histogram+LBP: ~0.55)
        """
        if 0.0 <= threshold <= 2.0:
            self.threshold = threshold
            logger.info(f"Threshold set to {threshold}")
        else:
            logger.error(f"Invalid threshold: {threshold} (must be 0-2)")
    
    def get_threshold(self) -> float:
        """Get current threshold"""
        return self.threshold
    
    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """
        L2 normalize vector
        
        Args:
            vector: Input vector
            
        Returns:
            Normalized vector
        """
        try:
            norm = np.linalg.norm(vector)
            if norm == 0:
                logger.warning("⚠️ Vector norm is 0")
                return vector
            
            return (vector / (norm + 1e-10)).astype(np.float32)
            
        except Exception as e:
            logger.error(f"❌ Normalization failed: {str(e)}")
            return vector
    
    def get_search_config(self) -> dict:
        """
        Get search configuration
        
        Returns:
            Dictionary with search settings
        """
        return {
            "metric_type": "L2 (Euclidean)",
            "threshold": self.threshold,
            "embedding_dim": 512,
            "threshold_interpretation": {
                "below": f"< {self.threshold} = MATCH (same person)",
                "above": f">= {self.threshold} = NO MATCH (different person)"
            }
        }


# Singleton instance
_search_service = None


def get_search_service(threshold: float = None) -> FaceSearchService:
    """
    Get or create face search service (singleton).
    Threshold is auto-detected from the active feature extraction model.
    """
    global _search_service

    if _search_service is None:
        _search_service = FaceSearchService(threshold=threshold)

    return _search_service
