import logging
from typing import List, Tuple, Optional
import numpy as np

from sqlalchemy import select, delete, func
from backend.infrastructure.database import SessionLocal, FaceEmbedding, WantedFace

logger = logging.getLogger(__name__)


class PgVectorClient:
    """
    PostgreSQL Vector Database Client using pgvector
    Mimics MilvusClient interface to minimize changes across the application
    """

    def __init__(self):
        self.default_collection = "face_embeddings_512"
        self.wanted_collection = "wanted_faces"
        self.is_connected = True  # Assuming DB is connected via SQLAlchemy

    def insert_vector(
        self,
        embedding: List[float] or np.ndarray,
        name: str,
        timestamp: int,
        collection_name: str = None,
        extra_fields: dict = None
    ) -> Optional[int]:
        target_name = collection_name or self.default_collection
        try:
            db = SessionLocal()
            
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
                
            if target_name == self.wanted_collection:
                wanted_id = extra_fields.get("wanted_id", "") if extra_fields else ""
                alert_level = extra_fields.get("alert_level", "HIGH") if extra_fields else "HIGH"
                new_vec = WantedFace(
                    name=name,
                    embedding=embedding,
                    timestamp=timestamp,
                    wanted_id=wanted_id,
                    alert_level=alert_level
                )
            else:
                new_vec = FaceEmbedding(
                    name=name,
                    embedding=embedding,
                    timestamp=timestamp
                )
                
            db.add(new_vec)
            db.commit()
            db.refresh(new_vec)
            vec_id = new_vec.id
            db.close()
            
            logger.info(f"Vector inserted into {target_name}: ID={vec_id}, Name={name}")
            return vec_id
        except Exception as e:
            logger.error(f"Insert failed: {str(e)}")
            return None

    def search_vector(
        self,
        embedding: List[float] or np.ndarray,
        limit: int = 3,
        threshold: float = 0.3,
        collection_name: str = None
    ) -> List[Tuple]:
        target_name = collection_name or self.default_collection
        try:
            db = SessionLocal()
            
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
                
            matches = []
            
            if target_name == self.wanted_collection:
                results = db.query(
                    WantedFace,
                    WantedFace.embedding.l2_distance(embedding).label('distance')
                ).filter(
                    WantedFace.embedding.l2_distance(embedding) < threshold
                ).order_by('distance').limit(limit).all()
                
                for hit, dist in results:
                    matches.append((hit.id, hit.name, dist, hit.wanted_id, hit.alert_level))
            else:
                results = db.query(
                    FaceEmbedding,
                    FaceEmbedding.embedding.l2_distance(embedding).label('distance')
                ).filter(
                    FaceEmbedding.embedding.l2_distance(embedding) < threshold
                ).order_by('distance').limit(limit).all()
                
                for hit, dist in results:
                    matches.append((hit.id, hit.name, dist))
                    
            db.close()
            return matches
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []

    def get_all_vectors(self, collection_name: str = None) -> List[dict]:
        target_name = collection_name or self.default_collection
        try:
            db = SessionLocal()
            results = []
            
            if target_name == self.wanted_collection:
                records = db.query(WantedFace).all()
                for r in records:
                    results.append({
                        "id": r.id,
                        "name": r.name,
                        "timestamp": r.timestamp,
                        "wanted_id": r.wanted_id,
                        "alert_level": r.alert_level,
                        "embedding": r.embedding
                    })
            else:
                records = db.query(FaceEmbedding).all()
                for r in records:
                    results.append({
                        "id": r.id,
                        "name": r.name,
                        "timestamp": r.timestamp,
                        "embedding": r.embedding
                    })
                    
            db.close()
            return results
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            return []

    def delete_by_id(self, vec_id: int, collection_name: str = None) -> bool:
        target_name = collection_name or self.default_collection
        try:
            db = SessionLocal()
            if target_name == self.wanted_collection:
                db.query(WantedFace).filter(WantedFace.id == vec_id).delete()
            else:
                db.query(FaceEmbedding).filter(FaceEmbedding.id == vec_id).delete()
            db.commit()
            db.close()
            return True
        except Exception as e:
            logger.error(f"Delete by ID failed: {str(e)}")
            return False

    def delete_by_wanted_id(self, wanted_id: str) -> bool:
        try:
            db = SessionLocal()
            db.query(WantedFace).filter(WantedFace.wanted_id == wanted_id).delete()
            db.commit()
            db.close()
            return True
        except Exception as e:
            logger.error(f"Delete by wanted_id failed: {str(e)}")
            return False

    def delete_by_name(self, name: str) -> bool:
        try:
            db = SessionLocal()
            db.query(FaceEmbedding).filter(FaceEmbedding.name == name).delete()
            db.query(WantedFace).filter(WantedFace.name == name).delete()
            db.commit()
            db.close()
            logger.info(f"Deleted all vectors with name={name}")
            return True
        except Exception as e:
            logger.error(f"Delete by name failed: {str(e)}")
            return False

    def get_collection_stats(self) -> dict:
        try:
            db = SessionLocal()
            count = db.query(FaceEmbedding).count()
            db.close()
            
            return {
                "collection_name": self.default_collection,
                "total_vectors": count,
                "vector_dimension": 512,
                "status": "healthy"
            }
        except Exception as e:
            logger.error(f"Stats retrieval failed: {str(e)}")
            return {}

    def get_name_by_id(self, vec_id: int) -> Optional[str]:
        try:
            db = SessionLocal()
            record = db.query(FaceEmbedding).filter(FaceEmbedding.id == vec_id).first()
            db.close()
            if record:
                return record.name
            return None
        except Exception as e:
            logger.error(f"Name lookup failed: {str(e)}")
            return None

    def list_all_names(self) -> List[str]:
        try:
            db = SessionLocal()
            names = db.query(FaceEmbedding.name).distinct().all()
            db.close()
            return sorted([n[0] for n in names if n[0]])
        except Exception as e:
            logger.error(f"Name listing failed: {str(e)}")
            return []

# Singleton instance
_pgvector_client = None

def get_pgvector_client():
    """Get or create PgVector client."""
    global _pgvector_client
    if _pgvector_client is None:
        _pgvector_client = PgVectorClient()
        logger.info("Using PostgreSQL pgvector database")
    return _pgvector_client
