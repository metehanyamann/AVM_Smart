"""
Milvus Vector Database Client
Handles all vector operations: insert, search, delete
Collections: face_embeddings_512 (512D) + wanted_faces (512D)
"""

import logging
from typing import List, Tuple, Optional
import numpy as np
import time

try:
    from pymilvus import (
        connections,
        Collection,
        CollectionSchema,
        FieldSchema,
        DataType,
        utility,
    )
    PYMILVUS_AVAILABLE = True
except ImportError:
    PYMILVUS_AVAILABLE = False

logger = logging.getLogger(__name__)


class MilvusClient:
    """
    Milvus Vector Database Client
    Manages face embeddings in 512D vector space
    """

    def __init__(self, host: str = "localhost", port: int = 19530):
        self.host = host
        self.port = port
        self.default_collection = "face_embeddings_512"
        self.wanted_collection = "wanted_faces"
        self.collections = {}
        self.is_connected = False

    def connect(self):
        """Connect to Milvus server with retries"""
        max_retries = 3
        retry_count = 0
        timeout = 5

        while retry_count < max_retries:
            try:
                connections.connect(
                    alias="default",
                    host=self.host,
                    port=self.port,
                    timeout=timeout
                )
                self.is_connected = True
                logger.info(f"Connected to Milvus at {self.host}:{self.port}")
                return True
            except Exception as e:
                retry_count += 1
                logger.warning(f"Milvus connection attempt {retry_count}/{max_retries} failed: {e}")
                if retry_count < max_retries:
                    time.sleep(2)

        logger.error("Milvus unavailable after retries")
        self.is_connected = False
        return False

    def disconnect(self):
        """Disconnect from Milvus"""
        try:
            connections.disconnect(alias="default")
            self.is_connected = False
            logger.info("Disconnected from Milvus")
            return True
        except Exception as e:
            logger.error(f"Disconnect failed: {str(e)}")
            return False

    def create_collection(self, collection_name: str = None, is_wanted: bool = False):
        """Create or load a collection"""
        target_name = collection_name or (self.wanted_collection if is_wanted else self.default_collection)
        try:
            if not self.is_connected:
                logger.error("Cannot create collection: Not connected to Milvus")
                return False

            if utility.has_collection(target_name):
                logger.info(f"Collection '{target_name}' already exists, loading...")
                collection = Collection(target_name)
                collection.load()
                self.collections[target_name] = collection
            else:
                logger.info(f"Creating collection '{target_name}'...")

                fields = [
                    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512),
                    FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=100),
                    FieldSchema(name="timestamp", dtype=DataType.INT64)
                ]
                
                if is_wanted:
                    fields.extend([
                        FieldSchema(name="wanted_id", dtype=DataType.VARCHAR, max_length=100),
                        FieldSchema(name="alert_level", dtype=DataType.VARCHAR, max_length=50)
                    ])

                schema = CollectionSchema(fields=fields, description=f"Collection {target_name}")
                collection = Collection(name=target_name, schema=schema)
                self.collections[target_name] = collection
                logger.info(f"Collection '{target_name}' created")

                self._create_index(target_name)

            return True

        except Exception as e:
            logger.error(f"Collection creation failed for {target_name}: {str(e)}")
            return False

    def _create_index(self, collection_name: str):
        """Create HNSW index — better recall than IVF_FLAT for small-medium face DBs."""
        try:
            index_params = {
                "metric_type": "L2",
                "index_type": "HNSW",
                "params": {"M": 16, "efConstruction": 200}
            }
            collection = self.collections[collection_name]
            collection.create_index(field_name="embedding", index_params=index_params)
            collection.load()
            logger.info(f"HNSW index created and collection '{collection_name}' loaded")
            return True
        except Exception as e:
            logger.error(f"Index creation failed for {collection_name}: {str(e)}")
            return False

    def insert_vector(
        self,
        embedding: List[float] or np.ndarray,
        name: str,
        timestamp: int,
        collection_name: str = None,
        extra_fields: dict = None
    ) -> Optional[int]:
        """Insert a face embedding into Milvus"""
        target_name = collection_name or self.default_collection
        collection = self.collections.get(target_name)
        if not collection:
            return None

        try:
            if isinstance(embedding, list):
                embedding = np.array(embedding, dtype=np.float32)
            elif isinstance(embedding, np.ndarray):
                embedding = embedding.astype(np.float32)

            data = [
                [embedding.tolist()],
                [name],
                [timestamp]
            ]
            
            if extra_fields and target_name == self.wanted_collection:
                data.append([extra_fields.get("wanted_id", "")])
                data.append([extra_fields.get("alert_level", "HIGH")])

            result = collection.insert(data)
            milvus_id = result.primary_keys[0]
            collection.flush()
            logger.info(f"Vector inserted into {target_name}: ID={milvus_id}, Name={name}")
            return milvus_id
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
        """Search for similar faces in the database"""
        target_name = collection_name or self.default_collection
        collection = self.collections.get(target_name)
        if not collection:
            return []

        try:
            if isinstance(embedding, list):
                embedding = np.array(embedding, dtype=np.float32)
            elif isinstance(embedding, np.ndarray):
                embedding = embedding.astype(np.float32)

            # HNSW uses ef (efSearch). Higher ef = better recall at cost of speed.
            # Use ef=256 for wanted_faces (criminal detection — recall is critical).
            ef = 256 if target_name == self.wanted_collection else 64
            search_params = {"metric_type": "L2", "params": {"ef": ef}}
            out_fields = ["name", "timestamp"]
            if target_name == self.wanted_collection:
                out_fields.extend(["wanted_id", "alert_level"])

            results = collection.search(
                data=[embedding.tolist()],
                anns_field="embedding",
                param=search_params,
                limit=limit,
                output_fields=out_fields
            )

            matches = []
            if results and len(results[0]) > 0:
                for hit in results[0]:
                    distance = hit.distance
                    if distance < threshold:
                        if target_name == self.wanted_collection:
                            matches.append((hit.id, hit.entity.get('name'), distance, hit.entity.get('wanted_id'), hit.entity.get('alert_level')))
                        else:
                            matches.append((hit.id, hit.entity.get('name'), distance))

            return matches
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []

    def get_all_vectors(self, collection_name: str = None) -> List[dict]:
        """Get all vectors from collection"""
        target_name = collection_name or self.default_collection
        collection = self.collections.get(target_name)
        if not collection:
            return []
        try:
            out_fields = ["name", "timestamp"]
            if target_name == self.wanted_collection:
                out_fields.extend(["wanted_id", "alert_level", "embedding"])
            # Use 'id > 0' instead of empty expr — empty expr fails on some Milvus versions
            collection.load()
            results = collection.query(expr="id > 0", output_fields=out_fields)
            return results
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            return []

    def delete_by_id(self, milvus_id: int, collection_name: str = None) -> bool:
        target_name = collection_name or self.default_collection
        collection = self.collections.get(target_name)
        if not collection: return False
        try:
            collection.delete(f"id == {milvus_id}")
            collection.flush()
            return True
        except Exception as e:
            return False

    def delete_by_wanted_id(self, wanted_id: str) -> bool:
        collection = self.collections.get(self.wanted_collection)
        if not collection: return False
        try:
            collection.delete(f'wanted_id == "{wanted_id}"')
            collection.flush()
            return True
        except Exception as e:
            return False

    def delete_by_name(self, name: str, collection_name: str = None) -> bool:
        """Delete all vectors with a specific name"""
        target_name = collection_name or self.default_collection
        collection = self.collections.get(target_name)
        if not collection:
            return False
        try:
            collection.delete(f'name == "{name}"')
            collection.flush()
            logger.info(f"Deleted all vectors with name={name}")
            return True
        except Exception as e:
            logger.error(f"Delete by name failed: {str(e)}")
            return False

    def get_collection_stats(self, collection_name: str = None) -> dict:
        """Get collection statistics"""
        target_name = collection_name or self.default_collection
        collection = self.collections.get(target_name)
        if not collection:
            return {}
        try:
            count = collection.num_entities
            stats = {
                "collection_name": target_name,
                "total_vectors": count,
                "vector_dimension": 512,
                "status": "healthy" if count >= 0 else "error"
            }
            logger.info(f"Collection stats: {stats}")
            return stats
        except Exception as e:
            logger.error(f"Stats retrieval failed: {str(e)}")
            return {}

    def get_name_by_id(self, milvus_id: int, collection_name: str = None) -> Optional[str]:
        """Get person name by Milvus ID"""
        target_name = collection_name or self.default_collection
        collection = self.collections.get(target_name)
        if not collection:
            return None
        try:
            result = collection.query(
                expr=f"id == {milvus_id}",
                output_fields=["name"]
            )
            if result and len(result) > 0:
                name = result[0].get('name')
                logger.info(f"Found name for ID={milvus_id}: {name}")
                return name
            logger.warning(f"No name found for ID={milvus_id}")
            return None
        except Exception as e:
            logger.error(f"Name lookup failed: {str(e)}")
            return None

    def list_all_names(self, collection_name: str = None) -> List[str]:
        """Get list of all unique names in collection"""
        target_name = collection_name or self.default_collection
        collection = self.collections.get(target_name)
        if not collection:
            return []
        try:
            results = collection.query(
                expr="",
                output_fields=["name"]
            )
            names = list(set([r.get('name') for r in results]))
            logger.info(f"Found {len(names)} unique names")
            return sorted(names)
        except Exception as e:
            logger.error(f"Name listing failed: {str(e)}")
            return []


# Singleton instance
_milvus_client = None


def get_milvus_client(host: str = None, port: int = None):
    """Get or create Milvus client."""
    global _milvus_client

    if _milvus_client is None:
        from backend.infrastructure.config import settings
        _host = host or settings.milvus_host
        _port = port or settings.milvus_port

        if PYMILVUS_AVAILABLE:
            _milvus_client = MilvusClient(host=_host, port=_port)
            if _milvus_client.connect():
                _milvus_client.create_collection()
                _milvus_client.create_collection(is_wanted=True)
                logger.info("Using Milvus vector database")
                return _milvus_client

        logger.warning("Milvus unavailable")
        _milvus_client = MilvusClient(host=_host, port=_port)

    return _milvus_client
