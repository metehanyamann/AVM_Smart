"""
User Management Service
Handles user registration, deletion, and management
"""

import logging
import time
from typing import List, Optional, Dict
from backend.infrastructure.milvus_client import get_milvus_client

logger = logging.getLogger(__name__)


class UserService:
    """
    Manage user faces and embeddings
    """

    def __init__(self):
        """Initialize user service"""
        self.milvus_client = get_milvus_client()
    
    def register_user(
        self,
        name: str
    ) -> Optional[int]:
        """
        Register new user (without embedding - face must be added separately)
        
        Note: This registers a user entry only. You MUST call insert_face_embedding() 
        to add actual face embeddings for recognition to work.
        
        Args:
            name: User full name
            
        Returns:
            User ID or None if failed
        """
        try:
            if not name or len(name.strip()) == 0:
                logger.error("❌ User name cannot be empty")
                return None
            
            name = name.strip()[:100]  # Max 100 chars
            
            # Create a random normalized embedding instead of zeros
            # This prevents zero-vectors from interfering with search
            import numpy as np
            placeholder_embedding = np.random.randn(512).astype(np.float32)
            norm = np.linalg.norm(placeholder_embedding)
            if norm > 0:
                placeholder_embedding = (placeholder_embedding / (norm + 1e-10)).astype(np.float32)
            
            timestamp = int(time.time())
            
            user_id = self.milvus_client.insert_vector(
                embedding=placeholder_embedding,
                name=name,
                timestamp=timestamp
            )
            
            if user_id is None:
                logger.error(f"❌ Failed to register user: {name}")
                return None
            
            logger.info(f"✅ User registered: {name} (ID={user_id}). ⚠️ Remember to call insert_face_embedding() to add actual faces!")
            return user_id
            
        except Exception as e:
            logger.error(f"❌ Registration failed: {str(e)}")
            return None
    
    def insert_face_embedding(
        self,
        name: str,
        embedding: list or 'np.ndarray'
    ) -> Optional[int]:
        """
        Insert face embedding for a user
        
        Args:
            name: User name
            embedding: 512D face embedding
            
        Returns:
            Milvus ID or None if failed
        """
        try:
            if not name or len(name.strip()) == 0:
                logger.error("❌ User name cannot be empty")
                return None
            
            name = name.strip()[:100]
            timestamp = int(time.time())
            
            # Convert to numpy if needed
            import numpy as np
            if isinstance(embedding, list):
                embedding = np.array(embedding, dtype=np.float32)
            else:
                embedding = np.asarray(embedding, dtype=np.float32)
            
            # L2 Normalize the embedding before saving
            # This ensures consistency with search normalization
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = (embedding / (norm + 1e-10)).astype(np.float32)
                logger.debug(f"📊 Embedding normalized (L2 norm was {norm:.4f})")
            
            milvus_id = self.milvus_client.insert_vector(
                embedding=embedding,
                name=name,
                timestamp=timestamp
            )
            
            if milvus_id is None:
                logger.error(f"❌ Failed to insert embedding for: {name}")
                return None
            
            logger.info(f"✅ Face embedding inserted: {name} (ID={milvus_id}, norm={np.linalg.norm(embedding):.4f})")
            return milvus_id
            
        except Exception as e:
            logger.error(f"❌ Embedding insertion failed: {str(e)}")
            return None
    
    def get_user_by_name(self, name: str) -> Optional[Dict]:
        """
        Get user information by name
        
        Args:
            name: User name
            
        Returns:
            User info dict or None if not found
        """
        try:
            # Get all vectors with this name
            all_vectors = self.milvus_client.get_all_vectors()
            
            for vector in all_vectors:
                if vector.get('name') == name:
                    return {
                        "milvus_id": vector.get('id'),
                        "name": vector.get('name'),
                        "timestamp": vector.get('timestamp')
                    }
            
            logger.warning(f"⚠️ User not found: {name}")
            return None
            
        except Exception as e:
            logger.error(f"❌ User lookup failed: {str(e)}")
            return None
    
    def delete_user(self, name: str) -> bool:
        """
        Delete user and all their face embeddings
        
        Args:
            name: User name to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not name:
                logger.error("❌ User name cannot be empty")
                return False
            
            name = name.strip()
            
            # Delete all vectors with this name
            success = self.milvus_client.delete_by_name(name)
            
            if success:
                logger.info(f"✅ User deleted: {name}")
            else:
                logger.error(f"❌ Failed to delete user: {name}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Deletion failed: {str(e)}")
            return False
    
    def delete_user_by_id(self, milvus_id: int) -> bool:
        """
        Delete specific face embedding by ID
        
        Args:
            milvus_id: Milvus entity ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            success = self.milvus_client.delete_by_id(milvus_id)
            
            if success:
                logger.info(f"✅ Face embedding deleted: ID={milvus_id}")
            else:
                logger.error(f"❌ Failed to delete ID={milvus_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Deletion failed: {str(e)}")
            return False
    
    def get_all_users(self) -> List[str]:
        """
        Get list of all registered users
        
        Returns:
            List of unique user names
        """
        try:
            names = self.milvus_client.list_all_names()
            logger.info(f"✅ Found {len(names)} users")
            return names
            
        except Exception as e:
            logger.error(f"❌ User listing failed: {str(e)}")
            return []
    
    def get_user_face_count(self, name: str) -> int:
        """
        Get number of face embeddings for a user
        
        Args:
            name: User name
            
        Returns:
            Number of faces
        """
        try:
            all_vectors = self.milvus_client.get_all_vectors()
            count = sum(1 for v in all_vectors if v.get('name') == name)
            
            logger.info(f"✅ User '{name}' has {count} face(s)")
            return count
            
        except Exception as e:
            logger.error(f"❌ Count failed: {str(e)}")
            return 0
    
    def get_user_details(self, name: str) -> Optional[Dict]:
        """
        Get detailed information about a user
        
        Args:
            name: User name
            
        Returns:
            User details dict or None
        """
        try:
            face_count = self.get_user_face_count(name)
            user_info = self.get_user_by_name(name)
            
            if user_info is None:
                return None
            
            return {
                "name": name,
                "face_count": face_count,
                "first_registered": user_info.get('timestamp'),
                "milvus_id": user_info.get('milvus_id')
            }
            
        except Exception as e:
            logger.error(f"❌ Details retrieval failed: {str(e)}")
            return None
    
    def list_all_users_with_count(self) -> Dict[str, int]:
        """
        Get all users with their face count
        
        Returns:
            Dictionary of {name: face_count}
        """
        try:
            all_vectors = self.milvus_client.get_all_vectors()
            users = {}
            
            for vector in all_vectors:
                name = vector.get('name')
                users[name] = users.get(name, 0) + 1
            
            logger.info(f"✅ Got info for {len(users)} users")
            return users
            
        except Exception as e:
            logger.error(f"❌ Listing failed: {str(e)}")
            return {}
    
    def get_statistics(self) -> Dict:
        """
        Get user and face statistics
        
        Returns:
            Statistics dictionary
        """
        try:
            all_vectors = self.milvus_client.get_all_vectors()
            users_dict = self.list_all_users_with_count()
            
            return {
                "total_users": len(users_dict),
                "total_faces": len(all_vectors),
                "users": users_dict,
                "average_faces_per_user": len(all_vectors) / len(users_dict) if users_dict else 0
            }
            
        except Exception as e:
            logger.error(f"❌ Statistics failed: {str(e)}")
            return {
                "total_users": 0,
                "total_faces": 0,
                "users": {},
                "average_faces_per_user": 0
            }


# Singleton instance
_user_service = None


def get_user_service() -> UserService:
    """
    Get or create user service (singleton)
    
    Returns:
        UserService instance
    """
    global _user_service
    
    if _user_service is None:
        _user_service = UserService()
    
    return _user_service
