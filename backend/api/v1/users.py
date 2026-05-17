"""
User Management Endpoints
Register, list, update, and delete users
"""

from fastapi import APIRouter, HTTPException, status, Body, Depends
from typing import List
from datetime import datetime
import time

from backend.api.v1.schemas import (
    UserRegistrationRequest,
    UserResponse,
    UserListResponse,
    UserDeleteResponse
)
from backend.api.v1.auth import get_current_user, require_role
import logging
import numpy as np

router = APIRouter()
logger = logging.getLogger(__name__)


from backend.application.user_service import get_user_service

def _get_user_svc():
    return get_user_service()


# Mock database (will be replaced with actual database)
mock_users_db = {
    1: {
        "user_id": 1,
        "name": "Merco",
        "embeddings_count": 5,
        "created_at": datetime(2024, 1, 1, 10, 0, 0),
        "updated_at": datetime(2024, 1, 5, 15, 30, 0),
        "metadata": {"department": "R&D", "role": "Engineer"}
    }
}

user_counter = 2


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register New User",
    description="Register a new user and optionally associate a face embedding",
    responses={
        201: {"description": "User registered successfully"},
        400: {"description": "Invalid user data"},
        409: {"description": "User already exists"},
        422: {"description": "Validation error"}
    }
)
async def register_user(
    request: UserRegistrationRequest = Body(...),
    auth_user: dict = Depends(require_role("operator")),
):
    """
    Register a new user
    
    **Request Body:**
    ```json
    {
        "name": "John Doe",
        "embedding": [0.123, -0.456, ...],  // Optional 512D vector
        "metadata": {"department": "R&D"}   // Optional custom data
    }
    ```
    
    **Returns:**
    - User ID
    - Registration timestamp
    - User details
    """
    try:
        logger.info(f"Registering new user: {request.name}")
        
        # Check if user already exists
        for user in mock_users_db.values():
            if user["name"].lower() == request.name.lower():
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"User '{request.name}' already exists"
                )
        
        # Create new user
        global user_counter
        user_id = user_counter
        user_counter += 1
        
        now = datetime.utcnow()
        
        new_user = UserResponse(
            user_id=user_id,
            name=request.name,
            embeddings_count=1 if request.embedding else 0,
            created_at=now,
            updated_at=now,
            metadata=request.metadata
        )
        
        # Store in mock database
        mock_users_db[user_id] = {
            "user_id": user_id,
            "name": request.name,
            "embeddings_count": 1 if request.embedding else 0,
            "created_at": now,
            "updated_at": now,
            "metadata": request.metadata
        }
        
        # If embedding provided, save to Milvus
        if request.embedding:
            if len(request.embedding) != 512:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Embedding must be 512-dimensional"
                )
            logger.info(f"Saving embedding for user: {request.name}")
        
        logger.info(f"User registered successfully: {request.name} (ID: {user_id})")
        
        return new_user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"User registration error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {str(e)}"
        )


@router.delete(
    "/clear-all",
    status_code=status.HTTP_200_OK,
    summary="Clear All Face Data",
    description="Delete all registered faces from the database",
    responses={
        200: {"description": "Database cleared successfully"},
        500: {"description": "Failed to clear database"}
    }
)
async def clear_all_faces(auth_user: dict = Depends(require_role("admin"))):
    """
    Delete all face embeddings from the database (irreversible!)
    
    **Returns:**
    - Number of deleted faces
    - Confirmation message
    """
    try:
        logger.info("🗑️ Clearing all face data from database...")
        
        milvus_client = _get_user_svc().milvus_client
        
        # Get all vectors
        all_vectors = milvus_client.get_all_vectors()
        
        if not all_vectors:
            logger.info("✅ Database already empty")
        
        # Delete each vector
        deleted_count = 0
        for vec in all_vectors:
            if milvus_client.delete_by_id(vec['id']):
                deleted_count += 1
        
        # Also clear tracker in-memory cache so deleted people are no longer recognized
        try:
            from backend.application.tracking import get_tracker
            tracker = get_tracker()
            tracker.global_identities.clear()
            tracker.clear_all()
            logger.info("✅ Tracker memory cleared (global_identities + active tracks)")
        except Exception as te:
            logger.warning(f"Tracker clear skipped: {te}")
        
        logger.info(f"✅ Cleared {deleted_count} faces from database")
        
        return {
            "success": True,
            "deleted_count": deleted_count,
            "message": f"Successfully deleted {deleted_count} registered faces"
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to clear database: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear database: {str(e)}"
        )


@router.get(
    "/list",
    response_model=UserListResponse,
    status_code=status.HTTP_200_OK,
    summary="List All Users",
    description="Retrieve list of all registered users",
    responses={
        200: {"description": "Users retrieved successfully"},
        422: {"description": "Validation error"}
    }
)
async def list_users(
    skip: int = 0,
    limit: int = 100,
    auth_user: dict = Depends(require_role("viewer")),
):
    """
    Get list of all registered users
    
    **Parameters:**
    - **skip**: Number of users to skip (pagination)
    - **limit**: Maximum number of users to return (1-100)
    
    **Returns:**
    - List of users with metadata
    - Total user count
    """
    try:
        if limit < 1 or limit > 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="limit must be between 1 and 100"
            )
        
        if skip < 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="skip must be non-negative"
            )
        
        # Get all users
        all_users = list(mock_users_db.values())
        
        # Apply pagination
        paginated_users = all_users[skip:skip + limit]
        
        # Convert to response models
        user_responses = [
            UserResponse(**user) for user in paginated_users
        ]
        
        logger.info(f"Listing users: skip={skip}, limit={limit}, total={len(all_users)}")
        
        return UserListResponse(
            success=True,
            users=user_responses,
            total=len(all_users)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"User list error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve users: {str(e)}"
        )


@router.get(
    "/{user_id}",
    response_model=UserResponse,
    status_code=status.HTTP_200_OK,
    summary="Get User Details",
    description="Retrieve detailed information about a specific user",
    responses={
        200: {"description": "User retrieved successfully"},
        404: {"description": "User not found"},
        422: {"description": "Validation error"}
    }
)
async def get_user(user_id: int, auth_user: dict = Depends(require_role("viewer"))):
    """
    Get specific user details
    
    **Parameters:**
    - **user_id**: ID of the user to retrieve
    
    **Returns:**
    - User information
    - Embedding count
    - Metadata
    """
    try:
        if user_id not in mock_users_db:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with ID {user_id} not found"
            )
        
        user_data = mock_users_db[user_id]
        
        logger.info(f"Retrieved user: {user_data['name']} (ID: {user_id})")
        
        return UserResponse(**user_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get user error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve user: {str(e)}"
        )


@router.delete(
    "/name/{user_name}",
    status_code=status.HTTP_200_OK,
    summary="Delete User by Name",
    description="Delete a user and all associated face embeddings by name",
    responses={
        200: {"description": "User deleted successfully"},
        500: {"description": "Validation error"}
    }
)
async def delete_user_by_name(user_name: str, auth_user: dict = Depends(require_role("operator"))):
    try:
        milvus_client = _get_user_svc().milvus_client
        
        # Delete from Milvus
        success = milvus_client.delete_by_name(user_name)
        if not success:
            logger.warning(f"Failed to delete {user_name} from Milvus or user not found.")
            
        # Delete from Tracker in-memory cache
        try:
            from backend.application.tracking import get_tracker
            tracker = get_tracker()
            if user_name in tracker.global_identities:
                del tracker.global_identities[user_name]
            
            # Also clear from active tracks
            for track_id, track in list(tracker.active_tracks.items()):
                if track.name == user_name:
                    del tracker.active_tracks[track_id]
            logger.info(f"✅ Cleared {user_name} from tracker memory")
        except Exception as te:
            logger.warning(f"Tracker clear skipped: {te}")
            
        return {
            "success": True,
            "message": f"User '{user_name}' deleted."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"User deletion error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete user: {str(e)}"
        )


@router.get(
    "/{user_id}/faces",
    tags=["Users"],
    status_code=status.HTTP_200_OK,
    summary="Get User's Face Embeddings",
    description="Retrieve all face embeddings associated with a user",
    responses={
        200: {"description": "Embeddings retrieved successfully"},
        404: {"description": "User not found"},
        422: {"description": "Validation error"}
    }
)
async def get_user_faces(user_id: int, auth_user: dict = Depends(require_role("viewer"))):
    """
    Get all face embeddings for a user
    
    **Parameters:**
    - **user_id**: ID of the user
    
    **Returns:**
    - List of embeddings with metadata
    - Total embedding count
    """
    try:
        if user_id not in mock_users_db:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with ID {user_id} not found"
            )
        
        user_data = mock_users_db[user_id]
        
        logger.info(f"Retrieving faces for user: {user_data['name']}")
        
        return {
            "success": True,
            "user_id": user_id,
            "user_name": user_data["name"],
            "embeddings": [
                {
                    "milvus_id": i,
                    "registered_at": datetime.utcnow().isoformat(),
                    "quality_score": 0.95
                }
                for i in range(user_data["embeddings_count"])
            ],
            "total_embeddings": user_data["embeddings_count"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get user faces error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve faces: {str(e)}"
        )


@router.post(
    "/register-face",
    status_code=status.HTTP_201_CREATED,
    summary="Register Face Embedding",
    description="Register a face embedding for a user (save to Milvus database)",
    responses={
        201: {"description": "Face registered successfully"},
        400: {"description": "Invalid embedding"},
        422: {"description": "Validation error"}
    }
)
async def register_face(
    request_body: dict = Body(...),
    auth_user: dict = Depends(require_role("operator")),
):
    """
    Register a face embedding for a user in the Milvus database
    
    **Request Body:**
    ```json
    {
        "name": "John Doe",
        "embedding": [0.123, -0.456, ...]  // 512D vector
    }
    ```
    
    **Returns:**
    - Milvus ID
    - Embedding norm
    - Confirmation message
    """
    try:
        name = request_body.get('name')
        embedding = request_body.get('embedding')
        
        if not name or not embedding:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Both 'name' and 'embedding' are required"
            )
        
        if not isinstance(embedding, list) or len(embedding) != 512:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Embedding must be a 512D list, got {len(embedding) if isinstance(embedding, list) else 'non-list'}"
            )
        
        logger.info(f"📤 Registering face for: {name}")
        
        # Insert embedding to Milvus via user service
        milvus_id = _get_user_svc().insert_face_embedding(
            name=name,
            embedding=embedding
        )
        
        if milvus_id is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to register face for {name}"
            )
        
        # Calculate norm for response
        emb_array = np.array(embedding, dtype=np.float32)
        norm = float(np.linalg.norm(emb_array))
        
        logger.info(f"✅ Face registered successfully: {name} (ID={milvus_id}, norm={norm:.4f})")
        
        return {
            "success": True,
            "milvus_id": milvus_id,
            "name": name,
            "embedding_norm": norm,
            "message": f"Face embedding registered for {name}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Face registration error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to register face: {str(e)}"
        )



