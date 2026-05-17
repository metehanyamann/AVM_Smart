"""
Vector Operations Endpoints
Manage embeddings in vector database
"""

from fastapi import APIRouter, HTTPException, status, Body
from typing import List
from datetime import datetime

from backend.api.v1.schemas import (
    VectorInsertRequest,
    VectorInsertResponse,
    VectorStatsResponse
)
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


# Mock vector store (will be replaced with Milvus)
mock_vectors_db = {
    1: {"embedding": [0.1] * 512, "name": "Merco", "timestamp": 1704067200},
    2: {"embedding": [0.2] * 512, "name": "Yunus", "timestamp": 1704067300},
}

vector_counter = 3


@router.post(
    "/insert",
    response_model=VectorInsertResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Insert Face Vector",
    description="Insert a face embedding into the vector database",
    responses={
        201: {"description": "Vector inserted successfully"},
        400: {"description": "Invalid embedding"},
        422: {"description": "Validation error"}
    }
)
async def insert_vector(request: VectorInsertRequest = Body(...)):
    """
    Insert face embedding into vector database
    
    **Request Body:**
    ```json
    {
        "embedding": [0.123, -0.456, ...],  // 512D vector
        "name": "John Doe"
    }
    ```
    
    **Returns:**
    - Milvus assigned ID
    - Insertion confirmation
    """
    try:
        logger.info(f"Inserting vector for: {request.name}")
        
        # Validate embedding
        if len(request.embedding) != 512:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid embedding dimension: {len(request.embedding)}. Expected 512"
            )
        
        # Validate values (should be normalized)
        for val in request.embedding:
            if not isinstance(val, (int, float)):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Embedding must contain only numeric values"
                )
        
        # Generate ID
        global vector_counter
        milvus_id = vector_counter
        vector_counter += 1
        
        # Store in mock database
        mock_vectors_db[milvus_id] = {
            "embedding": request.embedding,
            "name": request.name,
            "timestamp": int(datetime.utcnow().timestamp())
        }
        
        logger.info(f"Vector inserted: ID={milvus_id}, name={request.name}")
        
        return VectorInsertResponse(
            success=True,
            milvus_id=milvus_id,
            message=f"Vector for '{request.name}' inserted successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Vector insertion error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vector insertion failed: {str(e)}"
        )


@router.delete(
    "/{vector_id}",
    tags=["Vectors"],
    status_code=status.HTTP_200_OK,
    summary="Delete Vector",
    description="Delete a face embedding from the vector database",
    responses={
        200: {"description": "Vector deleted successfully"},
        404: {"description": "Vector not found"},
        422: {"description": "Validation error"}
    }
)
async def delete_vector(vector_id: int):
    """
    Delete vector from database
    
    **Parameters:**
    - **vector_id**: Milvus ID of the vector to delete
    
    **Returns:**
    - Deletion confirmation
    """
    try:
        if vector_id not in mock_vectors_db:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Vector with ID {vector_id} not found"
            )
        
        vector_data = mock_vectors_db[vector_id]
        person_name = vector_data["name"]
        
        del mock_vectors_db[vector_id]
        
        logger.info(f"Vector deleted: ID={vector_id}, name={person_name}")
        
        return {
            "success": True,
            "message": f"Vector {vector_id} for '{person_name}' deleted successfully",
            "deleted_id": vector_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Vector deletion error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vector deletion failed: {str(e)}"
        )


@router.get(
    "/{vector_id}",
    tags=["Vectors"],
    status_code=status.HTTP_200_OK,
    summary="Get Vector",
    description="Retrieve a specific vector from the database",
    responses={
        200: {"description": "Vector retrieved successfully"},
        404: {"description": "Vector not found"},
        422: {"description": "Validation error"}
    }
)
async def get_vector(vector_id: int):
    """
    Get vector by ID
    
    **Parameters:**
    - **vector_id**: Milvus ID
    
    **Returns:**
    - Vector metadata (ID, name, timestamp)
    - Note: Embedding data not returned for performance reasons
    """
    try:
        if vector_id not in mock_vectors_db:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Vector with ID {vector_id} not found"
            )
        
        vector_data = mock_vectors_db[vector_id]
        
        return {
            "success": True,
            "milvus_id": vector_id,
            "name": vector_data["name"],
            "timestamp": vector_data["timestamp"],
            "dimension": 512,
            "note": "Embedding not returned. Use search endpoint to get similarity results."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get vector error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve vector: {str(e)}"
        )


@router.get(
    "/stats/database",
    response_model=VectorStatsResponse,
    status_code=status.HTTP_200_OK,
    summary="Database Statistics",
    description="Get statistics about the vector database",
    responses={
        200: {"description": "Statistics retrieved successfully"},
        422: {"description": "Validation error"}
    }
)
async def get_database_stats():
    """
    Get vector database statistics
    
    **Returns:**
    - Total number of vectors
    - Vectors per user
    - Database capacity info
    """
    try:
        # Count vectors per user
        user_counts = {}
        for vector_data in mock_vectors_db.values():
            name = vector_data["name"]
            user_counts[name] = user_counts.get(name, 0) + 1
        
        logger.info(f"Database stats requested: total_vectors={len(mock_vectors_db)}")
        
        return VectorStatsResponse(
            success=True,
            total_vectors=len(mock_vectors_db),
            collection_name="face_embeddings_512",
            vector_dimension=512,
            vector_count_by_user=user_counts
        )
        
    except Exception as e:
        logger.error(f"Statistics error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve statistics: {str(e)}"
        )


@router.get(
    "/stats/collection",
    tags=["Vectors"],
    status_code=status.HTTP_200_OK,
    summary="Collection Statistics",
    description="Get Milvus collection detailed statistics"
)
async def get_collection_stats():
    """
    Get detailed collection statistics
    
    **Returns:**
    - Collection metadata
    - Index information
    - Capacity details
    """
    try:
        return {
            "success": True,
            "collection": {
                "name": "face_embeddings_512",
                "status": "healthy",
                "row_count": len(mock_vectors_db),
                "schema": {
                    "id": {"type": "INT64", "primary": True},
                    "embedding": {"type": "FLOAT_VECTOR", "dim": 512},
                    "name": {"type": "VARCHAR", "max_length": 100},
                    "timestamp": {"type": "INT64"}
                },
                "index": {
                    "field": "embedding",
                    "type": "IVF_FLAT",
                    "metric": "L2",
                    "nlist": 256,
                    "nprobe": 16
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Collection stats error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve collection statistics: {str(e)}"
        )


@router.post(
    "/flush",
    tags=["Vectors"],
    status_code=status.HTTP_200_OK,
    summary="Flush Database",
    description="Commit all pending writes to disk"
)
async def flush_database():
    """
    Flush vector database
    
    Ensures all pending writes are committed to disk
    
    **Returns:**
    - Flush confirmation
    - Timestamp
    """
    try:
        logger.info("Database flush requested")
        
        return {
            "success": True,
            "message": "Database flushed successfully",
            "timestamp": datetime.utcnow().isoformat(),
            "vectors_flushed": len(mock_vectors_db)
        }
        
    except Exception as e:
        logger.error(f"Flush error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Flush failed: {str(e)}"
        )
