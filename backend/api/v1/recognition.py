"""
Face Recognition Endpoints
Feature extraction and face search/matching
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, status, Body
from typing import List
import time
import cv2
import numpy as np
from datetime import datetime

from backend.api.v1.schemas import (
    FeatureExtractionRequest,
    FeatureExtractionResponse,
    FaceSearchRequest,
    FaceSearchResponse,
    FaceSearchResult
)
from backend.application.feature_extraction import get_feature_service
from backend.application.face_search import get_search_service
from backend.application.face_detection import get_detection_service
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/extract-features",
    response_model=FeatureExtractionResponse,
    status_code=status.HTTP_200_OK,
    summary="Extract Face Features",
    description="Extract 512D face embedding from a face image using ArcFace or Histogram+LBP",
    responses={
        200: {"description": "Features extracted successfully"},
        400: {"description": "Invalid image"},
        422: {"description": "Validation error"}
    }
)
async def extract_features(request: FeatureExtractionRequest):
    """
    Extract face features as 512D embedding from base64-encoded image
    
    **Returns:**
    - 512D embedding vector
    - Model used
    - Processing time
    """
    try:
        logger.info(f"Extracting features (model: {request.model})")
        start_time = time.time()
        
        # Decode base64
        import base64
        try:
            image_bytes = base64.b64decode(request.face_roi)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid base64 encoding: {str(e)}"
            )
        
        # Convert bytes to image
        nparr = np.frombuffer(image_bytes, np.uint8)
        face_roi = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if face_roi is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid image format"
            )
        
        det_svc = get_detection_service()
        # Validate face quality
        if not det_svc.is_valid_face(face_roi):
            logger.warning("Face image failed validation (blurry/too small/poor lighting)")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Face image failed quality validation (blurry, too small, or poor lighting)"
            )
        
        # Extract features using service (with automatic fallback)
        feat_svc = get_feature_service()
        embedding, model_used = feat_svc.extract_features(face_roi)
        
        if embedding is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Feature extraction failed"
            )
        
        processing_time = (time.time() - start_time) * 1000
        
        logger.info(f"✅ Features extracted using {model_used} in {processing_time:.2f}ms")
        
        return FeatureExtractionResponse(
            success=True,
            embedding=embedding.tolist(),
            model_used=model_used,
            processing_time_ms=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Feature extraction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Feature extraction failed: {str(e)}"
        )


@router.post(
    "/search",
    response_model=FaceSearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Search Similar Faces",
    description="Search for similar faces in the database using a face embedding",
    responses={
        200: {"description": "Search completed successfully"},
        400: {"description": "Invalid embedding"},
        422: {"description": "Validation error"}
    }
)
async def search_similar_faces(request: FaceSearchRequest = Body(...)):
    """
    Search for similar faces in database
    
    **Request Body:**
    ```json
    {
        "embedding": [0.123, -0.456, ...],  // 512D vector
        "top_k": 3,
        "threshold": 0.3
    }
    ```
    
    **Returns:**
    - List of matching faces (sorted by similarity)
    - Person names and IDs
    - Distance (L2 metric)
    """
    try:
        logger.info(f"Searching for similar faces: top_k={request.top_k}, threshold={request.threshold}")
        start_time = time.time()
        
        # Validate embedding
        if len(request.embedding) != 512:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid embedding dimension: {len(request.embedding)}. Expected 512"
            )
        
        # Validate parameters
        if request.top_k < 1 or request.top_k > 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="top_k must be between 1 and 100"
            )
        
        if request.threshold < 0.0 or request.threshold > 1.0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="threshold must be between 0.0 and 1.0"
            )
        
        # Perform search using service
        embedding_array = np.array(request.embedding, dtype=np.float32)
        matches = get_search_service().search_face(
            embedding=embedding_array,
            top_k=request.top_k,
            threshold=request.threshold
        )
        
        # Format results
        match_results = [
            FaceSearchResult(
                milvus_id=m[0],
                name=m[1],
                distance=float(m[2]),
                timestamp=int(time.time())
            )
            for m in matches
        ]
        
        processing_time = (time.time() - start_time) * 1000
        
        logger.info(f"✅ Search completed: {len(matches)} matches found in {processing_time:.2f}ms")
        
        return FaceSearchResponse(
            success=True,
            matches=match_results,
            processing_time_ms=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@router.post(
    "/identify",
    status_code=status.HTTP_200_OK,
    summary="Identify Face",
    description="Full face identification pipeline: detect → extract → search",
    responses={
        200: {"description": "Face identified successfully"},
        400: {"description": "Invalid image"},
        404: {"description": "No faces found"},
        422: {"description": "Validation error"}
    }
)
async def identify_face(
    file: UploadFile = File(..., description="Image file to identify"),
    threshold: float = 0.3
):
    """
    Full face identification pipeline
    
    **Process:**
    1. Detect faces in image
    2. Extract features for each face
    3. Search in database
    4. Return matches
    
    **Parameters:**
    - **file**: Image file
    - **threshold**: Match threshold (0.0 - 1.0)
    
    **Returns:**
    - List of identified people
    - Confidence scores
    - Processing details
    """
    try:
        logger.info(f"Identifying faces in: {file.filename}")
        
        contents = await file.read()
        
        if len(contents) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File is empty"
            )
        
        # TODO: Implement full pipeline
        # 1. Detect faces
        # 2. Extract features
        # 3. Search database
        # 4. Return results
        
        return {
            "success": True,
            "faces_identified": 2,
            "results": [
                {
                    "person": "Merco",
                    "confidence": 0.95,
                    "milvus_id": 1,
                    "distance": 0.25
                },
                {
                    "person": "Yunus",
                    "confidence": 0.89,
                    "milvus_id": 2,
                    "distance": 0.45
                }
            ],
            "processing_time_ms": 185.4
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Identification error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Identification failed: {str(e)}"
        )


@router.get(
    "/models",
    tags=["Recognition"],
    summary="Available Recognition Models",
    description="Get list of available face recognition models"
)
async def list_models():
    """Get available recognition models"""
    return {
        "available_models": [
            {
                "name": "arcface",
                "type": "Deep Learning",
                "accuracy": "95%+",
                "speed": "Medium",
                "vector_dim": 512,
                "description": "ArcFace ResNet-50 model for face recognition"
            },
            {
                "name": "histogram_lbp",
                "type": "Classical",
                "accuracy": "70-80%",
                "speed": "Fast",
                "vector_dim": 512,
                "description": "Histogram + Local Binary Pattern (fallback)"
            }
        ]
    }


@router.get(
    "/status",
    tags=["Recognition"],
    summary="Recognition Service Status",
    description="Check if face recognition service is available"
)
async def recognition_status():
    """Check recognition service status"""
    return {
        "service": "face_recognition",
        "status": "operational",
        "models": ["arcface", "histogram_lbp"],
        "timestamp": datetime.utcnow().isoformat()
    }
