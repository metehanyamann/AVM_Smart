"""
Face Detection Endpoints
Detect faces from uploaded images using Haar Cascade
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, status
from typing import List
import base64
import io
import time
from datetime import datetime
import cv2
import numpy as np

from backend.api.v1.schemas import FaceDetectionRequest, FaceDetectionResponse, ErrorResponse
from backend.application.face_detection import get_detection_service
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


def _get_detection():
    return get_detection_service()


@router.post(
    "/detect",
    response_model=FaceDetectionResponse,
    status_code=status.HTTP_200_OK,
    summary="Detect Faces from Image",
    description="Detect all faces in an uploaded image using Haar Cascade",
    responses={
        200: {"description": "Faces detected successfully"},
        400: {"description": "Invalid image"},
        422: {"description": "Validation error"}
    }
)
async def detect_faces(
    file: UploadFile = File(..., description="Image file (JPG, PNG, etc.)"),
    min_confidence: float = 0.5
):
    """
    Detect faces in uploaded image
    
    **Parameters:**
    - **file**: Image file to process
    - **min_confidence**: Minimum confidence threshold (0.0 - 1.0)
    
    **Returns:**
    - Detected face locations with coordinates
    - Number of faces found
    - Processing time
    """
    try:
        logger.info(f"Processing image: {file.filename}")
        start_time = time.time()
        
        # Read file content
        contents = await file.read()
        
        if len(contents) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File is empty"
            )
        
        # Check file size (max 10MB)
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="File size exceeds 10MB limit"
            )
        
        # Convert bytes to numpy array
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid image format"
            )
        
        # Perform detection using service
        svc = _get_detection()
        faces = svc.detect_faces(frame, min_confidence)
        
        # Format face locations
        face_locations = [
            {"x": int(x), "y": int(y), "width": int(w), "height": int(h)}
            for x, y, w, h in faces
        ]
        
        processing_time = (time.time() - start_time) * 1000
        
        logger.info(f"Detection completed: {len(faces)} faces found in {processing_time:.2f}ms")
        
        return FaceDetectionResponse(
            success=True,
            faces_detected=len(faces),
            face_locations=face_locations,
            processing_time_ms=processing_time,
            model_used=svc.detection_model,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Detection error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Detection failed: {str(e)}"
        )


@router.post(
    "/detect-base64",
    response_model=FaceDetectionResponse,
    status_code=status.HTTP_200_OK,
    summary="Detect Faces from Base64",
    description="Detect faces from a base64-encoded image string",
    responses={
        200: {"description": "Faces detected successfully"},
        400: {"description": "Invalid base64 image"},
        422: {"description": "Validation error"}
    }
)
async def detect_faces_base64(request: FaceDetectionRequest):
    """
    Detect faces from base64-encoded image
    
    **Returns:**
    - Detected face locations with coordinates
    - Number of faces found
    - Processing time
    """
    try:
        start_time = time.time()
        image_b64 = request.image
        min_confidence = request.min_confidence
        
        if not image_b64:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="'image' field is required"
            )
        
        # Decode base64
        try:
            image_bytes = base64.b64decode(image_b64)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid base64 encoding: {str(e)}"
            )
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid image format"
            )
        
        # Perform detection using service
        svc = _get_detection()
        faces = svc.detect_faces(frame, min_confidence)
        
        # Format face locations
        face_locations = [
            {"x": int(x), "y": int(y), "width": int(w), "height": int(h)}
            for x, y, w, h in faces
        ]
        
        processing_time = (time.time() - start_time) * 1000
        
        logger.info(f"Base64 detection completed: {len(faces)} faces found in {processing_time:.2f}ms")
        
        return FaceDetectionResponse(
            success=True,
            faces_detected=len(faces),
            face_locations=face_locations,
            processing_time_ms=processing_time,
            model_used=svc.detection_model,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Base64 detection error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Detection failed: {str(e)}"
        )


@router.get(
    "/status",
    tags=["Detection"],
    summary="Detection Service Status",
    description="Check if face detection service is available"
)
async def detection_status():
    """Check detection service status"""
    svc = _get_detection()
    return {
        "service": "face_detection",
        "status": "operational",
        "active_model": svc.detection_model,
        "model_info": svc.get_model_info(),
        "timestamp": datetime.utcnow().isoformat(),
    }
