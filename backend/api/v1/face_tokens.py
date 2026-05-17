"""
Face Token Generation Endpoints
Generate, verify, and manage face-based identity tokens
"""

from fastapi import APIRouter, HTTPException, status, Body, Depends
from typing import Optional
from datetime import datetime
import numpy as np
import logging

from backend.api.v1.schemas import (
    FaceTokenRequest,
    FaceTokenResponse,
    FaceTokenVerifyResponse,
    FaceTokenStatsResponse,
)
from backend.application.face_token import get_face_token_service
from backend.api.v1.auth import require_role

router = APIRouter()
logger = logging.getLogger(__name__)
token_service = get_face_token_service()


@router.post(
    "/generate",
    response_model=FaceTokenResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Generate Face Token",
    description="Generate a unique identity token for a recognized face",
    responses={
        201: {"description": "Token generated successfully"},
        400: {"description": "Invalid request"},
    },
)
async def generate_face_token(
    request: FaceTokenRequest,
    auth_user: dict = Depends(require_role("operator")),
):
    """
    Generate a face token after successful face recognition.

    **Request:**
    - person_name: Name of the recognized person
    - embedding: 512D face embedding vector
    - confidence: Recognition confidence score
    - camera_id: (optional) Source camera
    - expiry_minutes: (optional) Custom expiry time
    """
    if len(request.embedding) != 512:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Embedding must be 512D, got {len(request.embedding)}D",
        )

    embedding = np.array(request.embedding, dtype=np.float32)

    token = token_service.generate_token(
        person_name=request.person_name,
        embedding=embedding,
        confidence=request.confidence,
        camera_id=request.camera_id,
        expiry_minutes=request.expiry_minutes,
        metadata=request.metadata,
    )

    token_data = token.to_dict()

    return FaceTokenResponse(
        success=True,
        token_id=token_data["token_id"],
        person_name=token_data["person_name"],
        issued_at=token_data["issued_at"],
        expires_at=token_data["expires_at"],
        is_valid=token_data["is_valid"],
    )


@router.get(
    "/verify/{token_id}",
    response_model=FaceTokenVerifyResponse,
    summary="Verify Face Token",
    description="Check if a face token is valid",
    responses={
        200: {"description": "Token verification result"},
    },
)
async def verify_face_token(token_id: str, auth_user: dict = Depends(require_role("viewer"))):
    """Verify a face token by its ID"""
    token = token_service.verify_token(token_id)

    if token is None:
        return FaceTokenVerifyResponse(
            valid=False,
            message="Token is invalid, expired, or revoked",
            token_id=token_id,
        )

    return FaceTokenVerifyResponse(
        valid=True,
        message="Token is valid",
        token_id=token_id,
        person_name=token.person_name,
        issued_at=datetime.fromtimestamp(token.issued_at).isoformat(),
        expires_at=datetime.fromtimestamp(token.expires_at).isoformat(),
    )


@router.post(
    "/revoke/{token_id}",
    summary="Revoke Face Token",
    description="Revoke a specific face token",
)
async def revoke_face_token(token_id: str, auth_user: dict = Depends(require_role("operator"))):
    """Revoke a face token"""
    success = token_service.revoke_token(token_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Token '{token_id}' not found",
        )

    return {"success": True, "message": f"Token '{token_id}' has been revoked"}


@router.post(
    "/revoke/person/{person_name}",
    summary="Revoke All Tokens for Person",
    description="Revoke all face tokens for a specific person",
)
async def revoke_person_tokens(person_name: str, auth_user: dict = Depends(require_role("operator"))):
    """Revoke all tokens for a person"""
    count = token_service.revoke_all_for_person(person_name)
    return {
        "success": True,
        "message": f"Revoked {count} tokens for '{person_name}'",
        "revoked_count": count,
    }


@router.get(
    "/person/{person_name}",
    summary="Get Person Tokens",
    description="Get all active tokens for a specific person",
)
async def get_person_tokens(person_name: str, auth_user: dict = Depends(require_role("viewer"))):
    """Get all active tokens for a person"""
    tokens = token_service.get_tokens_for_person(person_name)
    return {
        "success": True,
        "person_name": person_name,
        "tokens": [t.to_dict() for t in tokens],
        "total": len(tokens),
    }


@router.get(
    "/stats",
    response_model=FaceTokenStatsResponse,
    summary="Token Statistics",
    description="Get face token service statistics",
)
async def get_token_stats(auth_user: dict = Depends(require_role("viewer"))):
    """Get token service statistics"""
    stats = token_service.get_statistics()
    return FaceTokenStatsResponse(
        success=True,
        total_tokens=stats["total_tokens"],
        active_tokens=stats["active_tokens"],
        expired_tokens=stats["expired_tokens"],
        revoked_tokens=stats["revoked_tokens"],
        token_expiry_minutes=stats["token_expiry_minutes"],
    )
