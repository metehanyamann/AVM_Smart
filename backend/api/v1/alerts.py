"""
Wanted Person Alert Endpoints
Manage wanted persons list and view/manage triggered alerts.
"""

from fastapi import APIRouter, HTTPException, status, File, UploadFile, Depends, Body
from typing import Optional
from datetime import datetime
import logging
import base64
import cv2
import numpy as np

from backend.api.v1.auth import require_role
from backend.application.alert_service import get_alert_service
from backend.application.face_detection import get_detection_service
from backend.application.feature_extraction import get_feature_service

router = APIRouter()
logger = logging.getLogger(__name__)


# ─── Wanted Person Management ────────────────────

@router.post(
    "/wanted",
    status_code=status.HTTP_201_CREATED,
    summary="Add Wanted Person",
    description="Upload a photo to add a person to the wanted watch list",
)
async def add_wanted_person(
    file: UploadFile = File(..., description="Photo of the wanted person"),
    name: str = "Unknown",
    description: str = "No description",
    alert_level: str = "HIGH",
    auth_user: dict = Depends(require_role("admin")),
):
    """
    Add a wanted person by uploading their photo.

    The system will:
    1. Detect the face in the photo
    2. Extract 512D embedding
    3. Store in wanted persons list
    4. Begin matching against all tracked faces
    """
    try:
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty file")

        # Decode image
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image format")

        # Detect face
        det_svc = get_detection_service()
        faces = det_svc.detect_faces(frame, min_confidence=0.5)
        if not faces:
            raise HTTPException(status_code=400, detail="No face detected in image")

        # Extract embedding from first face
        x, y, w, h = faces[0]
        face_roi = det_svc.extract_roi(frame, x, y, w, h, padding=10)
        if face_roi is None:
            raise HTTPException(status_code=400, detail="Failed to extract face ROI")

        feat_svc = get_feature_service()
        embedding, model_used = feat_svc.extract_features(face_roi)
        if embedding is None:
            raise HTTPException(status_code=500, detail="Failed to extract embedding")

        # Store photo as base64
        photo_b64 = base64.b64encode(contents).decode("utf-8")

        # Add to alert service
        alert_svc = get_alert_service()
        person = alert_svc.add_wanted_person(
            name=name,
            description=description,
            embedding=embedding,
            alert_level=alert_level,
            photo_base64=photo_b64,
            added_by=auth_user.get("username", "admin"),
        )

        return {
            "success": True,
            "message": f"Wanted person '{name}' added successfully",
            **person.to_dict(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Add wanted person failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/wanted",
    summary="List Wanted Persons",
    description="Get all persons on the wanted watch list",
)
async def list_wanted_persons(
    auth_user: dict = Depends(require_role("viewer")),
):
    """Get the full wanted persons list."""
    alert_svc = get_alert_service()
    persons = alert_svc.get_wanted_list()
    return {
        "success": True,
        "wanted_persons": [p.to_dict() for p in persons],
        "total": len(persons),
    }


@router.delete(
    "/wanted/{wanted_id}",
    summary="Remove Wanted Person",
    description="Remove a person from the wanted watch list",
)
async def remove_wanted_person(
    wanted_id: str,
    auth_user: dict = Depends(require_role("admin")),
):
    """Remove a wanted person by ID."""
    alert_svc = get_alert_service()
    success = alert_svc.remove_wanted_person(wanted_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Wanted person '{wanted_id}' not found")

    return {"success": True, "message": f"Wanted person '{wanted_id}' removed"}





# ─── Active Alerts ────────────────────────────────

@router.get(
    "/active",
    summary="Get Active Alerts",
    description="Get all currently active (unresolved) alerts",
)
async def get_active_alerts(
    auth_user: dict = Depends(require_role("viewer")),
):
    """Get active alerts that haven't been resolved."""
    alert_svc = get_alert_service()
    alerts = alert_svc.get_active_alerts()
    return {
        "success": True,
        "alerts": [a.to_dict() for a in alerts],
        "total": len(alerts),
    }


@router.get(
    "/history",
    summary="Alert History",
    description="Get historical alerts",
)
async def get_alert_history(
    limit: int = 100,
    auth_user: dict = Depends(require_role("viewer")),
):
    """Get alert history."""
    alert_svc = get_alert_service()
    alerts = alert_svc.get_alert_history(limit=limit)
    return {
        "success": True,
        "alerts": [a.to_dict() for a in alerts],
        "total": len(alerts),
    }


@router.post(
    "/acknowledge/{alert_id}",
    summary="Acknowledge Alert",
    description="Mark an alert as acknowledged",
)
async def acknowledge_alert(
    alert_id: str,
    auth_user: dict = Depends(require_role("operator")),
):
    """Acknowledge an alert."""
    alert_svc = get_alert_service()
    alert = alert_svc.acknowledge_alert(
        alert_id=alert_id,
        acknowledged_by=auth_user.get("username", "operator"),
    )
    if alert is None:
        raise HTTPException(status_code=404, detail=f"Alert '{alert_id}' not found")

    return {"success": True, **alert.to_dict()}


@router.post(
    "/resolve/{alert_id}",
    summary="Resolve Alert",
    description="Mark an alert as resolved and archive it",
)
async def resolve_alert(
    alert_id: str,
    auth_user: dict = Depends(require_role("operator")),
):
    """Resolve and archive an alert."""
    alert_svc = get_alert_service()
    alert = alert_svc.resolve_alert(alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail=f"Alert '{alert_id}' not found")

    return {"success": True, **alert.to_dict()}


@router.get(
    "/stats",
    summary="Alert Statistics",
    description="Get alert system statistics",
)
async def get_alert_stats(
    auth_user: dict = Depends(require_role("viewer")),
):
    """Get alert system statistics."""
    alert_svc = get_alert_service()
    stats = alert_svc.get_statistics()
    return {"success": True, **stats}
