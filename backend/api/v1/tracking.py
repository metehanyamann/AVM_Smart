"""
Real-Time Tracking Endpoints
Multi-camera face tracking with cross-camera re-identification,
floor-based traffic analysis, and comprehensive reporting.
"""

from fastapi import APIRouter, HTTPException, status, Depends
from typing import Optional, List
from datetime import datetime
import logging
from pydantic import BaseModel

from sqlalchemy.orm import Session

from backend.api.v1.schemas import (
    CameraRegisterRequest,
    TrackingUpdateRequest,
    TrackingResponse,
    TrackingStatsResponse,
)
from backend.application.tracking import get_tracker
from backend.api.v1.auth import require_role
from backend.infrastructure.database import get_db, FloorTrafficLog

router = APIRouter()
logger = logging.getLogger(__name__)
tracker = get_tracker()


@router.post(
    "/cameras/register",
    status_code=status.HTTP_201_CREATED,
    summary="Register Camera",
    description="Register a new camera source for real-time tracking with floor assignment",
)
async def register_camera(
    request: CameraRegisterRequest,
    auth_user: dict = Depends(require_role("operator")),
):
    """Register a camera for multi-camera tracking with floor assignment"""
    success = tracker.register_camera(
        camera_id=request.camera_id,
        location=request.location or "",
        floor=request.floor if request.floor is not None else 0,
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Camera '{request.camera_id}' is already registered",
        )

    return {
        "success": True,
        "message": f"Camera '{request.camera_id}' registered (floor {request.floor})",
        "camera_id": request.camera_id,
        "floor": request.floor,
    }


@router.delete(
    "/cameras/{camera_id}",
    summary="Unregister Camera",
    description="Remove a camera from tracking",
)
async def unregister_camera(camera_id: str, auth_user: dict = Depends(require_role("operator"))):
    """Remove a camera and its associated tracks"""
    success = tracker.unregister_camera(camera_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera '{camera_id}' not found",
        )

    return {
        "success": True,
        "message": f"Camera '{camera_id}' unregistered",
    }


@router.get(
    "/cameras",
    summary="List Cameras",
    description="Get all registered cameras and their status",
)
async def list_cameras(auth_user: dict = Depends(require_role("viewer"))):
    """Get all registered cameras"""
    cameras = tracker.get_cameras()
    return {
        "success": True,
        "cameras": cameras,
        "total": len(cameras),
    }


@router.get(
    "/cameras/available",
    summary="Available Camera Slots",
    description="Get info about camera registration status for UI setup",
)
async def get_available_cameras():
    """Get camera registration status (public, no auth required for UI setup)"""
    cameras = tracker.get_cameras()
    return {
        "success": True,
        "registered_cameras": cameras,
        "total_registered": len(cameras),
        "max_cameras": 10,
    }


@router.post(
    "/update",
    response_model=TrackingResponse,
    summary="Update Tracking",
    description="Submit new face detections from a camera frame for tracking",
)
async def update_tracking(
    request: TrackingUpdateRequest,
):
    """
    Update tracker with new detections from a camera frame.
    
    No auth required - called frequently by frontend camera feeds.
    Uses cosine similarity on 512D embeddings for cross-camera matching.
    Each detection automatically gets an embedding-derived hash ID.

    Each detection should include:
    - bbox: Face bounding box {x, y, width, height}
    - embedding: 512D face embedding
    - name: (optional) Recognized person name
    - confidence: Detection confidence
    """
    detections = []
    for det in request.detections:
        detections.append(
            {
                "bbox": (
                    det.get("x", 0),
                    det.get("y", 0),
                    det.get("width", 0),
                    det.get("height", 0),
                ),
                "embedding": det.get("embedding"),
                "name": det.get("name"),
                "confidence": det.get("confidence", 0.0),
            }
        )

    updated_tracks = tracker.update(
        camera_id=request.camera_id,
        detections=detections,
    )

    return TrackingResponse(
        success=True,
        active_tracks=[t.to_dict() for t in updated_tracks],
        total_active=len(tracker.active_tracks),
        camera_id=request.camera_id,
        timestamp=datetime.utcnow().isoformat(),
    )


@router.get(
    "/tracks",
    summary="Get Active Tracks",
    description="Get all currently active face tracks",
)
async def get_active_tracks(
    camera_id: Optional[str] = None,
    auth_user: dict = Depends(require_role("viewer")),
):
    """Get active tracks, optionally filtered by camera"""
    tracks = tracker.get_active_tracks(camera_id)
    return {
        "success": True,
        "tracks": [t.to_dict() for t in tracks],
        "total": len(tracks),
        "camera_filter": camera_id,
    }


@router.get(
    "/tracks/{track_id}",
    summary="Get Track Details",
    description="Get details of a specific track",
)
async def get_track_detail(track_id: str, auth_user: dict = Depends(require_role("viewer"))):
    """Get a specific track by ID"""
    track = tracker.get_track(track_id)
    if track is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Track '{track_id}' not found",
        )

    return {"success": True, "track": track.to_dict()}


@router.get(
    "/person/{name}/trail",
    summary="Person Trail",
    description="Get movement trail of a person across all cameras",
)
async def get_person_trail(name: str, auth_user: dict = Depends(require_role("viewer"))):
    """Get cross-camera movement trail for a person"""
    trail = tracker.get_person_trail(name)
    return {
        "success": True,
        "person": name,
        "trail": trail,
        "cameras_visited": len(set(t.get("camera_id") for t in trail)),
    }


@router.get(
    "/hash/{hash_id}/trail",
    summary="Hash ID Trail",
    description="Get movement trail of a person by embedding hash ID across all cameras",
)
async def get_hash_trail(hash_id: str, auth_user: dict = Depends(require_role("viewer"))):
    """Get cross-camera movement trail for a person by hash ID"""
    trail = tracker.get_person_trail_by_hash(hash_id)
    return {
        "success": True,
        "hash_id": hash_id,
        "trail": trail,
        "cameras_visited": len(set(t.get("camera_id") for t in trail)),
        "floors_visited": list(set(t.get("floor") for t in trail)),
    }


@router.get(
    "/floor-report",
    summary="Floor Traffic Report",
    description="Get comprehensive floor-by-floor traffic analysis report",
)
async def get_floor_report(auth_user: dict = Depends(require_role("viewer"))):
    """
    Generate floor traffic analysis report.
    
    Includes:
    - Busiest floor identification
    - Per-floor visitor counts and average durations
    - Per-person floor movement history with durations
    """
    report = tracker.get_floor_traffic_report()
    return {
        "success": True,
        **report,
    }


@router.get(
    "/stats",
    response_model=TrackingStatsResponse,
    summary="Tracking Statistics",
    description="Get real-time tracking statistics",
)
async def get_tracking_stats(auth_user: dict = Depends(require_role("viewer"))):
    """Get tracker statistics"""
    stats = tracker.get_statistics()
    return TrackingStatsResponse(
        success=True,
        active_tracks=stats["active_tracks"],
        expired_tracks=stats["expired_tracks"],
        registered_cameras=stats["registered_cameras"],
        match_threshold=stats["match_threshold"],
        track_timeout_seconds=stats["track_timeout_seconds"],
    )


@router.delete(
    "/clear",
    summary="Clear All Tracking Data",
    description="Clear all tracks and camera registrations",
)
async def clear_tracking(auth_user: dict = Depends(require_role("admin"))):
    """Clear all tracking data"""
    tracker.clear_all()
    return {"success": True, "message": "All tracking data cleared"}


# ── Floor Traffic Analytics ─────────────────────────────────────────


class _FloorCount(BaseModel):
    floor: int
    count: int


class _FloorSnapshotRequest(BaseModel):
    floors: List[_FloorCount]


@router.post(
    "/floor-snapshot",
    summary="Log Floor Snapshot",
    description="Record current per-floor people counts. Called by frontend every 2 minutes.",
)
async def log_floor_snapshot(
    data: _FloorSnapshotRequest,
    db: Session = Depends(get_db),
):
    """Persist a floor-count snapshot for time-series analytics."""
    now = datetime.utcnow()
    for fc in data.floors:
        db.add(FloorTrafficLog(floor_number=fc.floor, count=fc.count, recorded_at=now))
    db.commit()
    return {"success": True, "recorded_at": now.isoformat(), "floors": len(data.floors)}


@router.get(
    "/floor-analytics",
    summary="Floor Analytics",
    description="Return time-series snapshots and summary stats for a specific floor.",
)
async def get_floor_analytics(
    floor: int,
    limit: int = 30,
    auth_user: dict = Depends(require_role("viewer")),
    db: Session = Depends(get_db),
):
    """
    Return last `limit` 2-minute snapshots for `floor` (1-indexed),
    plus summary: peak count, peak time, average, current count.
    Also returns which floor had the highest count in the most recent snapshot.
    """
    rows = (
        db.query(FloorTrafficLog)
        .filter(FloorTrafficLog.floor_number == floor)
        .order_by(FloorTrafficLog.recorded_at.desc())
        .limit(limit)
        .all()
    )
    rows_asc = list(reversed(rows))

    snapshots = [
        {
            "count": r.count,
            "recorded_at": r.recorded_at.isoformat(),
            "label": r.recorded_at.strftime("%H:%M"),
        }
        for r in rows_asc
    ]

    counts = [r.count for r in rows]
    peak_count = max(counts, default=0)
    avg_count = round(sum(counts) / len(counts), 1) if counts else 0.0
    peak_row = max(rows, key=lambda r: r.count) if rows else None

    # Peak floor: query the most recent snapshot timestamp across all floors
    latest_ts = (
        db.query(FloorTrafficLog.recorded_at)
        .order_by(FloorTrafficLog.recorded_at.desc())
        .first()
    )
    peak_floor_info = None
    if latest_ts:
        latest_rows = (
            db.query(FloorTrafficLog)
            .filter(FloorTrafficLog.recorded_at == latest_ts[0])
            .all()
        )
        if latest_rows:
            best = max(latest_rows, key=lambda r: r.count)
            peak_floor_info = {"floor": best.floor_number, "count": best.count}

    return {
        "success": True,
        "floor": floor,
        "snapshots": snapshots,
        "summary": {
            "peak_count": peak_count,
            "peak_time": peak_row.recorded_at.strftime("%H:%M") if peak_row else None,
            "avg_count": avg_count,
            "total_snapshots": len(snapshots),
            "latest_count": rows[0].count if rows else 0,
        },
        "peak_floor": peak_floor_info,
    }
