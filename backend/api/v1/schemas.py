"""
Request and Response Models (Pydantic Schemas)
Used for API documentation, validation, and serialization
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime


# ============================================================
# FACE DETECTION SCHEMAS
# ============================================================

class FaceDetectionRequest(BaseModel):
    """Request model for face detection"""
    image: str = Field(..., description="Base64 encoded image")
    min_confidence: float = Field(0.5, ge=0.0, le=1.0, description="Minimum confidence threshold")

    class Config:
        json_schema_extra = {
            "example": {
                "image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                "min_confidence": 0.5
            }
        }


class FaceDetectionResponse(BaseModel):
    """Response model for face detection"""
    success: bool = Field(..., description="Operation success status")
    faces_detected: int = Field(..., description="Number of faces detected")
    face_locations: List[dict] = Field(default_factory=list, description="Bounding boxes of detected faces")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    model_used: Optional[str] = Field(None, description="Detection model used (retinaface/haar)")


# ============================================================
# FEATURE EXTRACTION SCHEMAS
# ============================================================

class FeatureExtractionRequest(BaseModel):
    """Request model for feature extraction"""
    face_roi: str = Field(..., description="Base64 encoded face ROI image")
    model: str = Field("arcface", description="Feature extraction model: arcface or histogram_lbp")


class FeatureExtractionResponse(BaseModel):
    """Response model for feature extraction"""
    success: bool = Field(..., description="Operation success status")
    embedding: List[float] = Field(..., description="512D face embedding vector")
    model_used: str = Field(..., description="Model that was used for extraction")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


# ============================================================
# FACE SEARCH SCHEMAS
# ============================================================

class FaceSearchRequest(BaseModel):
    """Request model for face search"""
    embedding: List[float] = Field(..., description="512D face embedding vector")
    top_k: int = Field(3, ge=1, le=100, description="Number of top results to return")
    threshold: float = Field(0.3, ge=0.0, le=1.0, description="Similarity threshold")


class FaceSearchResult(BaseModel):
    """Single search result"""
    milvus_id: int = Field(..., description="Milvus database ID")
    name: str = Field(..., description="Person name")
    distance: float = Field(..., description="L2 distance (lower is better)")
    timestamp: int = Field(..., description="Unix timestamp when face was registered")


class FaceSearchResponse(BaseModel):
    """Response model for face search"""
    success: bool = Field(..., description="Operation success status")
    matches: List[FaceSearchResult] = Field(default_factory=list, description="List of matching faces")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


# ============================================================
# USER MANAGEMENT SCHEMAS
# ============================================================

class UserRegistrationRequest(BaseModel):
    """Request model for user registration"""
    name: str = Field(..., min_length=1, max_length=100, description="User full name")
    embedding: Optional[List[float]] = Field(None, description="Optional: 512D face embedding")
    metadata: Optional[dict] = Field(None, description="Additional user metadata")


class UserResponse(BaseModel):
    """Response model for user data"""
    user_id: int = Field(..., description="User database ID")
    name: str = Field(..., description="User full name")
    embeddings_count: int = Field(..., description="Number of face embeddings for this user")
    created_at: datetime = Field(..., description="Account creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    metadata: Optional[dict] = Field(None, description="User metadata")


class UserListResponse(BaseModel):
    """Response model for user list"""
    success: bool = Field(..., description="Operation success status")
    users: List[UserResponse] = Field(default_factory=list, description="List of users")
    total: int = Field(..., description="Total number of users")


class UserDeleteResponse(BaseModel):
    """Response model for user deletion"""
    success: bool = Field(..., description="Deletion success status")
    message: str = Field(..., description="Deletion confirmation message")
    deleted_embeddings: int = Field(..., description="Number of embeddings deleted")


# ============================================================
# VECTOR OPERATIONS SCHEMAS
# ============================================================

class VectorInsertRequest(BaseModel):
    """Request model for vector insertion"""
    embedding: List[float] = Field(..., description="512D face embedding vector")
    name: str = Field(..., description="Associated person name")


class VectorInsertResponse(BaseModel):
    """Response model for vector insertion"""
    success: bool = Field(..., description="Insertion success status")
    milvus_id: int = Field(..., description="Milvus assigned ID")
    message: str = Field(..., description="Confirmation message")


class VectorStatsResponse(BaseModel):
    """Response model for vector statistics"""
    success: bool = Field(..., description="Operation success status")
    total_vectors: int = Field(..., description="Total vectors in database")
    collection_name: str = Field(..., description="Milvus collection name")
    vector_dimension: int = Field(..., description="Dimension of vectors (512)")
    vector_count_by_user: dict = Field(default_factory=dict, description="Vector count per user")


# ============================================================
# HEALTH & STATUS SCHEMAS
# ============================================================

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str = Field(..., description="System status (healthy/degraded/unhealthy)")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Health check timestamp")
    version: str = Field(..., description="API version")
    services: dict = Field(default_factory=dict, description="Status of connected services")


# ============================================================
# AUTHENTICATION SCHEMAS
# ============================================================

class LoginRequest(BaseModel):
    """Request model for login"""
    username: str = Field(..., description="Username")
    password: str = Field(..., description="Password")

    class Config:
        json_schema_extra = {
            "example": {"username": "admin", "password": "admin123"}
        }


class TokenResponse(BaseModel):
    """Response model for token generation"""
    access_token: str = Field(..., description="JWT access token")
    refresh_token: Optional[str] = Field(None, description="JWT refresh token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration in seconds")


class UserRegisterAuthRequest(BaseModel):
    """Request model for auth user registration"""
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    password: str = Field(..., min_length=6, description="Password (min 6 chars)")
    email: Optional[str] = Field(None, description="Email address")
    full_name: Optional[str] = Field(None, description="Full name")

    class Config:
        json_schema_extra = {
            "example": {
                "username": "newuser",
                "password": "password123",
                "email": "user@example.com",
                "full_name": "New User",
            }
        }


class UserAuthResponse(BaseModel):
    """Response model for auth user info"""
    username: str = Field(..., description="Username")
    email: str = Field("", description="Email address")
    full_name: str = Field("", description="Full name")
    role: str = Field(..., description="User role (admin/operator/viewer/user)")
    disabled: bool = Field(..., description="Account disabled status")
    created_at: str = Field("", description="Account creation timestamp")


class TokenVerifyResponse(BaseModel):
    """Response model for token verification"""
    valid: bool = Field(..., description="Token validity")
    message: str = Field(..., description="Verification message")
    username: Optional[str] = Field(None, description="Token owner username")
    role: Optional[str] = Field(None, description="User role")
    expires_at: Optional[str] = Field(None, description="Token expiry timestamp")


# ============================================================
# REAL-TIME TRACKING SCHEMAS
# ============================================================

class CameraRegisterRequest(BaseModel):
    """Request model for camera registration"""
    camera_id: str = Field(..., description="Unique camera identifier")
    location: Optional[str] = Field(None, description="Camera location description")
    floor: Optional[int] = Field(0, description="Floor number this camera monitors (0=entrance, 1=first floor, etc.)")

    class Config:
        json_schema_extra = {
            "example": {"camera_id": "cam-entrance-01", "location": "Main Entrance", "floor": 1}
        }


class TrackingUpdateRequest(BaseModel):
    """Request model for tracking update"""
    camera_id: str = Field(..., description="Source camera ID")
    detections: List[Dict] = Field(..., description="List of face detections")

    class Config:
        json_schema_extra = {
            "example": {
                "camera_id": "cam-entrance-01",
                "detections": [
                    {
                        "x": 100, "y": 150, "width": 80, "height": 100,
                        "embedding": [0.1] * 512,
                        "name": "John",
                        "confidence": 0.95,
                    }
                ],
            }
        }


class TrackingResponse(BaseModel):
    """Response model for tracking update"""
    success: bool
    active_tracks: List[Dict]
    total_active: int
    camera_id: str
    timestamp: str


class TrackingStatsResponse(BaseModel):
    """Response model for tracking statistics"""
    success: bool
    active_tracks: int
    expired_tracks: int
    registered_cameras: int
    match_threshold: float
    track_timeout_seconds: float


# ============================================================
# FACE TOKEN SCHEMAS
# ============================================================

class FaceTokenRequest(BaseModel):
    """Request model for face token generation"""
    person_name: str = Field(..., description="Recognized person's name")
    embedding: List[float] = Field(..., description="512D face embedding")
    confidence: float = Field(0.0, description="Recognition confidence")
    camera_id: Optional[str] = Field(None, description="Source camera ID")
    expiry_minutes: Optional[int] = Field(None, description="Custom token expiry in minutes")
    metadata: Optional[Dict] = Field(None, description="Additional metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "person_name": "John Doe",
                "embedding": [0.1] * 512,
                "confidence": 0.95,
                "camera_id": "cam-entrance-01",
                "expiry_minutes": 60,
            }
        }


class FaceTokenResponse(BaseModel):
    """Response model for generated face token"""
    success: bool
    token_id: str
    person_name: str
    issued_at: str
    expires_at: str
    is_valid: bool


class FaceTokenVerifyResponse(BaseModel):
    """Response model for token verification"""
    valid: bool
    message: str
    token_id: str
    person_name: Optional[str] = None
    issued_at: Optional[str] = None
    expires_at: Optional[str] = None


class FaceTokenStatsResponse(BaseModel):
    """Response model for token statistics"""
    success: bool
    total_tokens: int
    active_tokens: int
    expired_tokens: int
    revoked_tokens: int
    token_expiry_minutes: int


# ============================================================
# ERROR SCHEMAS
# ============================================================

class ErrorResponse(BaseModel):
    """Response model for errors"""
    success: bool = Field(False, description="Operation success status")
    error_code: str = Field(..., description="Error code identifier")
    message: str = Field(..., description="Error message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    details: Optional[dict] = Field(None, description="Additional error details")
