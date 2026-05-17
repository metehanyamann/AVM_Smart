"""
Application Configuration
Uses environment variables from .env file
"""

from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """Application Settings"""

    # App
    app_name: str = "AvmSmart API"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    api_version: str = "v1"

    # Milvus
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection_name: str = "face_embeddings_512"
    milvus_wanted_collection: str = "wanted_faces"

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = ""

    # Database
    database_url: str = "postgresql://admin:root@localhost:5432/avm_db"

    # Security / JWT
    secret_key: str = "your-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 1440

    # ML Models
    face_detection_model: str = "retinaface"
    face_recognition_model: str = "arcface"
    confidence_threshold: float = 0.5
    match_threshold: float = 0.3

    # Tracking
    tracking_match_threshold: float = 0.6
    tracking_timeout_seconds: float = 30.0
    tracking_max_tracks: int = 1000

    # Face Token
    face_token_expiry_minutes: int = 60
    face_token_max_count: int = 10000

    # Alert System
    # 0.40 is more permissive than 0.55 — surveillance angles/lighting
    # push genuine matches down to 0.35-0.50, so 0.55 misses them.
    alert_similarity_threshold: float = 0.40
    alert_cooldown_seconds: int = 60

    # Logging
    log_level: str = "INFO"

    # CORS
    cors_origins: List[str] = ["*"]
    cors_credentials: bool = True
    cors_methods: List[str] = ["*"]
    cors_headers: List[str] = ["*"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = Settings()
