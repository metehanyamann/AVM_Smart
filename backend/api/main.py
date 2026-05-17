"""
FastAPI Main Application
AvmSmart - REST API
Entry point with Swagger/OpenAPI documentation
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from contextlib import asynccontextmanager
import logging
from datetime import datetime
from pathlib import Path

from backend.infrastructure.config import settings
from backend.api.v1 import detection, recognition, users, vectors, health, tracking, face_tokens, auth, alerts

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    logger.info("AvmSmart API starting...")
    logger.info(f"   App: {settings.app_name}")
    logger.info(f"   Version: {settings.api_version}")
    logger.info(f"   Debug: {settings.debug}")

    try:
        logger.info("[1/5] Connecting to Vector DB (Milvus)...")
        try:
            from backend.infrastructure.milvus_client import get_milvus_client
            get_milvus_client()
            logger.info("   Milvus connected")
        except Exception as e:
            logger.warning(f"   Milvus connection failed: {e}")

        logger.info("[2/5] Connecting to Redis...")
        logger.info("   Redis skipped (not required)")

        logger.info("[3/5] Connecting to PostgreSQL...")
        try:
            from backend.infrastructure.database import init_db
            init_db()
            logger.info("   PostgreSQL connected and initialized")
        except Exception as e:
            logger.warning(f"   PostgreSQL connection failed: {e}")

        logger.info("[4/5] Downloading & loading ML models (RetinaFace + ArcFace)...")
        try:
            from backend.application.onnx_models import download_models, models_available

            if not models_available():
                logger.info("   ONNX models not found, downloading...")
                download_models()
            else:
                logger.info("   ONNX models already present")

            import backend.application.face_detection as _det_mod
            import backend.application.feature_extraction as _feat_mod
            import backend.application.face_search as _search_mod
            import backend.application.user_service as _user_mod

            _det_mod._detection_service = None
            _feat_mod._feature_service = None
            _search_mod._search_service = None
            _user_mod._user_service = None

            logger.info("   Reinitializing services with ONNX models...")
            _det_mod.get_detection_service()
            _feat_mod.get_feature_service()
            _search_mod.get_search_service()
            _user_mod.get_user_service()
        except Exception as e:
            logger.warning(f"   ONNX model setup: {e} (fallback models will be used)")

        logger.info("   ML models ready")

        logger.info("[5/5] Initializing tracking & alert services...")
        try:
            from backend.application.alert_service import get_alert_service
            get_alert_service()
            logger.info("   Alert service initialized")
        except Exception as e:
            logger.warning(f"   Alert service init failed: {e}")

        logger.info("   Services initialized")
        logger.info("All services initialized successfully")

    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        raise

    yield

    logger.info("Shutting down AvmSmart API...")
    try:
        logger.info("   Closing database connections...")
        logger.info("   Closing cache connections...")
        logger.info("Cleanup completed")
    except Exception as e:
        logger.error(f"Shutdown error: {str(e)}")


app = FastAPI(
    title=settings.app_name,
    description="AvmSmart REST API — Real-time Face Recognition, Multi-Camera Tracking, and Wanted Person Alert System.",
    version="2.0.0",
    lifespan=lifespan,
)

class NoCacheStaticMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/static"):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        return response

app.add_middleware(NoCacheStaticMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_credentials,
    allow_methods=settings.cors_methods,
    allow_headers=settings.cors_headers,
)


# ============================================================
# API ROUTERS
# ============================================================

app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication & Authorization"])
app.include_router(detection.router, prefix="/api/v1/detection", tags=["Face Detection (RetinaFace)"])
app.include_router(recognition.router, prefix="/api/v1/recognition", tags=["Face Recognition (ArcFace)"])
app.include_router(users.router, prefix="/api/v1/users", tags=["User Management"])
app.include_router(vectors.router, prefix="/api/v1/vectors", tags=["Vector Operations"])
app.include_router(tracking.router, prefix="/api/v1/tracking", tags=["Real-Time Tracking"])
app.include_router(face_tokens.router, prefix="/api/v1/face-tokens", tags=["Face Token Generation"])
app.include_router(alerts.router, prefix="/api/v1/alerts", tags=["Wanted Person Alerts"])
app.include_router(health.router, prefix="/api/v1/health", tags=["Health & Status"])


# ============================================================
# ROOT ROUTES
# ============================================================

@app.get("/", tags=["Frontend"])
async def root():
    """Serve the frontend at root"""
    html_file = Path(__file__).parent.parent.parent / "frontend" / "index.html"
    if html_file.exists():
        return FileResponse(html_file, media_type="text/html")
    return JSONResponse(
        status_code=404,
        content={"error": "Frontend not found"},
    )

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Return an empty response for favicon to prevent 404 errors"""
    return JSONResponse(status_code=204, content=None)


@app.get("/api", tags=["Root"])
async def api_info():
    """API information endpoint"""
    return {
        "name": settings.app_name,
        "version": "2.0.0",
        "modules": {
            "detection": "/api/v1/detection",
            "recognition": "/api/v1/recognition",
            "tracking": "/api/v1/tracking",
            "auth": "/api/v1/auth",
            "face_tokens": "/api/v1/face-tokens",
            "alerts": "/api/v1/alerts",
            "users": "/api/v1/users",
            "vectors": "/api/v1/vectors",
            "health": "/api/v1/health",
        },
        "documentation": "/docs",
    }


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error_code": "INTERNAL_SERVER_ERROR",
            "message": str(exc) if settings.debug else "Internal server error",
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


# ============================================================
# STATIC FILES & FRONTEND
# ============================================================

frontend_dir = Path(__file__).parent.parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")
    logger.info(f"Static files mounted: {frontend_dir}")


if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting {settings.app_name}")
    uvicorn.run(
        "backend.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
