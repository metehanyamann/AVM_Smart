import logging
from typing import Generator
from sqlalchemy import create_engine, Column, String, Boolean, DateTime, Integer, Float, Text, text
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from datetime import datetime
from pgvector.sqlalchemy import Vector

from backend.infrastructure.config import settings

logger = logging.getLogger(__name__)

SQLALCHEMY_DATABASE_URL = settings.database_url

try:
    engine = create_engine(SQLALCHEMY_DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()
except Exception as e:
    logger.error(f"Failed to initialize database engine: {e}")
    engine = None
    SessionLocal = None
    Base = declarative_base()


class UserDB(Base):
    """SQLAlchemy model for Users"""
    __tablename__ = "users"

    username = Column(String(50), primary_key=True, index=True)
    hashed_password = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=True)
    full_name = Column(String(100), nullable=True)
    role = Column(String(20), default="user")
    disabled = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class FaceEmbedding(Base):
    """SQLAlchemy model for Face Vectors"""
    __tablename__ = "face_embeddings"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(100), index=True)
    embedding = Column(Vector(512))
    timestamp = Column(Integer)


class WantedFace(Base):
    """SQLAlchemy model for Wanted Face Vectors"""
    __tablename__ = "wanted_faces"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    wanted_id = Column(String(100), index=True)
    name = Column(String(100))
    description = Column(Text, nullable=True)
    alert_level = Column(String(50), default="HIGH")
    embedding = Column(Vector(512))
    timestamp = Column(Integer)
    photo_base64 = Column(Text, nullable=True)
    added_by = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class FloorTrafficLog(Base):
    """Periodic floor traffic snapshots for time-series analytics"""
    __tablename__ = "floor_traffic_logs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    floor_number = Column(Integer, nullable=False, index=True)
    count = Column(Integer, default=0)
    recorded_at = Column(DateTime, default=datetime.utcnow, index=True)


def init_db():
    """Create all tables in the database."""
    if engine:
        try:
            with engine.connect() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                conn.commit()
            
            Base.metadata.create_all(bind=engine)
            with engine.connect() as conn:
                for ddl in (
                    "ALTER TABLE wanted_faces ADD COLUMN IF NOT EXISTS description TEXT;",
                    "ALTER TABLE wanted_faces ADD COLUMN IF NOT EXISTS photo_base64 TEXT;",
                    "ALTER TABLE wanted_faces ADD COLUMN IF NOT EXISTS added_by VARCHAR(100);",
                    "ALTER TABLE wanted_faces ADD COLUMN IF NOT EXISTS created_at TIMESTAMP;",
                ):
                    conn.execute(text(ddl))
                conn.commit()
            logger.info("Database tables and pgvector extension created successfully.")
            
            # Create default admin user if it doesn't exist
            db = SessionLocal()
            admin = db.query(UserDB).filter(UserDB.username == "admin").first()
            if not admin:
                import bcrypt
                hashed = bcrypt.hashpw("admin123".encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
                new_admin = UserDB(
                    username="admin",
                    hashed_password=hashed,
                    email="admin@avm.local",
                    full_name="Admin User",
                    role="admin",
                    created_at=datetime.utcnow()
                )
                db.add(new_admin)
                db.commit()
                logger.info("Default admin user created.")
            db.close()
            
        except Exception as e:
            logger.error(f"Error creating database tables: {e}")


def get_db() -> Generator[Session, None, None]:
    """Dependency to get database session."""
    if not SessionLocal:
        raise RuntimeError("Database not initialized")
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
