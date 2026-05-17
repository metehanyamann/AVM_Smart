"""
Authentication & Authorization Service
JWT token management, password hashing, role-based access control.
Uses PostgreSQL via SQLAlchemy for user persistence.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import bcrypt

try:
    from jose import JWTError, jwt
except ImportError:
    JWTError = Exception
    jwt = None

from backend.infrastructure.config import settings
from backend.infrastructure.database import SessionLocal, UserDB

logger = logging.getLogger(__name__)


def hash_password(password: str) -> str:
    """Hash a password using bcrypt"""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"), hashed_password.encode("utf-8")
        )
    except Exception:
        return False


TOKEN_BLACKLIST: set = set()


def create_access_token(
    data: dict, expires_delta: Optional[timedelta] = None
) -> str:
    """Create JWT access token"""
    if jwt is None:
        raise RuntimeError("python-jose is not installed")

    to_encode = data.copy()
    expire = datetime.utcnow() + (
        expires_delta or timedelta(minutes=settings.access_token_expire_minutes)
    )
    to_encode.update({"exp": expire, "iat": datetime.utcnow(), "type": "access"})

    encoded_jwt = jwt.encode(
        to_encode, settings.secret_key, algorithm=settings.algorithm
    )
    return encoded_jwt


def create_refresh_token(data: dict) -> str:
    """Create JWT refresh token (longer expiry)"""
    if jwt is None:
        raise RuntimeError("python-jose is not installed")

    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=7)
    to_encode.update({"exp": expire, "iat": datetime.utcnow(), "type": "refresh"})

    encoded_jwt = jwt.encode(
        to_encode, settings.secret_key, algorithm=settings.algorithm
    )
    return encoded_jwt


def verify_token(token: str) -> Optional[Dict]:
    """Verify and decode a JWT token. Returns payload dict or None if invalid."""
    if jwt is None:
        return None

    if token in TOKEN_BLACKLIST:
        logger.warning("Token is blacklisted")
        return None

    try:
        payload = jwt.decode(
            token, settings.secret_key, algorithms=[settings.algorithm]
        )
        username: str = payload.get("sub")
        if username is None:
            return None
        return payload
    except JWTError as e:
        logger.warning(f"Token verification failed: {str(e)}")
        return None


def blacklist_token(token: str):
    """Add token to blacklist (for logout)"""
    TOKEN_BLACKLIST.add(token)


def authenticate_user(username: str, password: str) -> Optional[Dict]:
    """Authenticate user with username and password"""
    if not SessionLocal:
        return None
        
    with SessionLocal() as db:
        user_db = db.query(UserDB).filter(UserDB.username == username).first()
        if not user_db:
            return None
        if user_db.disabled:
            return None
        if not verify_password(password, user_db.hashed_password):
            return None
            
        return {
            "username": user_db.username,
            "role": user_db.role,
            "email": user_db.email,
            "full_name": user_db.full_name,
            "disabled": user_db.disabled,
            "created_at": user_db.created_at.isoformat() if user_db.created_at else ""
        }


def register_user(
    username: str,
    password: str,
    email: str = "",
    full_name: str = "",
    role: str = "user",
) -> Optional[Dict]:
    """Register a new user"""
    if len(password) < 6:
        logger.warning("Password too short")
        return None
        
    if not SessionLocal:
        return None

    with SessionLocal() as db:
        existing = db.query(UserDB).filter(UserDB.username == username).first()
        if existing:
            logger.warning(f"User already exists: {username}")
            return None

        new_user = UserDB(
            username=username,
            hashed_password=hash_password(password),
            email=email,
            full_name=full_name,
            role=role,
            disabled=False
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        logger.info(f"User registered: {username} (role={role})")
        return {
            "username": new_user.username,
            "role": new_user.role,
            "email": new_user.email,
            "full_name": new_user.full_name,
            "disabled": new_user.disabled,
            "created_at": new_user.created_at.isoformat() if new_user.created_at else ""
        }


def get_user(username: str) -> Optional[Dict]:
    """Get user by username"""
    if not SessionLocal:
        return None
        
    with SessionLocal() as db:
        user_db = db.query(UserDB).filter(UserDB.username == username).first()
        if not user_db:
            return None
        return {
            "username": user_db.username,
            "role": user_db.role,
            "email": user_db.email,
            "full_name": user_db.full_name,
            "disabled": user_db.disabled,
            "created_at": user_db.created_at.isoformat() if user_db.created_at else ""
        }


def get_all_users() -> List[Dict]:
    """Get all users (without passwords)"""
    if not SessionLocal:
        return []
        
    with SessionLocal() as db:
        users = db.query(UserDB).all()
        return [
            {
                "username": u.username,
                "role": u.role,
                "email": u.email,
                "full_name": u.full_name,
                "disabled": u.disabled,
                "created_at": u.created_at.isoformat() if u.created_at else ""
            }
            for u in users
        ]


def update_user_role(username: str, new_role: str) -> bool:
    """Update user role (admin only)"""
    valid_roles = {"admin", "operator", "viewer", "user"}
    if new_role not in valid_roles:
        return False
        
    if not SessionLocal:
        return False

    with SessionLocal() as db:
        user_db = db.query(UserDB).filter(UserDB.username == username).first()
        if not user_db:
            return False
            
        user_db.role = new_role
        db.commit()
        logger.info(f"User role updated: {username} -> {new_role}")
        return True


def disable_user(username: str) -> bool:
    """Disable a user account"""
    if not SessionLocal:
        return False
        
    with SessionLocal() as db:
        user_db = db.query(UserDB).filter(UserDB.username == username).first()
        if not user_db:
            return False
            
        user_db.disabled = True
        db.commit()
        logger.info(f"User disabled: {username}")
        return True


def check_permission(user: Dict, required_role: str) -> bool:
    """
    Check if user has required role.
    Role hierarchy: admin > operator > viewer > user
    """
    role_levels = {"admin": 4, "operator": 3, "viewer": 2, "user": 1}
    user_level = role_levels.get(user.get("role", "user"), 0)
    required_level = role_levels.get(required_role, 0)
    return user_level >= required_level
