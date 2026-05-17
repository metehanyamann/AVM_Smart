"""
Authentication & Authorization Endpoints
JWT login, registration, token management, role-based access
"""

from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from datetime import datetime
import logging

from backend.infrastructure.config import settings
from backend.api.v1.schemas import (
    LoginRequest,
    TokenResponse,
    UserRegisterAuthRequest,
    UserAuthResponse,
    TokenVerifyResponse,
)
from backend.application.auth_service import (
    authenticate_user,
    register_user,
    create_access_token,
    create_refresh_token,
    verify_token,
    blacklist_token,
    get_user,
    get_all_users,
    update_user_role,
    disable_user,
    check_permission,
)

router = APIRouter()
logger = logging.getLogger(__name__)
security = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    """Extract and verify JWT token, return current user"""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = verify_token(credentials.credentials)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    username = payload.get("sub")
    user = get_user(username)
    if user is None or user.get("disabled"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or disabled",
        )

    return user


def require_role(required_role: str):
    """Dependency that checks user role"""

    async def role_checker(user: dict = Depends(get_current_user)):
        if not check_permission(user, required_role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required: {required_role}",
            )
        return user

    return role_checker


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Login",
    description="Authenticate with username/password, receive JWT tokens",
    responses={
        200: {"description": "Login successful"},
        401: {"description": "Invalid credentials"},
    },
)
async def login(request: LoginRequest):
    """
    Login and get JWT access + refresh tokens.

    **Demo Credentials:**
    - Username: `admin`, Password: `admin123`
    """
    user = authenticate_user(request.username, request.password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    access_token = create_access_token(
        data={"sub": user["username"], "role": user["role"]}
    )
    refresh_token = create_refresh_token(data={"sub": user["username"]})

    logger.info(f"Login successful: {request.username}")

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=settings.access_token_expire_minutes * 60,
    )


@router.post(
    "/register",
    response_model=UserAuthResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register User",
    description="Register a new user account",
    responses={
        201: {"description": "User registered"},
        409: {"description": "Username already exists"},
    },
)
async def register(request: UserRegisterAuthRequest):
    """Register a new user account"""
    user = register_user(
        username=request.username,
        password=request.password,
        email=request.email or "",
        full_name=request.full_name or "",
        role="user",
    )
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username already exists or password too short (min 6 chars)",
        )

    return UserAuthResponse(
        username=user["username"],
        email=user.get("email", ""),
        full_name=user.get("full_name", ""),
        role=user["role"],
        disabled=user["disabled"],
        created_at=user["created_at"],
    )


@router.post(
    "/verify",
    response_model=TokenVerifyResponse,
    summary="Verify Token",
    description="Verify JWT token validity",
    responses={
        200: {"description": "Token verification result"},
    },
)
async def verify_token_endpoint(
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    """Verify if a JWT token is valid"""
    if credentials is None:
        return TokenVerifyResponse(valid=False, message="No token provided")

    payload = verify_token(credentials.credentials)
    if payload is None:
        return TokenVerifyResponse(valid=False, message="Invalid or expired token")

    return TokenVerifyResponse(
        valid=True,
        message="Token is valid",
        username=payload.get("sub"),
        role=payload.get("role"),
        expires_at=datetime.fromtimestamp(payload.get("exp", 0)).isoformat(),
    )


@router.post(
    "/logout",
    summary="Logout",
    description="Invalidate JWT token",
)
async def logout(
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    """Logout by blacklisting the current token"""
    if credentials:
        blacklist_token(credentials.credentials)

    return {
        "success": True,
        "message": "Logged out successfully. Token has been invalidated.",
    }


@router.get(
    "/me",
    response_model=UserAuthResponse,
    summary="Current User",
    description="Get current authenticated user info",
)
async def get_me(user: dict = Depends(get_current_user)):
    """Get current user profile"""
    return UserAuthResponse(
        username=user["username"],
        email=user.get("email", ""),
        full_name=user.get("full_name", ""),
        role=user["role"],
        disabled=user["disabled"],
        created_at=user.get("created_at", ""),
    )


@router.get(
    "/users",
    summary="List All Users (Admin)",
    description="Get all registered users (admin only)",
)
async def list_auth_users(user: dict = Depends(require_role("admin"))):
    """List all users (admin only)"""
    users = get_all_users()
    return {"success": True, "users": users, "total": len(users)}


@router.put(
    "/users/{username}/role",
    summary="Update User Role (Admin)",
    description="Change a user's role (admin only)",
)
async def change_user_role(
    username: str,
    role: str,
    user: dict = Depends(require_role("admin")),
):
    """Update user role (admin only)"""
    if not update_user_role(username, role):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid username or role. Valid roles: admin, operator, viewer, user",
        )
    return {"success": True, "message": f"User {username} role updated to {role}"}


@router.put(
    "/users/{username}/disable",
    summary="Disable User (Admin)",
    description="Disable a user account (admin only)",
)
async def disable_user_endpoint(
    username: str,
    user: dict = Depends(require_role("admin")),
):
    """Disable a user account (admin only)"""
    if not disable_user(username):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {username} not found",
        )
    return {"success": True, "message": f"User {username} has been disabled"}
