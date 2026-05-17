"""
Domain Enums
Business-level enumeration types for AvmSmart.
"""

from enum import Enum


class TrackStatus(str, Enum):
    """Track lifecycle status"""
    PENDING = "PENDING"
    CONFIRMED = "CONFIRMED"
    EXPIRED = "EXPIRED"


class AlertLevel(str, Enum):
    """Alert severity levels"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AlertStatus(str, Enum):
    """Alert lifecycle status"""
    ACTIVE = "ACTIVE"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    RESOLVED = "RESOLVED"
    DISMISSED = "DISMISSED"


class UserRole(str, Enum):
    """User role hierarchy: admin > operator > viewer > user"""
    ADMIN = "admin"
    OPERATOR = "operator"
    VIEWER = "viewer"
    USER = "user"
