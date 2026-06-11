"""
Project Sentinel — Pydantic schemas for request / response validation.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


# ── Person ───────────────────────────────────────────────────────────────────

class PersonCreate(BaseModel):
    name: str
    role: str = "User"
    notes: Optional[str] = None


class PersonResponse(BaseModel):
    id: int
    name: str
    role: str
    notes: Optional[str] = None
    created_at: datetime
    enrollment_count: int
    is_enrolled: bool

    model_config = ConfigDict(from_attributes=True)


# ── Enrollment ───────────────────────────────────────────────────────────────

class EnrollRequest(BaseModel):
    person_id: int
    image_base64: str
    angle_label: str = "front"


class EnrollResponse(BaseModel):
    success: bool
    person_id: int
    angle_label: str
    quality_score: float
    message: str
    embedding_id: int


class EnrollmentStatus(BaseModel):
    person_id: int
    person_name: str
    angles_enrolled: list[str]
    total_embeddings: int


# ── Authentication ───────────────────────────────────────────────────────────

class AuthRequest(BaseModel):
    image_base64: str


class AuthResult(BaseModel):
    matched: bool
    person_id: Optional[int] = None
    person_name: Optional[str] = None
    person_role: Optional[str] = None
    confidence: float
    message: str
