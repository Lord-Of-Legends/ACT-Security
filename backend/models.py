"""
Project Sentinel — SQLAlchemy ORM models.
Defines Person and FaceEmbedding tables.
"""

from datetime import datetime, timezone
from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Float, ForeignKey, LargeBinary,
)
from sqlalchemy.orm import relationship

from database import Base


class Person(Base):
    __tablename__ = "persons"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    name: str = Column(String(100), nullable=False)
    role: str = Column(String(20), default="User")  # Admin | Manager | User
    notes: str | None = Column(Text, nullable=True)
    created_at: datetime = Column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )

    face_embeddings = relationship(
        "FaceEmbedding",
        back_populates="person",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Person(id={self.id}, name='{self.name}', role='{self.role}')>"


class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    person_id: int = Column(
        Integer, ForeignKey("persons.id"), nullable=False
    )
    embedding_blob: bytes = Column(LargeBinary, nullable=False)
    snapshot_path: str | None = Column(String(255), nullable=True)
    angle_label: str = Column(String(20), default="front")  # front | left | right
    quality_score: float | None = Column(Float, nullable=True)
    created_at: datetime = Column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )

    person = relationship("Person", back_populates="face_embeddings")

    def __repr__(self) -> str:
        return (
            f"<FaceEmbedding(id={self.id}, person_id={self.person_id}, "
            f"angle='{self.angle_label}')>"
        )
