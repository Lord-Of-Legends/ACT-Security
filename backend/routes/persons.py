"""
Project Sentinel — Person management routes.
CRUD operations for persons in the system.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from database import get_db
from models import Person, FaceEmbedding
from schemas import PersonCreate, PersonResponse

import os
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/persons", tags=["persons"])


def _person_to_response(person: Person) -> PersonResponse:
    """Convert a Person ORM object to a PersonResponse with enrollment info."""
    enrollment_count = len(person.face_embeddings)
    return PersonResponse(
        id=person.id,
        name=person.name,
        role=person.role,
        notes=person.notes,
        created_at=person.created_at,
        enrollment_count=enrollment_count,
        is_enrolled=enrollment_count > 0,
    )


@router.post("/", response_model=PersonResponse, status_code=status.HTTP_201_CREATED)
async def create_person(
    person_data: PersonCreate,
    db: Session = Depends(get_db),
) -> PersonResponse:
    """Create a new person in the system."""
    person = Person(
        name=person_data.name,
        role=person_data.role,
        notes=person_data.notes,
    )
    db.add(person)
    db.commit()
    db.refresh(person)
    logger.info("Created person: id=%d name='%s'", person.id, person.name)
    return _person_to_response(person)


@router.get("/", response_model=list[PersonResponse])
async def list_persons(
    db: Session = Depends(get_db),
) -> list[PersonResponse]:
    """List all persons with their enrollment status."""
    persons = db.query(Person).order_by(Person.id).all()
    return [_person_to_response(p) for p in persons]


@router.get("/{person_id}", response_model=PersonResponse)
async def get_person(
    person_id: int,
    db: Session = Depends(get_db),
) -> PersonResponse:
    """Get a single person by ID with enrollment info."""
    person = db.query(Person).filter(Person.id == person_id).first()
    if not person:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Person with id {person_id} not found",
        )
    return _person_to_response(person)


@router.delete("/{person_id}", status_code=status.HTTP_200_OK)
async def delete_person(
    person_id: int,
    db: Session = Depends(get_db),
) -> dict:
    """Delete a person and all their face embeddings / snapshot files."""
    person = db.query(Person).filter(Person.id == person_id).first()
    if not person:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Person with id {person_id} not found",
        )

    # Delete snapshot files from disk
    for embedding in person.face_embeddings:
        if embedding.snapshot_path and os.path.exists(embedding.snapshot_path):
            try:
                os.remove(embedding.snapshot_path)
                logger.info("Deleted snapshot: %s", embedding.snapshot_path)
            except OSError as exc:
                logger.warning("Failed to delete snapshot %s: %s", embedding.snapshot_path, exc)

    person_name = person.name
    db.delete(person)
    db.commit()
    logger.info("Deleted person: id=%d name='%s'", person_id, person_name)
    return {"message": f"Person '{person_name}' and all enrollments deleted"}
