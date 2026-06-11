"""
Project Sentinel — Face enrollment routes.
Handles enrolling face embeddings against registered persons.
"""

import os
import pickle
import time
import logging

import cv2
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from database import get_db
from models import Person, FaceEmbedding
from schemas import EnrollRequest, EnrollResponse, EnrollmentStatus
from face_engine import face_engine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/enroll", tags=["enrollment"])

FACES_DIR = "storage/faces"


@router.post("/", response_model=EnrollResponse)
async def enroll_face(
    request: EnrollRequest,
    db: Session = Depends(get_db),
) -> EnrollResponse:
    """Enroll a face embedding for an existing person."""
    # 1. Verify person exists
    person = db.query(Person).filter(Person.id == request.person_id).first()
    if not person:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Person with id {request.person_id} not found",
        )

    # 2. Decode the base64 image
    try:
        image = face_engine.decode_base64_image(request.image_base64)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to decode image: {exc}",
        )

    # 3. Extract embedding
    embedding, quality_score, metadata = face_engine.get_embedding(image)

    if embedding is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=metadata.get("error", "No face detected in the image"),
        )

    # 4. Save snapshot image
    os.makedirs(FACES_DIR, exist_ok=True)
    timestamp = int(time.time() * 1000)
    snapshot_filename = f"{request.person_id}_{request.angle_label}_{timestamp}.jpg"
    snapshot_path = os.path.join(FACES_DIR, snapshot_filename)
    cv2.imwrite(snapshot_path, image)
    logger.info("Saved face snapshot: %s", snapshot_path)

    # 5. Pickle the embedding and store in DB
    embedding_bytes = pickle.dumps(embedding)

    face_record = FaceEmbedding(
        person_id=request.person_id,
        embedding_blob=embedding_bytes,
        snapshot_path=snapshot_path,
        angle_label=request.angle_label,
        quality_score=quality_score,
    )
    db.add(face_record)
    db.commit()
    db.refresh(face_record)

    logger.info(
        "Enrolled face for person_id=%d angle=%s quality=%.1f embedding_id=%d",
        request.person_id,
        request.angle_label,
        quality_score,
        face_record.id,
    )

    return EnrollResponse(
        success=True,
        person_id=request.person_id,
        angle_label=request.angle_label,
        quality_score=quality_score,
        message=f"Face enrolled successfully (quality: {quality_score})",
        embedding_id=face_record.id,
    )


@router.get("/{person_id}", response_model=EnrollmentStatus)
async def get_enrollment_status(
    person_id: int,
    db: Session = Depends(get_db),
) -> EnrollmentStatus:
    """Get enrollment status for a person — which angles are enrolled."""
    person = db.query(Person).filter(Person.id == person_id).first()
    if not person:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Person with id {person_id} not found",
        )

    embeddings = (
        db.query(FaceEmbedding)
        .filter(FaceEmbedding.person_id == person_id)
        .all()
    )

    angles = [e.angle_label for e in embeddings]

    return EnrollmentStatus(
        person_id=person_id,
        person_name=person.name,
        angles_enrolled=angles,
        total_embeddings=len(embeddings),
    )


@router.delete("/{person_id}", status_code=status.HTTP_200_OK)
async def delete_enrollments(
    person_id: int,
    db: Session = Depends(get_db),
) -> dict:
    """Delete all face enrollments for a person (DB records + snapshot files)."""
    person = db.query(Person).filter(Person.id == person_id).first()
    if not person:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Person with id {person_id} not found",
        )

    embeddings = (
        db.query(FaceEmbedding)
        .filter(FaceEmbedding.person_id == person_id)
        .all()
    )

    deleted_count = 0
    for emb in embeddings:
        # Delete snapshot file
        if emb.snapshot_path and os.path.exists(emb.snapshot_path):
            try:
                os.remove(emb.snapshot_path)
            except OSError as exc:
                logger.warning("Failed to delete snapshot %s: %s", emb.snapshot_path, exc)
        db.delete(emb)
        deleted_count += 1

    db.commit()
    logger.info(
        "Deleted %d enrollments for person_id=%d", deleted_count, person_id
    )

    return {
        "message": f"Deleted {deleted_count} enrollment(s) for '{person.name}'",
        "deleted_count": deleted_count,
    }
