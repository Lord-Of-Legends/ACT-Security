"""
Project Sentinel — Face authentication routes.
Matches a probe face against all enrolled embeddings.
"""

import pickle
import logging

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from database import get_db
from models import FaceEmbedding, Person
from schemas import AuthRequest, AuthResult
from face_engine import face_engine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["authentication"])

# Cosine similarity threshold for a positive match
MATCH_THRESHOLD: float = 0.45


@router.post("/authenticate", response_model=AuthResult)
async def authenticate_face(
    request: AuthRequest,
    db: Session = Depends(get_db),
) -> AuthResult:
    """Authenticate a face against all enrolled identities."""

    # 1. Decode the probe image
    try:
        image = face_engine.decode_base64_image(request.image_base64)
    except Exception as exc:
        logger.warning("Image decode failed: %s", exc)
        return AuthResult(
            matched=False,
            confidence=0.0,
            message=f"Failed to decode image: {exc}",
        )

    # 2. Extract embedding from probe image
    query_embedding, _quality, metadata = face_engine.get_embedding(image)

    if query_embedding is None:
        return AuthResult(
            matched=False,
            confidence=0.0,
            message="No face detected",
        )

    # 3. Load all stored embeddings from DB
    records = (
        db.query(FaceEmbedding, Person)
        .join(Person, FaceEmbedding.person_id == Person.id)
        .all()
    )

    if not records:
        return AuthResult(
            matched=False,
            confidence=0.0,
            message="No enrolled faces in the system",
        )

    stored_embeddings: list[tuple[int, int, any]] = []
    for face_emb, person in records:
        try:
            emb_array = pickle.loads(face_emb.embedding_blob)
            stored_embeddings.append((face_emb.id, person.id, emb_array))
        except Exception as exc:
            logger.warning(
                "Failed to deserialize embedding id=%d: %s", face_emb.id, exc
            )

    if not stored_embeddings:
        return AuthResult(
            matched=False,
            confidence=0.0,
            message="No valid embeddings found in the system",
        )

    # 4. Compare against all stored embeddings
    best_emb_id, best_person_id, best_similarity = face_engine.compare_embeddings(
        query_embedding, stored_embeddings
    )

    confidence = round(best_similarity * 100, 1)

    # 5. Apply threshold
    if best_similarity >= MATCH_THRESHOLD and best_person_id is not None:
        person = db.query(Person).filter(Person.id == best_person_id).first()
        if person:
            logger.info(
                "Authentication MATCH: person_id=%d name='%s' confidence=%.1f%%",
                person.id,
                person.name,
                confidence,
            )
            return AuthResult(
                matched=True,
                person_id=person.id,
                person_name=person.name,
                person_role=person.role,
                confidence=confidence,
                message="Identity verified",
            )

    logger.info("Authentication NO MATCH: best_similarity=%.4f", best_similarity)
    return AuthResult(
        matched=False,
        confidence=confidence,
        message="No matching identity found",
    )
