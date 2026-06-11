import numpy as np
import cv2
from insightface.app import FaceAnalysis
import base64
from io import BytesIO
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class FaceEngine:
    def __init__(self):
        self.app = None
        self._initialized = False
    
    def initialize(self):
        """Initialize InsightFace model. Call once at startup."""
        if self._initialized:
            return
        logger.info("Loading InsightFace model (buffalo_l)...")
        self.app = FaceAnalysis(
            name='buffalo_l',
            providers=['CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self._initialized = True
        logger.info("InsightFace model loaded successfully.")
    
    def decode_base64_image(self, base64_str: str) -> np.ndarray:
        """Decode base64 image string to OpenCV numpy array (BGR)."""
        # Handle data URL prefix if present
        if ',' in base64_str:
            base64_str = base64_str.split(',', 1)[1]
        
        img_bytes = base64.b64decode(base64_str)
        img_pil = Image.open(BytesIO(img_bytes)).convert('RGB')
        img_np = np.array(img_pil)
        # Convert RGB to BGR for OpenCV/InsightFace
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        return img_bgr
    
    def detect_faces(self, image: np.ndarray) -> list:
        """Detect all faces in image. Returns list of Face objects."""
        if not self._initialized:
            raise RuntimeError("FaceEngine not initialized")
        faces = self.app.get(image)
        return faces
    
    def get_embedding(self, image: np.ndarray) -> tuple[np.ndarray | None, float, dict]:
        """Extract face embedding from image.
        
        Returns: (embedding, quality_score, metadata)
        - embedding: 512-dim numpy array, or None if no face found
        - quality_score: 0-100 based on face detection confidence and size
        - metadata: dict with bbox, det_score, etc.
        """
        faces = self.detect_faces(image)
        
        if not faces:
            return None, 0.0, {"error": "No face detected"}
        
        # Use the face with highest detection score
        best_face = max(faces, key=lambda f: f.det_score)
        
        embedding = best_face.normed_embedding
        
        # Calculate quality score
        bbox = best_face.bbox.astype(int)
        face_width = bbox[2] - bbox[0]
        face_height = bbox[3] - bbox[1]
        face_area = face_width * face_height
        img_area = image.shape[0] * image.shape[1]
        face_ratio = face_area / img_area
        
        # Quality based on: detection confidence (50%) + face size ratio (50%)
        det_quality = float(best_face.det_score) * 50
        size_quality = min(face_ratio / 0.15, 1.0) * 50  # Max quality at 15% of frame
        quality_score = round(det_quality + size_quality, 1)
        quality_score = min(quality_score, 100.0)
        
        metadata = {
            "bbox": bbox.tolist(),
            "det_score": float(best_face.det_score),
            "face_size": [int(face_width), int(face_height)],
            "num_faces_detected": len(faces)
        }
        
        return embedding, quality_score, metadata
    
    def compare_embeddings(self, query_embedding: np.ndarray, stored_embeddings: list[tuple[int, int, np.ndarray]]) -> tuple[int | None, int | None, float]:
        """Compare query embedding against stored embeddings.
        
        Args:
            query_embedding: 512-dim embedding of the query face
            stored_embeddings: list of (embedding_id, person_id, embedding_array)
        
        Returns: (best_embedding_id, best_person_id, best_similarity)
            similarity is cosine similarity, range [-1, 1], higher is better
        """
        if not stored_embeddings:
            return None, None, 0.0
        
        best_id = None
        best_person_id = None
        best_sim = -1.0
        
        for emb_id, person_id, stored_emb in stored_embeddings:
            # Cosine similarity
            sim = float(np.dot(query_embedding, stored_emb) / 
                       (np.linalg.norm(query_embedding) * np.linalg.norm(stored_emb)))
            if sim > best_sim:
                best_sim = sim
                best_id = emb_id
                best_person_id = person_id
        
        return best_id, best_person_id, best_sim


# Global singleton
face_engine = FaceEngine()
