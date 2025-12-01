from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Dict, Tuple
import numpy as np
import cv2
from deepface import DeepFace
import os
import pickle

MODEL_NAME = "ArcFace"
EMB_PATH = "models/embeddings.pkl"

app = FastAPI(title="Face Backend", version="1.0.0")


# ====== Base simple en disco (pickle) ======
def ensure_models_dir():
    os.makedirs("models", exist_ok=True)


def load_db() -> List[Dict]:
    ensure_models_dir()
    if not os.path.exists(EMB_PATH):
        return []
    with open(EMB_PATH, "rb") as f:
        return pickle.load(f)


def save_db(entries: List[Dict]):
    ensure_models_dir()
    with open(EMB_PATH, "wb") as f:
        pickle.dump(entries, f)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


def best_match(query_emb: np.ndarray, db: List[Dict]) -> Tuple[Dict | None, float]:
    best_entry = None
    best_sim = -1.0
    for item in db:
        sim = cosine_similarity(query_emb, item["embedding"])
        if sim > best_sim:
            best_sim = sim
            best_entry = item
    return best_entry, best_sim


def compute_embedding(image_bytes: bytes) -> np.ndarray:
    """Recibe bytes de imagen (JPG/PNG), devuelve embedding ArcFace."""
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("No se pudo decodificar la imagen")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (224, 224))

    reps = DeepFace.represent(
        img_rgb,
        model_name=MODEL_NAME,
        detector_backend="skip",  # el cliente ya manda rostro recortado
        enforce_detection=False
    )
    if not reps:
        raise ValueError("No se pudo generar embedding")

    emb = np.array(reps[0]["embedding"], dtype="float32")
    return emb


# ====== Endpoints ======

@app.post("/enroll")
async def enroll(
    person_id: str = Form(...),
    name: str = Form(...),
    image: UploadFile = File(...)
):
    """
    Enrola una persona con un rostro.
    - person_id: identificador lógico (DNI, código interno, etc.)
    - name: nombre para mostrar
    - image: imagen de rostro recortado (JPG/PNG)
    """
    try:
        img_bytes = await image.read()
        emb = compute_embedding(img_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error procesando imagen: {e}")

    db = load_db()
    db.append(
        {
            "person_id": person_id,
            "name": name,
            "embedding": emb,
        }
    )
    save_db(db)

    return {"status": "ok", "message": f"Enrolado {name}", "total_embeddings": len(db)}


@app.post("/identify")
async def identify(
    image: UploadFile = File(...)
):
    """
    Identificación 1:N:
    - recibe rostro
    - devuelve mejor match + similitud
    """
    db = load_db()
    if not db:
        raise HTTPException(status_code=400, detail="Base vacía, primero enrola personas")

    try:
        img_bytes = await image.read()
        query_emb = compute_embedding(img_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error procesando imagen: {e}")

    best_entry, best_sim = best_match(query_emb, db)

    return {
        "person_id": best_entry["person_id"] if best_entry else None,
        "name": best_entry["name"] if best_entry else None,
        "similarity": best_sim,
    }


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_NAME}
