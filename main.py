from fastapi import FastAPI, UploadFile, File
from deepface import DeepFace
import shutil
import os

app = FastAPI()

@app.get("/")
def root():
    return {"status": "Face API is running"}

@app.post("/verify-face/")
async def verify_face(new: UploadFile = File(...), registered: UploadFile = File(...)):
    try:
        with open("new.jpg", "wb") as f:
            shutil.copyfileobj(new.file, f)
        with open("registered.jpg", "wb") as f:
            shutil.copyfileobj(registered.file, f)

        result = DeepFace.verify(
            "new.jpg", "registered.jpg",
            model_name="Facenet",
            enforce_detection=True
        )

        os.remove("new.jpg")
        os.remove("registered.jpg")

        # Hitung similarity
        distance = result["distance"]
        max_distance = 0.10  # <- ubah batas maksimal jadi 0.10 (alias min similarity ~83.3%)
        similarity = max(0, min(100, (1 - (distance / max_distance)) * 100))
        is_match = distance <= max_distance  # hanya TRUE jika jarak <= 0.10

        return {
            "match": is_match,
            "distance": distance,
            "similarity": round(similarity, 2)
        }
    except Exception as e:
        return {"error": str(e)}
