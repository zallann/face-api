import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '0'

from fastapi import FastAPI, UploadFile, File
from deepface import DeepFace
import shutil

app = FastAPI()
model = None  # Lazy-load
THRESHOLD = 0.1667  # 83.3%

@app.get("/")
def root():
    return {"status": "Face API is running"}

@app.post("/verify-face/")
async def verify_face(new: UploadFile = File(...), registered: UploadFile = File(...)):
    global model

    try:
        # Load model pertama kali saat request, bukan saat startup
        if model is None:
            model = DeepFace.build_model("Facenet")

        with open("new.jpg", "wb") as f:
            shutil.copyfileobj(new.file, f)
        with open("registered.jpg", "wb") as f:
            shutil.copyfileobj(registered.file, f)

        result = DeepFace.verify(
            "new.jpg", "registered.jpg",
            model_name="Facenet",
            model=model,
            enforce_detection=True
        )

        os.remove("new.jpg")
        os.remove("registered.jpg")

        distance = result["distance"]
        similarity = round((1 - distance) * 100, 2)
        match = distance <= THRESHOLD

        return {
            "match": match,
            "distance": distance,
            "similarity_percent": similarity
        }
    except Exception as e:
        return {"error": str(e)}
