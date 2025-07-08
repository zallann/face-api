import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress TensorFlow logs
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '0'  # prevent OpenCV image decode crash

from fastapi import FastAPI, UploadFile, File
from deepface import DeepFace
import shutil

app = FastAPI()

# ✅ Health check endpoint to prevent Railway auto-crash
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

        result = DeepFace.verify("new.jpg", "registered.jpg", enforce_detection=True)

        os.remove("new.jpg")
        os.remove("registered.jpg")

        return {
            "match": result["verified"],
            "distance": result["distance"]
        }
    except Exception as e:
        return {"error": str(e)}
