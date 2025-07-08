from fastapi import FastAPI, UploadFile, File
from deepface import DeepFace
import shutil
import os

app = FastAPI()

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
