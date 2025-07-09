import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '0'

from fastapi import FastAPI, UploadFile, File
import shutil
import uvicorn
from deepface import DeepFace

# ✅ PRELOAD MODEL
DeepFace.build_model("Facenet")

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

        return {
            "match": result["verified"],
            "distance": result["distance"]
        }

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
