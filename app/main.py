from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from app.services.preprocessing import Preprocessing
import io
from PIL import Image
import numpy as np
import cv2

preprocess = Preprocessing()

app = FastAPI()


# Define the request body for training hyperparameters
class TrainRequest(BaseModel):
    epochs: int
    batch_size: int


@app.get("/")
async def status():
    return {"message": "server is running"}


@app.post("/preprocess")
async def preprocessing(image: UploadFile = File(...), config: dict = None):
    image_bytes = await image.read()
    image = Image.open(io.BytesIO(image_bytes))
    image = np.array(image)

    processed_image = preprocess.preprocess(image, config)

    _, buffer = cv2.imencode(".jpg", processed_image)
    return {"processed_image": buffer.tobytes()}
