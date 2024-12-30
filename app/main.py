from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
import io
from PIL import Image
import numpy as np
import cv2
import json
import base64
from app.services.preprocessing import Preprocessing
from app.services.featextraction import FeatureExtraction

app = FastAPI()
preprocess = Preprocessing()
featextraction = FeatureExtraction()

@app.get("/")
async def status():
    return {"message": "server is running"}

@app.post("/preprocess")
async def preprocessing(image: UploadFile = File(...), config: str = Form(...)):
    # Parse JSON string into a PreConfig object
    config_data = json.loads(config)

    # Read and convert image to RGB first
    image_bytes = await image.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Apply preprocessing
    processed_image = preprocess.preprocess(image, config_data)

    # Convert processed image to streamable format
    _, buffer = cv2.imencode(".jpg", processed_image)
    image_stream = io.BytesIO(buffer)
    return StreamingResponse(image_stream, media_type="image/jpeg")

@app.post("/featextract")
async def featextract(image: UploadFile = File(...), config: str = Form(...)):
    # Parse JSON string into a PreConfig object
    config_data = json.loads(config)

    # Read and convert image to RGB first
    image_bytes = await image.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Apply preprocessing
    processed_image = featextraction.extract_features(image, config_data)

    # Convert processed image to streamable format
    _, buffer = cv2.imencode(".jpg", processed_image)
    image_stream = io.BytesIO(buffer)
    return StreamingResponse(image_stream, media_type="image/jpeg")
