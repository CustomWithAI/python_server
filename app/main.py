from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import io
from PIL import Image
import numpy as np
import cv2
import json
import base64
from app.services.preprocessing import Preprocessing
from app.config import PreConfig

app = FastAPI()
preprocess = Preprocessing()

@app.get("/")
async def status():
    return {"message": "server is running"}

@app.post("/preprocess")
async def preprocessing(image: UploadFile = File(...), config: str = Form(...)):
    # Parse JSON string into a PreConfig object
    config_data = json.loads(config)
    config_obj = PreConfig(**config_data)

    # Read and convert image to RGB first
    image_bytes = await image.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Apply preprocessing
    processed_image = preprocess.preprocess(image, config_obj)

    # Convert processed image to streamable format
    _, buffer = cv2.imencode(".jpg", processed_image)
    image_stream = io.BytesIO(buffer)

    # Convert image to Base64
    image_base64 = base64.b64encode(image_stream.getvalue()).decode("utf-8")

    # Return Base64 encoded image as JSON
    return JSONResponse(content={"image_base64": image_base64})
