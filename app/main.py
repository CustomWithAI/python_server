# Import libraries
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import io
from PIL import Image
import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict
import json

# Import services
from app.services.preprocessing import Preprocessing

preprocess = Preprocessing()

# Import cofiguration file
from app.config import PreConfig


app = FastAPI()


@app.get("/")
async def status():
    return {"message": "server is running"}


@app.post("/preprocess")
async def preprocessing(image: UploadFile = File(...), config: str = Form(...)):
    # Parse JSON string into a PreConfig object
    config_data = json.loads(config)
    config_obj = PreConfig(**config_data)

    image_bytes = await image.read()
    image = Image.open(io.BytesIO(image_bytes))
    image = np.array(image)

    processed_image = preprocess.preprocess(image, config_obj)

    _, buffer = cv2.imencode(".jpg", processed_image)
    image_stream = io.BytesIO(buffer)
    return StreamingResponse(image_stream, media_type="image/jpeg")
