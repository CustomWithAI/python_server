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

'''
{
    "preprocess": {...},
    "feature_extraction": {...},
    "selection": {...},
    "augmentations": {...},
    ...
}
'''
@app.get("/training")
async def training(config: str = Form(...)):
    all_config = json.loads(config)
    config_preprocess = all_config["preprocess"]
    config_feature_extraction = all_config["feature_extraction"]
    config_feature_selection = all_config["feature_selection"]
    config_augmentation = all_config["augmentation"]
    config_model = all_config["model"]
    config_training = all_config["training"]


    # TODO: Get dataset
    dataset = []

    # TODO: Preprocess images
    for image in dataset:
        image = preprocessing(image, config_preprocess)
    

    # TODO: Feature Extraction
    for image in dataset:
        image = feature_extraction(image, config_feature_extraction)

    # TODO: Feature Selection
    for image in dataset:
        image = feature_selection(image, config_feature_selection)

    # TODO: Augmentations

    # TODO: Create models

    # TODO: Train model

    # TODO: Save & Upload model
    pass
def preprocessing(image,config: str = Form(...)):
    # Parse JSON string into a PreConfig object
    config_data = json.loads(config)

    # Read and convert image to RGB first
    image_bytes = image.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

    # Apply preprocessing
    processed_image = preprocess.preprocess(image, config_data)

    # Convert processed image to streamable format
    _, buffer = cv2.imencode(".jpg", processed_image)
    image_stream = io.BytesIO(buffer)
    return StreamingResponse(image_stream, media_type="image/jpeg")


def feature_extraction(image: UploadFile = File(...), config: str = Form(...)):
    # Parse JSON string into a PreConfig object
    config_data = json.loads(config)

    # Read and convert image to RGB first
    image_bytes = image.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Apply preprocessing
    processed_image = featextraction.extract_features(image, config_data)

    # Convert processed image to streamable format
    _, buffer = cv2.imencode(".jpg", processed_image)
    image_stream = io.BytesIO(buffer)
    return StreamingResponse(image_stream, media_type="image/jpeg")

def feature_selection (image, config_data):
    pass