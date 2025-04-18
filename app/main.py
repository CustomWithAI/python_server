from fastapi import FastAPI, Form, File, UploadFile, HTTPException
import os
import glob
import subprocess

from typing import Annotated
from app.services.dataset.dataset import preprocess_all_dataset, augment_dataset_class, augment_dataset_obj, augment_dataset_seg, prepare_dataset
from app.services.model.training import MLTraining, DLTrainingPretrained, ConstructTraining
from app.models.ml import MachineLearningClassificationRequest
from app.models.dl import (
    DeepLearningClassification,
    DeepLearningYoloRequest,
    DeepLearningClassificationConstruct,
    DeepLearningObjectDetectionConstructFeatex,
    DeepLearningObjectDetectionConstructRequest,
)
from app.models.dataset import DatasetConfigRequest, PrepareDatasetRequest
from app.models.use_model import UseModelRequest
from app.services.model.use_model import UseModel
from app.helpers.models import delete_all_models, get_model

ml_training = MLTraining()
dl_training_pretrained = DLTrainingPretrained()
construct_training = ConstructTraining()

app = FastAPI()


@app.get("/")
async def status():
    return {"message": "server is running"}


@app.post("/training-ml")
async def training_ml(config: MachineLearningClassificationRequest):
    ml_training.training_ml_cls(config)
    return get_model("cls", "ml")


@app.post("/training-dl-cls-pt")
async def training_dl(config: DeepLearningClassification):
    dl_training_pretrained.train_cls(config)
    return get_model("cls", "pt")


@app.post("/training-dl-cls-construct")
async def construct_model(config: DeepLearningClassificationConstruct):
    if config.featex:
        construct_training.train_cls_featex(config)
    else:
        construct_training.train_cls(config)
    return get_model("cls", "construct")

@app.post("/training-dl-od-construct")
async def construct_model(config: DeepLearningObjectDetectionConstructRequest):
    if isinstance(config, DeepLearningObjectDetectionConstructFeatex):
        construct_training.train_od_featex(config)
    else:
        construct_training.train_od(config)
    return get_model("od", "construct")


@app.post("/create_yolo_venv")
async def create_venv():
    # Create venv for yolov5
    subprocess.run(["python", "-m", "venv", "yolo5_venv"], check=True)
    subprocess.run(
        "yolo5_venv/bin/python -m pip install -r app/services/model/yolov5/requirements.txt", shell=True, check=True)

    # Create venv for yolov8
    subprocess.run(["python", "-m", "venv", "yolov8_venv"], check=True)
    subprocess.run(
        "yolov8_venv/bin/python -m pip install ultralytics==8.2.103 -q", shell=True, check=True)

    # Create venv for yolov11
    subprocess.run(["python", "-m", "venv", "yolov11_venv"], check=True)
    subprocess.run(
        'yolov11_venv/bin/python -m pip install "ultralytics<=8.3.40" supervision roboflow', shell=True, check=True)

    return {"message": "Venvs created successfully"}


@app.post("/training-yolo-pt")
async def training_yolo_pretrained(config: DeepLearningYoloRequest):
    dl_training_pretrained.train_yolo(config)
    if config.type == "object_detection":
        return get_model("od", "pt", config.model)
    elif config.type == "segmentation":
        return get_model("seg", "pt", config.model)
    else:
        raise ValueError("Invalid type for deep learning workflow")


@app.post("/dataset")
async def create_dataset(data: PrepareDatasetRequest):
    delete_all_models()
    prepare_dataset(data)


@app.post("/dataset-config")
async def config_dataset(config: DatasetConfigRequest):
    # TODO: Get dataset

    # TODO: Preprocess images
    if config.preprocess:
        print("DOING PREPROCESS")
        dataset_dir = "dataset"
        preprocess_all_dataset(dataset_dir, config.preprocess)

    # TODO: Augmentation
    if config.augmentation and config.type:
        print("DOING AUGMENTATION")
        training_path = "dataset/train"

        # Count how many training dataset exist
        image_extensions = ['*.png', '*.jpg']
        image_count = 0
        for ext in image_extensions:
            image_count += len(glob.glob(os.path.join(training_path,
                               '**', ext), recursive=True))

        total_target_number = config.augmentation.number - image_count
        print("TOTAL TARGER:", total_target_number)

        # Do Augmentation
        if total_target_number > 0:
            if config.type == "classification":
                augment_dataset_class(
                    training_path, config.augmentation)
            if config.type == "object_detection":
                augment_dataset_obj(
                    training_path, config.augmentation)
            if config.type == "segmentation":
                augment_dataset_seg(
                    training_path, config.augmentation)


@app.post("/use-model")
async def use_all_model(
    type: Annotated[str, Form(...)],
    img: UploadFile = File(...),
    model: UploadFile = File(...),
    version: str | None = Form(None),
):
    payload = UseModelRequest(type=type, version=version)
    image_bytes = await img.read()
    model_bytes = await model.read()

    use_model = UseModel(model_bytes=model_bytes)

    if payload.type == "ml":
        prediction = use_model.use_ml(image_bytes)
        return {"prediction": prediction.tolist()}

    if payload.type == "dl_cls":
        prediction = use_model.use_dl_cls(image_bytes)
        return {"prediction": int(prediction)}

    if payload.type == "dl_od_pt":
        prediction = use_model.use_dl_od_pt(image_bytes, payload.version)
        return {"prediction": prediction}

    if payload.type == "dl_od_con":
        prediction = use_model.use_dl_od_con(image_bytes)
        return {"prediction": prediction}

    if payload.type == "dl_seg":
        prediction = use_model.use_dl_seg(image_bytes, payload.version)
        return {"prediction": prediction}

    raise HTTPException(400, "Invalid Model Type")
