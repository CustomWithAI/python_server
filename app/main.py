from fastapi import FastAPI, Form, UploadFile
import json
import os
import glob
import subprocess

from typing import List
from app.services.dataset.dataset import preprocess_all_dataset, augment_dataset_class, augment_dataset_obj, augment_dataset_seg
from app.services.model.training import MLTraining, DLTrainingPretrained, ConstructTraining
from app.models.ml import MachineLearningClassificationRequest
from app.models.dl import (
    DeepLearningClassification,
    DeepLearningYoloRequest,
    DeepLearningClassificationConstruct,
    DeepLearningObjectDetectionConstructFeatex,
    DeepLearningObjectDetectionConstructRequest,
)
from app.models.dataset import DatasetConfig
from app.services.model.use_model import UseModel

mltraining = MLTraining()
dltrainingpretrain = DLTrainingPretrained()
constructtraining = ConstructTraining()

app = FastAPI()


@app.get("/")
async def status():
    return {"message": "server is running"}


@app.post("/training-ml")
async def training_ml(config: MachineLearningClassificationRequest):
    mltraining.training_ml_cls(config)


@app.post("/training-dl-cls-pt")
async def training_dl(config: DeepLearningClassification):
    dltrainingpretrain.train_cls(config)


@app.post("/training-dl-cls-construct")
async def construct_model(config: DeepLearningClassificationConstruct):
    if config.featex:
        constructtraining.train_cls_featex(config)
    else:
        constructtraining.train_cls(config)


@app.post("/training-dl-od-construct")
async def construct_model(config: DeepLearningObjectDetectionConstructRequest):
    if isinstance(config, DeepLearningObjectDetectionConstructFeatex):
        constructtraining.train_od_featex(config)
    else:
        constructtraining.train_od(config)


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
    dltrainingpretrain.train_yolo(config)


@app.post("/dataset")
async def create_dataset(images: List[UploadFile]):
    # TODO: Create dataset
    return {
        "message": "Dataset created successfully"
    }


@app.post("/dataset-config")
async def prepare_dataset(config: DatasetConfig):
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
async def use_all_model(img: UploadFile, config: str = Form(...)):
    convert_config = json.loads(config)
    model_type = convert_config["model"]
    if model_type == "dl_od_pt" or model_type == "dl_seg":
        version = convert_config["version"]
    convert_img = await img.read()

    usemodel = UseModel()

    if model_type == "ml":
        prediction = usemodel.use_ml(convert_img)
        return {"prediction": prediction.tolist()}

    if model_type == "dl_cls":
        prediction = usemodel.use_dl_cls(convert_img)
        return {"prediction": int(prediction)}

    if model_type == "dl_od":
        prediction = usemodel.use_dl_od_pt(convert_img, version)
        return {
            "prediction": prediction
        }

    if model_type == "dl_od_con":
        prediction = usemodel.use_dl_od_con(convert_img)
        return {
            "prediction": prediction
        }

    if model_type == "dl_seg":
        prediction = usemodel.use_dl_seg(convert_img, version)
        return {
            "prediction": prediction
        }

    return {"error": "Invalid model type"}
