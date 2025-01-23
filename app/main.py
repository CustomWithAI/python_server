from fastapi import FastAPI, Form
import json
import os
import glob
import subprocess
from app.services.dataset.dataset import preprocess_all_dataset,augment_dataset_class,augment_dataset_obj
from app.services.model.training import MLTraining,DLTrainingPretrained,ConstructTraining
from app.services.model.construct import ConstructDL
mltraining = MLTraining()
dltrainingpretrain = DLTrainingPretrained()
constructtraining = ConstructTraining()

app = FastAPI()
@app.get("/")
async def status():
    return {"message": "server is running"}

'''
{
    "type": "classification",
    "preprocess": {...},
    "feature_extraction": {...},
    "selection": {...},
    "augmentations": {...},
    ...
}
'''
@app.post("/training-ml")
async def training_ml(config: str = Form(...)):
    config_dict = json.loads(config)
    print(config)
    mltraining.training_ml(config_dict)

@app.post("/training-dl-pretrained")
async def training_dl(config: str = Form(...)):
    config_dict = json.loads(config)
    config_model = config_dict['model']
    config_training = config_dict['training']
    print(config)
    dltrainingpretrain.train(config_model,config_training)

@app.post("/create_yolo_venv")
async def create_venv():
    subprocess.run(["python", "-m", "venv", "yolo_venv"], check=True)
    subprocess.run("bash -c 'source yolo_venv/bin/activate && pip install -r app/services/model/yolov5/requirements.txt'", shell=True, check=True)
    

@app.post("/training-yolo-pretrained")
async def training_yolo_pretrained(config: str = Form(...)):
    config_dict = json.loads(config)
    config_model = config_dict['model']
    config_training = config_dict['training']
    dltrainingpretrain.train_yolo(config_model, config_training)

@app.post("/construct-classification")
async def construct_model(config: str = Form(...)):
    config_dict = json.loads(config)
    config_model = config_dict['model']
    config_training = config_dict['training']
    constructtraining.train(config_model, config_training)

@app.post("/dataset")
async def prepare_dataset(config: str = Form(...)):
    all_config = json.loads(config)
    config_type = all_config['type']
    config_preprocess = all_config["preprocess"]
    # config_featextraction = all_config["feature_extraction"]
    # config_featselection = all_config["feature_selection"]
    config_augmentation = all_config["augmentation"]


    # TODO: Get dataset

    # TODO: Preprocess images
    if config_preprocess != {}:
        print("DOING PREPROCESS")
        dataset_dir = 'dataset'
        preprocess_all_dataset(dataset_dir, config_preprocess)

    
    # TODO: Augmentation
    if config_augmentation != {}:
        print("DOING AUGMENTATION")
        training_path = 'dataset/train'

        # Count how many training dataset exist
        image_extensions = ['*.png', '*.jpg']
        image_count = 0
        for ext in image_extensions:
            image_count += len(glob.glob(os.path.join(training_path, '**', ext), recursive=True))
        
        total_target_number = config_augmentation["number"] - image_count
        print("TOTAL TARGER:", total_target_number)

        # Do Augmentation
        if total_target_number > 0:
            if config_type == "classification":
                augment_dataset_class(training_path, config_augmentation["number"], config_augmentation)
            if config_type == "object_detection":
                augment_dataset_obj(training_path, config_augmentation["number"],config_augmentation)


    # # TODO: Feature Extraction
    # if config_featextraction != {}:
    #     for image in dataset:
    #         image = feature_extraction(image, config_featextraction)

    # # TODO: Feature Selection
    # if config_featselection != {}:
    #     for image in dataset:
    #         image = feature_selection(image, config_featextraction)
    pass

