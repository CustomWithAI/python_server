from fastapi import FastAPI, Form
import json
import os
import glob
from app.services.dataset.dataset import preprocess_all_dataset,augment_dataset
from app.services.model.training import training

app = FastAPI()

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
@app.post("/training-ml")
async def training_ml(config: str = Form(...)):
    config_dict = json.loads(config)
    print(config)
    training(config_dict)


@app.post("/dataset")
async def prepare_dataset(config: str = Form(...)):
    all_config = json.loads(config)
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
        
        # Calculate target number per class
        total_target_number = config_augmentation["number"] - image_count

        # Do Augmentation
        if total_target_number > 0:
            augment_dataset(training_path, config_augmentation["number"], config_augmentation)

    # # TODO: Feature Extraction
    # if config_featextraction != {}:
    #     for image in dataset:
    #         image = feature_extraction(image, config_featextraction)

    # # TODO: Feature Selection
    # if config_featselection != {}:
    #     for image in dataset:
    #         image = feature_selection(image, config_featextraction)
    pass

