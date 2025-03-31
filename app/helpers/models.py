import os
import shutil

from typing import Literal, Optional
from fastapi.responses import FileResponse
from fastapi import HTTPException

def delete_path(path: str, silent: bool = True) -> None:
    """Delete a file or directory at the given path."""
    if os.path.isfile(path):
        os.remove(path)
    elif os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    elif not silent:
        raise FileNotFoundError(f"No such file or directory: '{path}'")

def clear_folder(path: str) -> None:
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path, ignore_errors=True)

def get_file_response(path: str, filename: str = None) -> FileResponse:
    """Return a FileResponse for a file at the given path."""
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path,
        filename=filename or os.path.basename(path),
        media_type='application/octet-stream'
    )

def get_model_path(
    workflow: Literal["cls", "od", "seg"],
    model: Literal["ml", "construct", "pt"],
    yolo: Optional[Literal["yolov5", "yolov8", "yolov11"]] = None,
):
    """Return a file path for the specified model type."""
    if workflow == "cls":
        if model == "ml":
            return os.path.join("ml_model.pkl")
        elif model == "construct" or model == "pt":
            return os.path.join("model.h5")
        else:
            raise ValueError(f"Invalid model type for classification workflow: '{model}'")
    elif workflow == "od":
        if model == "construct":
            return os.path.join("model.h5")
        elif model == "pt":
            if yolo and yolo == "yolov5":
                return os.path.join("app", "services", "model", "yolov5", "runs", "train", "exp", "weights", "best.pt")
            else:
                return os.path.join("runs", "detect", "train", "weights", "best.pt")
        else:
            raise ValueError(f"Invalid model type for object detection workflow: '{model}'")
    elif workflow == "seg":
        if model == "pt":
            return os.path.join("runs", "segment", "train", "weights", "best.pt")
        else:
            raise ValueError(f"Invalid model type for semantic segmentation workflow: '{model}'")
    else:
        raise ValueError(f"Invalid workflow type: '{workflow}'")

def get_model(
    workflow: Literal["cls", "od", "seg"],
    model: Literal["ml", "construct", "pt"],
    yolo: Optional[Literal["yolov5", "yolov8", "yolov11"]] = None,
):
    """Return a FileResponse for the specified model type."""
    return get_file_response(get_model_path(workflow, model, yolo))

def delete_model(
    workflow: Literal["cls", "od", "seg"],
    model: Literal["ml", "construct", "pt"],
    yolo: Optional[Literal["yolov5", "yolov8", "yolov11"]] = None,
):
    """Delete the model directory for the specified model type."""
    delete_path(get_model_path(workflow, model, yolo))

def delete_all_models():
    """Delete all model directories."""
    delete_path(os.path.join("ml_model.pkl"), silent=True)
    delete_path(os.path.join("model.h5"), silent=True)
    clear_folder(os.path.join("app", "services", "model", "yolov5", "runs", "train"))
    clear_folder(os.path.join("runs"))