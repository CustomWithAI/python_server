import os
import csv
import shutil

from typing import Literal, Optional

def clear_folder(path: str) -> None:
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path, ignore_errors=True)

def read_evaluation(filepath: str, type: Literal["txt", "csv"]) -> str:
    try:
        if type == "txt":
            with open(filepath, 'r') as file:
                lines = file.readlines()
                header = []
                values = []
                
                for line in lines:
                    if line.strip():
                        if ":" in line:
                            key, value = line.split(":")
                            header.append(key.strip())
                            values.append(value.strip())
                        else:
                            raise ValueError(f"Invalid line format in {filepath}: {line.strip()}")

                csv_output = ",".join(header) + "\n" + ",".join(values)
                return csv_output
        elif type == "csv":
            with open(filepath, newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                rows = [",".join(row) for row in reader]
                return "\n".join(rows).strip()
        else:
            raise ValueError(f"Unsupported file type: {type}")
    except FileNotFoundError:
        raise ValueError(f"File not found: {filepath}") 
    except Exception as e:
        raise ValueError(f"An error occurred reading {filepath}: {str(e)}")

def get_file_path(workflow: Literal["cls", "od", "seg"], yolo: Optional[Literal["yolov5", "yolov8", "yolov11"]] = None):
    if workflow == "cls":
        file_path = os.path.join("evaluation_results", "metrics.txt")
    elif workflow == "od":
        if yolo and yolo == "yolov5":
            file_path = os.path.join("app", "services", "model", "yolov5", "runs", "train", "exp", "results.csv")
        elif yolo:
            file_path = os.path.join("runs", "detect", "train", "results.csv")
        else:
            file_path = os.path.join("evaluation_results", "metrics.txt")
    elif workflow == "seg":
        file_path = os.path.join("runs", "segment", "train", "results.csv")
    else:
        raise ValueError(f"Invalid workflow type: '{workflow}'")
    return file_path

def get_evaluation(
    workflow: Literal["cls", "od", "seg"],
    yolo: Optional[Literal["yolov5", "yolov8", "yolov11"]] = None,
):
    file_type = "txt" if (workflow == "cls") or (workflow == "od" and yolo is None) else "csv"
    return read_evaluation(get_file_path(workflow, yolo), file_type)

def clear_evaluate_folder():
    clear_folder(os.path.join("evaluation_results"))