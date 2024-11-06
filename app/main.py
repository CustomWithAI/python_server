# main.py
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

app = FastAPI()


class TrainRequest(BaseModel):
    epochs: int
    batch_size: int


@app.get("/")
async def status():
    return {"message": "server is running"}


# Run with: uvicorn main:app --reload
