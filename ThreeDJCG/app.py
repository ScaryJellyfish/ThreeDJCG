from uuid import uuid4
from pathlib import Path
from typing import Optional, List

import aiofiles
import torch
from fastapi import FastAPI, UploadFile, Query
from PIL import Image

from .model import MMEntityLinking

app = FastAPI()
model: Optional[MMEntityLinking] = None


def get_model() -> MMEntityLinking:
    global model

    if model is None:
        model = MMEntityLinking()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()

    return model


@app.get("/")
async def root():
    return {"message": "Hello World"}


async def save_upload_file(upload_file: UploadFile) -> str:
    file_path = f"uploaded_images/{uuid4()}{Path(upload_file.filename).suffix}"
    async with aiofiles.open(file_path, "wb") as f:
        while content := await upload_file.read(4 * 1024):
            await f.write(content)

    return file_path


@app.post("/inference")
async def inference(
    image: UploadFile, texts: List[str] = Query(...)
):
    image_path = await save_upload_file(image)
    image = Image.open(image_path).convert("RGB")

    model = get_model()
    probs = model.inference(image, texts)

    return {"probs": probs}