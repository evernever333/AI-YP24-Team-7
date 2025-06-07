import torch
import numpy as np
import logging
from pathlib import Path
from io import BytesIO
from fastapi import APIRouter, UploadFile, HTTPException, Form
from PIL import Image
from torchvision import transforms
from logging.handlers import TimedRotatingFileHandler
from FastAPI.models import PredictResponse
from FastAPI.model_loader import (
    load_joblib_model,
    load_veg_classifier_model,
    load_vgg16_model
)

# --- Логирование ---
LOGS_DIR = Path(__file__).resolve().parent / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("inference_logger")
if not logger.hasHandlers():
    logger.setLevel(logging.INFO)
    handler = TimedRotatingFileHandler(
        filename=LOGS_DIR / "veg_api.log",
        when="midnight",
        interval=1,
        backupCount=7,
        encoding="utf-8"
    )
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# --- Роутер ---
router = APIRouter()

# --- Пути и устройство ---
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "shared" / "models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- DL трансформации ---
dl_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


@router.post("/predict", response_model=PredictResponse)
async def predict_route(
        file: UploadFile,
        selected_model: str = Form(...)
):
    if not selected_model:
        raise HTTPException(status_code=400, detail="Model ID is required.")

    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        raise HTTPException(status_code=400, detail="Файл должен быть изображением.")

    model_path = MODEL_DIR / selected_model
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model not found.")

    try:
        logger.info(f"Получен запрос для модели: {selected_model}")

        # === Исправлено: безопасное чтение изображения ===
        image_bytes = await file.read()
        img = Image.open(BytesIO(image_bytes)).convert("RGB")

        if selected_model.endswith(".joblib"):
            model = load_joblib_model(str(model_path))
            img_gray = img.convert("L").resize((64, 64))
            img_array = np.array(img_gray).astype(np.float32) / 255.0
            img_array = (img_array - 0.5) / 0.5
            input_vector = img_array.flatten().reshape(1, -1)
            if isinstance(model, tuple):
                clf, pca = model
                input_vector = pca.transform(input_vector)
                prediction = clf.predict(input_vector)
            else:
                prediction = model.predict(input_vector)

        elif selected_model.endswith("veg_cnn_final_224.pt"):
            model = load_veg_classifier_model(str(model_path), num_classes=15)
            model.eval()
            input_tensor = dl_transform(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                output = model(input_tensor)
                _, pred = torch.max(output, 1)
                prediction = [pred.item()]

        elif selected_model.endswith("veg_vgg16_final.pt"):
            model = load_vgg16_model(str(model_path), num_classes=15)
            model.eval()
            input_tensor = dl_transform(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                output = model(input_tensor)
                _, pred = torch.max(output, 1)
                prediction = [pred.item()]

        else:
            logger.error(f"Неизвестный тип модели: {selected_model}")
            raise HTTPException(status_code=400, detail="Unknown model type.")

        logger.info(f"Предсказание завершено: {prediction[0]}")
        return {"prediction": str(prediction[0])}

    except Exception as e:
        logger.exception("Ошибка при выполнении предсказания")
        raise HTTPException(status_code=500, detail=str(e))