import torch
import numpy as np
import logging
from pathlib import Path
from io import BytesIO
from fastapi import APIRouter, UploadFile, HTTPException, Form
from PIL import Image
from logging.handlers import TimedRotatingFileHandler
from FastAPI.models import PredictResponse
from FastAPI.model_loader import (
    load_joblib_model,
    load_veg_classifier_model,
    load_vgg16_model
)
from FastAPI.utils import (
    extract_color_features,
    extract_sift_hog_features,
    prepare_gray_image,
    prepare_dl_tensor
)

# --- Классы ---
CLASS_NAMES_EN_ML = [
    "Bottle_Gourd", "Papaya", "Radish", "Cauliflower", "Bitter_Gourd",
    "Carrot", "Pumpkin", "Cabbage", "Brinjal", "Capsicum", "Tomato",
    "Cucumber", "Potato", "Broccoli", "Bean"
]

CLASS_NAMES_RU_ML = [
    "Бутылочная тыква", "Папайя", "Редис", "Цветная капуста", "Горькая тыква",
    "Морковь", "Тыква", "Капуста", "Баклажан", "Болгарский перец", "Помидор",
    "Огурец", "Картофель", "Брокколи", "Фасоль"
]

CLASS_NAMES_EN_DL = sorted(CLASS_NAMES_EN_ML)
CLASS_NAMES_RU_DL = [CLASS_NAMES_RU_ML[CLASS_NAMES_EN_ML.index(name)] for name in CLASS_NAMES_EN_DL]

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

# --- Обработчики по имени модели ---
def get_model_handlers(model_path: Path):
    return {
        "logreg_pca.joblib": lambda img: (
            lambda clf, pca: clf.predict(pca.transform(prepare_gray_image(img)))
        )(*load_joblib_model(str(model_path))),

        "rf_sift_hog.joblib": lambda img: load_joblib_model(str(model_path)).predict(
            extract_sift_hog_features(img)
        ),

        "lgbm_color.joblib": lambda img: load_joblib_model(str(model_path)).predict(
            extract_color_features(img)
        ),

        "veg_cnn_final_224.pt": lambda img: (
            lambda model: torch.argmax(
                model(prepare_dl_tensor(img, DEVICE)), dim=1
            ).item()
        )(load_veg_classifier_model(str(model_path), num_classes=15)),

        "veg_vgg16_final.pt": lambda img: (
            lambda model: torch.argmax(
                model(prepare_dl_tensor(img, DEVICE)), dim=1
            ).item()
        )(load_vgg16_model(str(model_path), num_classes=15)),
    }

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

    handlers = get_model_handlers(model_path)
    if selected_model not in handlers:
        raise HTTPException(status_code=400, detail="Unknown model type.")

    try:
        logger.info(f"Получен запрос для модели: {selected_model}")

        image_bytes = await file.read()
        img = Image.open(BytesIO(image_bytes)).convert("RGB")

        result = handlers[selected_model](img)

        # --- Распаковка предсказания, если это список/массив из одного элемента ---
        if isinstance(result, (list, np.ndarray)) and len(result) == 1:
            result = result[0]

        # --- Преобразование индекса или строки в понятное название ---
        if isinstance(result, (np.integer, int)):
            index = int(result)
            if selected_model in ["veg_cnn_final_224.pt", "veg_vgg16_final.pt"]:
                prediction = f"{CLASS_NAMES_RU_DL[index]} ({CLASS_NAMES_EN_DL[index]})"
            else:
                prediction = f"{CLASS_NAMES_RU_ML[index]} ({CLASS_NAMES_EN_ML[index]})"
        elif isinstance(result, str):
            try:
                if selected_model in ["veg_cnn_final_224.pt", "veg_vgg16_final.pt"]:
                    index = CLASS_NAMES_EN_DL.index(result)
                    prediction = f"{CLASS_NAMES_RU_DL[index]} ({result})"
                else:
                    index = CLASS_NAMES_EN_ML.index(result)
                    prediction = f"{CLASS_NAMES_RU_ML[index]} ({result})"
            except ValueError:
                prediction = result
        else:
            prediction = str(result)

        logger.info(f"Предсказание завершено: {prediction}")
        return {"prediction": prediction}

    except Exception as e:
        logger.exception("Ошибка при выполнении предсказания")
        raise HTTPException(status_code=500, detail=str(e))
