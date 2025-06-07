from fastapi import FastAPI
from FastAPI.inference import router as inference_router
import logging
import os
from logging.handlers import TimedRotatingFileHandler

# === Логирование ===
LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

logger = logging.getLogger("veg_api")
if not logger.hasHandlers():
    logger.setLevel(logging.INFO)
    handler = TimedRotatingFileHandler(
        filename=os.path.join(LOGS_DIR, "veg_api.log"),
        when="midnight",
        interval=1,
        backupCount=7,
        encoding="utf-8",
    )
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# === FastAPI-приложение ===
app = FastAPI(
    title="Vegetable Classifier API",
    description="API для предсказаний с использованием различных моделей классификации изображений овощей.",
    version="1.0.0"
)

app.include_router(inference_router, prefix="/api", tags=["Inference"])

logger.info("Vegetable Classifier API initialized")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("FastAPI.main:app", host="0.0.0.0", port=8000, reload=True)