import uvicorn
import os
import zipfile
import shutil
import random
import json
import logging
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, Form
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

# Настройка логгирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Базовая директория для приложения
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "models")

app = FastAPI(
    title="Image Classification API",
    description="This API allows you to upload a dataset, specify model parameters, and train an image classification model.",
    version="1.0.0"
)


def load_images_and_labels(base_dir: str, category: str, sample_percentage: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    images = []
    labels = []
    path = os.path.join(base_dir, category)
    for class_name in os.listdir(path):
        class_dir = os.path.join(path, class_name)
        if not os.path.isdir(class_dir):
            continue
        all_files = os.listdir(class_dir)
        selected_files = random.sample(all_files, int(len(all_files) * sample_percentage))
        for img_name in selected_files:
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (64, 64))
                images.append(img.flatten())
                labels.append(class_name)
    return np.array(images), np.array(labels)


def get_model_from_client(model_data: dict) -> object:
    model_name = model_data["model"]
    params = model_data.get("params", {})
    if model_name == "SVC":
        return SVC(**params)
    elif model_name == "LogisticRegression":
        return LogisticRegression(**params)
    elif model_name == "RandomForestClassifier":
        return RandomForestClassifier(**params)
    else:
        raise ValueError(f"Unsupported model: {model_name}")


@app.post("/fit")
async def fit(
        file: UploadFile = File(...),
        model: str = Form(...),  # JSON с моделью и параметрами
        id: str = Form(...)  # Имя модели (идентификатор)
):
    try:
        # Создаем необходимые директории
        os.makedirs(DATASET_DIR, exist_ok=True)
        os.makedirs(MODEL_DIR, exist_ok=True)

        # 1. Распаковываем архив
        logger.info("Распаковываем архив...")
        with zipfile.ZipFile(file.file, 'r') as zip_ref:
            zip_ref.extractall(DATASET_DIR)

        # 2. Загружаем данные
        logger.info("Загружаем данные...")
        train_images, train_labels = load_images_and_labels(DATASET_DIR, "train", sample_percentage=0.15)
        test_images, test_labels = load_images_and_labels(DATASET_DIR, "test", sample_percentage=0.15)

        # 3. Применяем PCA
        logger.info("Применяем PCA...")
        pca = PCA(n_components=150, svd_solver='randomized', whiten=True, random_state=42)
        reduced_train_images = pca.fit_transform(train_images)
        reduced_test_images = pca.transform(test_images)

        # 4. Создаем модель на основе данных клиента
        logger.info("Создаем модель...")
        model_data = json.loads(model)
        classifier = get_model_from_client(model_data)

        # 5. Обучаем модель
        logger.info("Обучаем модель...")
        classifier.fit(reduced_train_images, train_labels)

        # 6. Предсказываем и оцениваем
        logger.info("Оцениваем модель...")
        y_pred = classifier.predict(reduced_test_images)
        accuracy = accuracy_score(test_labels, y_pred)

        # 7. Сохраняем модель
        logger.info("Сохраняем модель...")
        model_path = os.path.join(MODEL_DIR, f"{id}.joblib")
        dump(classifier, model_path)

        # Чистим временные файлы
        logger.info("Чистим временные файлы...")
        shutil.rmtree(DATASET_DIR, ignore_errors=True)

        return {
            "id": id,
            "accuracy": accuracy,
            "model_path": model_path
        }
    except Exception as e:
        logger.error(f"Ошибка при обучении модели: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
