import os
import zipfile
import shutil
import random
import json
import numpy as np
import cv2
import base64
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load
from pydantic import BaseModel
from typing import List, Dict

app = FastAPI(
    title="Image Classification API",
    description="API for uploading a dataset, specifying model parameters, and training an image classification model.",
    version="1.0.0"
)

class SuccessResponse(BaseModel):
    message: str

class Model(BaseModel):
    id: str

class Models(BaseModel):
    models: List[Model] = []

models_db = Models()

# Базовая директория, где будет все храниться
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
SHARED_DIR = os.path.join(CURRENT_DIR, "shared")
LOGS_DIR = os.path.join(CURRENT_DIR, "logs")

# Создание папок, если они отсутствуют
os.makedirs(SHARED_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

BASE_DIR = SHARED_DIR
LOG_FILE = os.path.join(LOGS_DIR, "ml_app.log")

def load_images_and_labels(base_dir: str, category: str, sample_percentage: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    """
    Загрузка изображений и меток из указанной категории, игнорируя промежуточные папки.
    :param base_dir: Базовая директория, содержащая папки с изображениями.
    :param category: Категория ("train", "test", "validation").
    :param sample_percentage: Процент выборки изображений для загрузки.
    :return: кортеж из массива изображений и соответствующих им меток.
    """
    images = []
    labels = []

    # Ищем папку категории на любом уровне вложенности
    for root, dirs, files in os.walk(base_dir):
        if os.path.basename(root) == category:
            for class_name in os.listdir(root):
                class_dir = os.path.join(root, class_name)
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
            break  # Останавливаем поиск, если нужная папка найдена

    return np.array(images), np.array(labels)

def get_model_from_client(model_data: dict) -> object:
    """
    Создание классификатора на основе данных, полученных от клиента.
    :param model_data: Словарь с именем модели и её параметрами.
    :return: Объект классификатора.
    """
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
):
    """
    Эндпоинт для тренировки модели на загруженном наборе данных.
    :param file: Архив с данными для тренировки.
    :param model: JSON с описанием модели и её параметров.
    :return: Словарь с результатами тренировки.
    """
    try:
        # 1. Распаковываем архив в общую папку
        dataset_dir = os.path.join(BASE_DIR, "dataset")
        os.makedirs(dataset_dir, exist_ok=True)

        zip_path = os.path.join(BASE_DIR, file.filename)
        with open(zip_path, "wb") as buffer:
            buffer.write(await file.read())

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_dir)

        # 2. Загружаем данные
        train_images, train_labels = load_images_and_labels(dataset_dir, "train", sample_percentage=0.15)
        test_images, test_labels = load_images_and_labels(dataset_dir, "test", sample_percentage=0.15)

        # 3. Применяем PCA
        pca = PCA(n_components=150, svd_solver='randomized', whiten=True, random_state=42)
        reduced_train_images = pca.fit_transform(train_images)
        reduced_test_images = pca.transform(test_images)

        # 4. Создаем модель на основе данных клиента
        model_data = json.loads(model)
        classifier = get_model_from_client(model_data)

        # 5. Обучаем модель
        classifier.fit(reduced_train_images, train_labels)

        # 6. Предсказываем и оцениваем
        y_pred = classifier.predict(reduced_test_images)
        accuracy = accuracy_score(test_labels, y_pred)

        # 7. Сохраняем модель
        model_id = model_data["model_id"]
        model_dir = os.path.join(BASE_DIR, "models")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{model_id}.joblib")
        dump((classifier, pca), model_path)
        existing_model = next((model for model in models_db.models if model.id == model_id), None)
        if existing_model == None:
            models_db.models.append(Model(id=model_id))
        # Чистим временные файлы
        shutil.rmtree(dataset_dir, ignore_errors=True)
        os.remove(zip_path)

        return {
            "id": model_id,
            "accuracy": accuracy,
            "model_path": model_path
        }
    except Exception as e:
        print(f"Ошибка при обучении модели: {e}")
        return {"error": str(e)}

@app.post("/predict")
async def predict(
        file: UploadFile = File(...),
        model_id: str = Form(...)
):
    """
    Эндпоинт для предсказания класса изображения с использованием сохраненной модели.
    :param file: Изображение для классификации.
    :param model_id: Идентификатор модели.
    :return: Предсказанный класс изображения.
    """
    try:
        # 1. Сохраняем загруженное изображение во временную папку
        temp_dir = os.path.join(BASE_DIR, "temp_predict")
        os.makedirs(temp_dir, exist_ok=True)

        file_path = os.path.join(temp_dir, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        # 2. Загружаем изображение
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Не удалось загрузить изображение. Проверьте путь или формат файла.")
        img_output = cv2.resize(img, (224, 224))
        img = cv2.resize(img, (64, 64))
        img_flat = img.flatten().reshape(1, -1)

        # 3. Загружаем модель и PCA
        model_path = os.path.join(BASE_DIR, "models", f"{model_id}.joblib")
        if not os.path.exists(model_path):
            raise ValueError(f"Модель с ID '{model_id}' не найдена.")
        classifier, pca = load(model_path)

        # 4. Преобразуем изображение через PCA
        reduced_img = pca.transform(img_flat)

        # 5. Предсказываем класс
        prediction = classifier.predict(reduced_img)

        # Чистим временные файлы
        os.remove(file_path)

        # Список фраз
        phrases = [
            "Ох ты ж, это же...",
            "Выглядит как...",
            "Я думаю это...",
            "К гадалке не ходи, это...",
            "Держите меня семеро, это...",
            "Встречайте!",
            "Разыскивается."
        ]

        selected_phrase = random.choice(phrases)

        # Конвертируем изображение в base64 строку
        success, encoded_img = cv2.imencode('.jpg', img_output)
        if not success:
            raise ValueError("Не удалось закодировать изображение.")
        img_base64 = base64.b64encode(encoded_img).decode('utf-8')

        return {
            "phrase": selected_phrase,
            "prediction": prediction[0],
            "image": f"data:image/jpeg;base64,{img_base64}"  # Передаем изображение в формате data URI
        }
    except Exception as e:
        print(f"Ошибка при предсказании: {e}")
        return JSONResponse({"error": str(e)})

@app.get("/list_models", response_model=List[Model])
async def list_models():
    return models_db.models

@app.delete("/remove_all", response_model=List[SuccessResponse])
async def remove_all():
    responses = []
    models_path = os.path.join(SHARED_DIR, "models")
    for model in models_db.models:
        responses.append(SuccessResponse(message=f"Model '{model.id}' removed"))
        model_file = os.path.join(models_path, f"{model.id}.joblib")
        os.remove(model_file)
    models_db.models = []
    return responses

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)