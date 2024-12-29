import os
import zipfile
import random
import json
import numpy as np
import cv2
import shutil
import logging
from logging.handlers import TimedRotatingFileHandler
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load
from pydantic import BaseModel, ValidationError, Field
from typing import List, Dict, Tuple, Union
import matplotlib.pyplot as plt
from collections import Counter
from io import BytesIO
import base64

app = FastAPI(
    title="Image Classification API",
    description="API for uploading a dataset, specifying model parameters, and training an image classification model.",
    version="1.0.0"
)

class ModelData(BaseModel):
    model: str = Field(..., description="Название модели, например, 'SVC'.")
    params: Dict = Field(default_factory=dict, description="Параметры для модели.")
    model_id: str = Field(..., description="Уникальный идентификатор для сохранения модели.")

class Model(BaseModel):
    id: str

# Базовая директория, где будет все храниться
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
SHARED_DIR = os.path.join(PARENT_DIR, "shared")
LOGS_DIR = os.path.join(CURRENT_DIR, "logs")

# Создание папок, если они отсутствуют
os.makedirs(SHARED_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Настройка логгера
logger = logging.getLogger("ml_app")
if not logger.hasHandlers():
    logger.setLevel(logging.INFO)
    handler = TimedRotatingFileHandler(
        filename=os.path.join(LOGS_DIR, "ml_app.log"),
        when="midnight",  # Ротация логов каждую ночь
        interval=1,
        backupCount=7,  # Храним логи за последние 7 дней
        encoding="utf-8",
    )
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

BASE_DIR = SHARED_DIR

# Переименуем папку dataset в datasets
datasets_dir = os.path.join(BASE_DIR, "datasets")
os.makedirs(datasets_dir, exist_ok=True)

# Папка для моделей
models_dir = os.path.join(BASE_DIR, "models")
os.makedirs(models_dir, exist_ok=True)

# Папка для временных файлов
temp_predict_dir = os.path.join(BASE_DIR, "temp_predict")
os.makedirs(temp_predict_dir, exist_ok=True)

def load_images_and_labels(base_dir: str, category: str, sample_percentage: float = 0.15) -> Tuple[np.ndarray, np.ndarray]:
    """
    Загрузка изображений и меток из указанной категории, игнорируя промежуточные папки.
    :param base_dir: Базовая директория, содержащая папки с изображениями.
    :param category: Категория ("train", "test", "validation").
    :param sample_percentage: Процент выборки изображений для загрузки.
    :return: кортеж из массива изображений и соответствующих им меток.
    """
    images: List[np.ndarray] = []
    labels: List[str] = []

    # Ищем папку категории на любом уровне вложенности
    for root, _, _ in os.walk(base_dir):
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

    if not images or not labels:
        raise ValueError(f"Категория {category} пуста или содержит некорректные данные.")

    return np.array(images), np.array(labels)

def get_model_from_client(model_data: Dict) -> Union[SVC, LogisticRegression, RandomForestClassifier]:
    """
    Создание классификатора на основе данных, полученных от клиента.
    :param model_data: Словарь с именем модели и её параметрами.
    :return: Объект классификатора.
    """
    model_name = model_data.get("model")
    if not model_name:
        raise ValueError("Модель не указана.")

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
        model_data: str = Form(...),
) -> Dict:
    """
    Эндпоинт для тренировки модели на загруженном наборе данных.
    :param file: Архив с данными для тренировки.
    :param model_data: JSON с описанием модели и её параметров.
    :return: Словарь с результатами тренировки.
    """
    try:
        # Проверка входных данных
        if not file.filename.endswith(".zip"):
            raise HTTPException(status_code=400, detail="Файл должен быть в формате .zip")

        # Валидация JSON данных модели
        try:
            model_data = json.loads(model_data)
            validated_model_data = ModelData(**model_data)
        except (json.JSONDecodeError, ValidationError) as e:
            raise HTTPException(status_code=400, detail=f"Ошибка валидации JSON: {e}")

        # 1. Распаковываем архив в общую папку
        dataset_dir = datasets_dir
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
        try:
            classifier = get_model_from_client(validated_model_data.dict())
        except Exception:
            raise HTTPException(status_code=400, detail="Некорректный формат модели.")

        logger.info(f"Начали обучение")

        # 5. Обучаем модель
        classifier.fit(reduced_train_images, train_labels)

        # 6. Предсказываем и оцениваем
        y_pred = classifier.predict(reduced_test_images)
        accuracy = accuracy_score(test_labels, y_pred)

        # 7. Создаем отчет об обучении
        report = classification_report(test_labels, y_pred, output_dict=True)

        # 8. Сохраняем модель
        model_id = validated_model_data.model_id
        model_path = os.path.join(models_dir, f"{model_id}.joblib")
        dump((classifier, pca), model_path)

        # Логируем успешное обучение
        logger.info(f"Модель {model_id} успешно обучена с точностью {accuracy}.")

        return {
            "id": model_id,
            "accuracy": accuracy,
            "report": report
            }
    except Exception as e:
        logger.info(f"Ошибка при обучении модели: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict(
        file: UploadFile = File(...),
        model_id: str = Form(...)
) -> Dict[str, Union[str, float]]:
    """
    Эндпоинт для предсказания класса изображения с использованием сохраненной модели.
    :param file: Изображение для классификации.
    :param model_id: Идентификатор модели.
    :return: Предсказанный класс изображения.
    """
    try:
        # Проверка входных данных
        if not model_id:
            raise HTTPException(status_code=400, detail="Идентификатор модели не указан.")

        if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
            raise HTTPException(status_code=400, detail="Файл должен быть изображением (.png, .jpg, .jpeg).")

        # 1. Сохраняем загруженное изображение во временную папку
        file_path = os.path.join(temp_predict_dir, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        # 2. Загружаем изображение
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise HTTPException(status_code=400, detail="Не удалось загрузить изображение. Проверьте путь или формат файла.")
        img = cv2.resize(img, (64, 64))
        img_flat = img.flatten().reshape(1, -1)

        # 3. Загружаем модель и PCA
        model_path = os.path.join(models_dir, f"{model_id}.joblib")
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"Модель с ID '{model_id}' не найдена.")

        classifier, pca = load(model_path)

        # 4. Преобразуем изображение через PCA
        reduced_img = pca.transform(img_flat)

        logger.info(f"Начали предсказывать")
        # 5. Предсказываем класс
        prediction = classifier.predict(reduced_img)

        # Список фраз
        phrases = [
            "Ох ты ж, это же...",
            "Выглядит как...",
            "Я думаю это...",
            "К гадалке не ходи, это...",
            "Держите меня семеро, это...",
            "Встречайте!...",
            "Тысяча чертей, вот и..."
        ]

        selected_phrase = random.choice(phrases)

        # Логируем успешное предсказание
        logger.info(f"Предсказание успешно произведено {prediction[0]}.")

        return {
            "phrase": selected_phrase,
            "prediction": prediction[0],
            }
    except Exception as e:
        logger.info(f"Ошибка при предсказании: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/list_models", response_model=List[Model])
async def list_models() -> List[Model]:
    logger.info(f"Получение моделей...")
    models_path = os.path.join(SHARED_DIR, "models")  # Путь к папке моделей
    if not os.path.exists(models_path):
        return []
    model_files = [f for f in os.listdir(models_path) if f.endswith(".joblib")]
    models = [Model(id=os.path.splitext(file)[0]) for file in model_files]
    logger.info("Список моделей получен.")
    return models

@app.delete("/remove_all", response_model=List[Model])
async def remove_all() -> List[Model]:
    logger.info(f"Удаление моделей...")
    models_path = os.path.join(SHARED_DIR)
    model_files = [f for f in os.listdir(f'{models_path}/models') if f.endswith(".joblib")]
    models = [Model(id=os.path.splitext(file)[0]) for file in model_files]
    if not os.path.exists(models_path):
        logger.info("Папка пуста")
        return []
    shutil.rmtree(models_path)
    os.mkdir(models_path)
    logger.info("Все данные и модели удалены.")
    return models

@app.post("/eda")
async def eda(
        file: UploadFile = File(...),
) -> Dict:
    """
    Анализ Exploratory Data Analysis (EDA) на основе загруженного архива изображений.
    """
    try:
        logger.info("Начали получение EDA")
        # 1. Сохраняем zip-файл и распаковываем
        zip_path = os.path.join(BASE_DIR, file.filename)
        with open(zip_path, "wb") as buffer:
            buffer.write(await file.read())

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(datasets_dir)

        # 2. Загружаем изображения из train и test
        train_images, train_labels = load_images_and_labels(datasets_dir, "train", sample_percentage=1.0)
        test_images, test_labels = load_images_and_labels(datasets_dir, "test", sample_percentage=1.0)

        # 3. Анализ распределения классов
        train_counter = Counter(train_labels)
        test_counter = Counter(test_labels)

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].bar(train_counter.keys(), train_counter.values(), color='skyblue')
        ax[0].set_title("Train Class Distribution")
        ax[0].set_xticks(list(range(len(train_counter.keys()))))
        ax[0].set_xticklabels(train_counter.keys(), rotation=45)

        ax[1].bar(test_counter.keys(), test_counter.values(), color='salmon')
        ax[1].set_title("Test Class Distribution")
        ax[1].set_xticks(list(range(len(test_counter.keys()))))
        ax[1].set_xticklabels(test_counter.keys(), rotation=45)

        plt.tight_layout()

        # 4. Пример изображений (первые 5)
        fig2, axs = plt.subplots(1, 5, figsize=(15, 5))
        for i in range(5):
            axs[i].imshow(train_images[i].reshape(64, 64), cmap='gray')
            axs[i].set_title(train_labels[i])
            axs[i].axis('off')
        plt.suptitle("Sample Images", fontsize=16)

        # 5. Преобразуем графики в base64 для отправки
        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        encoded_img = base64.b64encode(buffer.read()).decode("utf-8")
        plt.close()
        logger.info("Получение EDA закончено")
        return {
            "message": "EDA выполнен успешно!",
            "train_class_dist": dict(train_counter),
            "test_class_dist": dict(test_counter),
            "image": f"data:image/png;base64,{encoded_img}"
        }

    except Exception as e:
        logger.error(f"Ошибка при EDA: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)