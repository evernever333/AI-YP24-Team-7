import os
import zipfile
import random
import json
import numpy as np
import cv2
import shutil
import logging
import matplotlib.pyplot as plt
import base64
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
from collections import Counter
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# Создаем экземпляр приложения FastAPI
app = FastAPI(
    title="Image Classification API",
    description="API для загрузки датасета, указания параметров модели и обучения модели классификации изображений.",
    version="1.0.0"
)

# Модель данных для передачи информации о модели
class ModelData(BaseModel):
    model: str = Field(..., description="Название модели, например, 'SVC'.")
    params: Dict = Field(default_factory=dict, description="Параметры для модели.")
    model_id: str = Field(..., description="Уникальный идентификатор для сохранения модели.")


# Модель для представления информации о сохраненной модели
class Model(BaseModel):
    id: str

# Определяем пути к директориям
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
SHARED_DIR = os.path.join(PARENT_DIR, "shared")
LOGS_DIR = os.path.join(CURRENT_DIR, "logs")

# Директории для хранения датасетов, моделей и временных предсказаний
datasets_dir = os.path.join(SHARED_DIR, "datasets")
models_dir = os.path.join(SHARED_DIR, "models")
temp_predict_dir = os.path.join(SHARED_DIR, "temp_predict")

# Создаем необходимые директории, если они еще не существуют
os.makedirs(SHARED_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(datasets_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(temp_predict_dir, exist_ok=True)

# Настраиваем логирование
logger = logging.getLogger("ml_app")
if not logger.hasHandlers():
    logger.setLevel(logging.INFO)
    handler = TimedRotatingFileHandler(
        filename=os.path.join(LOGS_DIR, "ml_app.log"),
        when="midnight",
        interval=1,
        backupCount=7,
        encoding="utf-8",
    )
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Функция для загрузки изображений и меток из указанной директории
def load_images_and_labels(base_dir: str, category: str, sample_percentage: float = 1.00) -> Tuple[np.ndarray, np.ndarray]:
    images, labels = [], []

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
            break

    if not images or not labels:
        raise ValueError(f"Категория {category} пуста или содержит некорректные данные.")

    return np.array(images), np.array(labels)

# Функция для создания экземпляра модели на основе переданных данных от клиента
def get_model_from_client(model_data: Dict) -> Union[SVC, LogisticRegression, RandomForestClassifier]:
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
        raise ValueError(f"Неизвестная модель: {model_name}")

@app.post("/eda")
async def eda(
        file: UploadFile = File(...),
) -> Dict:
    """
    Маршрут для выполнения Exploratory Data Analysis (EDA) над загруженным датасетом.

    :param file: Загруженный файл архива с данными.
    :return: Словарь с результатами анализа, включая распределение классов и график аномалий.
    """
    try:
        logger.info("Начали выполнение EDA")

        if os.listdir(datasets_dir):
            shutil.rmtree(datasets_dir)
            os.makedirs(datasets_dir, exist_ok=True)

        zip_path = os.path.join(SHARED_DIR, file.filename)
        with open(zip_path, "wb") as buffer:
            buffer.write(await file.read())

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(datasets_dir)

        train_images, train_labels = load_images_and_labels(datasets_dir, "train", sample_percentage=1.00)
        test_images, test_labels = load_images_and_labels(datasets_dir, "test", sample_percentage=1.00)

        train_counter = Counter(train_labels)
        test_counter = Counter(test_labels)

        # Функция для анализа аномалий в изображениях
        def analyze_anomalies(images):
            mean_brightness = np.mean(images, axis=1)
            std_brightness = np.std(images, axis=1)

            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            ax[0].boxplot(mean_brightness)
            ax[0].set_title("Mean Brightness")

            ax[1].boxplot(std_brightness)
            ax[1].set_title("Brightness Standard Deviation")

            return fig

        anomaly_fig = analyze_anomalies(train_images)
        anomaly_buffer = BytesIO()
        anomaly_fig.savefig(anomaly_buffer, format="png")
        anomaly_buffer.seek(0)
        encoded_anomaly_img = base64.b64encode(anomaly_buffer.read()).decode("utf-8")
        plt.close(anomaly_fig)

        logger.info("EDA завершено успешно")

        # Сохраняем результаты для дальнейшего использования в обучении
        np.save(os.path.join(datasets_dir, "train_images.npy"), train_images)
        np.save(os.path.join(datasets_dir, "train_labels.npy"), train_labels)
        np.save(os.path.join(datasets_dir, "test_images.npy"), test_images)
        np.save(os.path.join(datasets_dir, "test_labels.npy"), test_labels)

        return {
            "message": "EDA выполнен успешно!",
            "train_class_dist": dict(train_counter),
            "test_class_dist": dict(test_counter),
            "anomaly_image": f"data:image/png;base64,{encoded_anomaly_img}"
        }

    except Exception as e:
        logger.error(f"Ошибка при выполнении EDA: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/fit")
async def fit(
        model_data: str = Form(...),
) -> Dict:
    """
    Маршрут для обучения модели на основании загруженных данных.

    :param model_data: JSON строка, содержащая информацию о модели, её параметрах и уникальном идентификаторе.
    :return: Словарь с информацией об обученной модели, включая точность и отчёт о классификации.
    """
    try:
        # Проверяем наличие результатов EDA
        train_images_path = os.path.join(datasets_dir, "train_images.npy")
        train_labels_path = os.path.join(datasets_dir, "train_labels.npy")
        test_images_path = os.path.join(datasets_dir, "test_images.npy")
        test_labels_path = os.path.join(datasets_dir, "test_labels.npy")

        if not all(os.path.exists(path) for path in [train_images_path, train_labels_path, test_images_path, test_labels_path]):
            raise HTTPException(status_code=400, detail="Данные не подготовлены. Выполните шаг EDA перед обучением.")

        # Загружаем данные
        train_images = np.load(train_images_path)
        train_labels = np.load(train_labels_path)
        test_images = np.load(test_images_path)
        test_labels = np.load(test_labels_path)

        try:
            # Парсим JSON строку и валидируем данные
            model_data = json.loads(model_data)
            validated_model_data = ModelData(**model_data)
        except (json.JSONDecodeError, ValidationError) as e:
            raise HTTPException(status_code=400, detail=f"Ошибка валидации JSON: {e}")

        # Используем PCA для уменьшения размерности данных
        pca = PCA(n_components=150, svd_solver='randomized', whiten=True, random_state=42)

        # Функция обучения модели
        def train_model():
            # Применяем PCA к данным
            reduced_train_images = pca.fit_transform(train_images)
            reduced_test_images = pca.transform(test_images)

            # Создаём классификатор на основе переданной модели и её параметров
            classifier = get_model_from_client(validated_model_data.dict())
            logger.info("Начали обучение модели")
            classifier.fit(reduced_train_images, train_labels)

            # Прогнозируем результаты на тестовых данных
            y_pred = classifier.predict(reduced_test_images)
            accuracy = accuracy_score(test_labels, y_pred)

            # Составляем отчет о классификации
            report = classification_report(test_labels, y_pred, output_dict=True)

            # Сохраняем модель и PCA вместе
            model_id = validated_model_data.model_id
            model_path = os.path.join(models_dir, f"{model_id}.joblib")
            dump((classifier, pca), model_path)

            logger.info(f"Модель {model_id} успешно обучена с точностью {accuracy}.")
            return {
                "id": model_id,
                "accuracy": accuracy,
                "report": report
            }

        # Обучаем модель асинхронно с таймаутом 1 минута
        with ThreadPoolExecutor() as executor:
            future = executor.submit(train_model)
            try:
                result = future.result(timeout=60)
            except TimeoutError:
                logger.error("Обучение модели заняло больше 1 минуты и было прервано.")
                raise HTTPException(status_code=500, detail="Процесс обучения занял больше 1 минуты и был прерван.")

        return result

    except Exception as e:
        logger.error(f"Ошибка при обучении модели: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict(
        file: UploadFile = File(...),
        model_id: str = Form(...)
) -> Dict[str, Union[str, float]]:
    """
    Маршрут для предсказания класса изображения с использованием сохраненной модели.

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
    """
    Маршрут для получения списка всех сохраненных моделей.

    :return: Список объектов Model, представляющих каждую сохранённую модель.
    """
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
    """
    Маршрут для удаления всех моделей, данных и временных файлов.

    :return: Список удалённых моделей.
    """
    logger.info("Удаление моделей и данных...")
    model_files = [f for f in os.listdir(models_dir) if f.endswith(".joblib")]
    models = [Model(id=os.path.splitext(file)[0]) for file in model_files]

    try:

        # Очистка данных
        shutil.rmtree(models_dir)
        shutil.rmtree(temp_predict_dir)
        shutil.rmtree(datasets_dir)
        for item in os.listdir(SHARED_DIR):
            item_path = os.path.join(SHARED_DIR, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)

        #Заново создаем директории
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(temp_predict_dir, exist_ok=True)
        os.makedirs(datasets_dir, exist_ok=True)

        logger.info("Все модели, данные и временные файлы удалены.")
    except Exception as e:
        logger.error(f"Ошибка при удалении: {e}")
        raise HTTPException(status_code=500, detail="Не удалось очистить директории.")

    return models

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

