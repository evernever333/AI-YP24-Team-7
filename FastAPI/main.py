import os
import zipfile
import shutil
import random
import json
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, Form
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

app = FastAPI(
    title="Image Classification API",
    description="API for uploading a dataset, specifying model parameters, and training an image classification model.",
    version="1.0.0"
)

# Путь к общей папке
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shared"))

def load_images_and_labels(base_dir: str, category: str, sample_percentage: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    """
    Загрузка изображений и меток из указанной категории.

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
        id: str = Form(...)  # Имя модели (идентификатор)
):
    """
    Эндпоинт для тренировки модели на загруженном наборе данных.

    :param file: Архив с данными для тренировки.
    :param model: JSON с описанием модели и её параметров.
    :param id: Идентификатор модели.
    :return: Словарь с результатами тренировки.
    """
    try:
        # 1. Распаковываем архив в общую папку
        os.makedirs(BASE_DIR, exist_ok=True)

        zip_path = os.path.join(BASE_DIR, file.filename)
        with open(zip_path, "wb") as buffer:
            buffer.write(await file.read())

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(BASE_DIR)

        # 2. Загружаем данные
        train_images, train_labels = load_images_and_labels(BASE_DIR, "train", sample_percentage=0.15)
        test_images, test_labels = load_images_and_labels(BASE_DIR, "test", sample_percentage=0.15)

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
        model_dir = os.path.join(BASE_DIR, "models")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{id}.joblib")
        dump(classifier, model_path)

        # Чистим временные файлы
        for item in os.listdir(BASE_DIR):
            item_path = os.path.join(BASE_DIR, item)
            if os.path.isfile(item_path) or item_path.endswith(file.filename):
                os.remove(item_path)
            elif os.path.isdir(item_path) and item not in ["models"]:
                shutil.rmtree(item_path, ignore_errors=True)

        return {
            "id": id,
            "accuracy": accuracy,
            "model_path": model_path
        }
    except Exception as e:
        print(f"Ошибка при обучении модели: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
