import torch
import joblib
from pathlib import Path
from torchvision import models
from FastAPI.models import VegClassifier

# Директория с моделями
MODELS_DIR = Path("shared/models")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Кэш моделей
model_cache = {}

def load_model(model_name: str, model_type: str, num_classes: int = None):
    """
    Загружает модель по имени и типу:
    - model_name: имя файла модели с расширением (.pt или .joblib)
    - model_type: 'sklearn', 'cnn', 'vgg'
    - num_classes: нужно указать для нейронных сетей
    """
    model_path = MODELS_DIR / model_name

    if model_path.exists() and model_name in model_cache:
        return model_cache[model_name]

    if model_type == "sklearn":
        model = joblib.load(model_path)

    elif model_type == "cnn":
        if num_classes is None:
            raise ValueError("Для 'cnn' моделей нужно указать num_classes")
        model = VegClassifier(num_classes)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

    elif model_type == "vgg":
        if num_classes is None:
            raise ValueError("Для 'vgg' моделей нужно указать num_classes")
        model = models.vgg16(pretrained=False)
        model.classifier = torch.nn.Sequential(
            torch.nn.Linear(25088, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, num_classes)
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

    else:
        raise ValueError(f"Неизвестный тип модели: {model_type}")

    model_cache[model_name] = model
    return model

# --- Обёртки для импорта в inference.py ---
def load_joblib_model(model_name: str):
    return load_model(model_name=model_name, model_type='sklearn')

def load_veg_classifier_model(model_name: str, num_classes: int):
    return load_model(model_name=model_name, model_type='cnn', num_classes=num_classes)

def load_vgg16_model(model_name: str, num_classes: int):
    return load_model(model_name=model_name, model_type='vgg', num_classes=num_classes)