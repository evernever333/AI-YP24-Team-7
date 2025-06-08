import torch
import joblib
from pathlib import Path
from torchvision import models
from FastAPI.models import VegClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_cache = {}

def load_joblib_model(model_path: str):
    path = Path(model_path)
    if path.name in model_cache:
        return model_cache[path.name]
    model = joblib.load(path)
    model_cache[path.name] = model
    return model

def load_veg_classifier_model(model_path: str, num_classes: int):
    if Path(model_path).name in model_cache:
        return model_cache[Path(model_path).name]
    model = VegClassifier(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    model_cache[Path(model_path).name] = model
    return model

def load_vgg16_model(model_path: str, num_classes: int):
    if Path(model_path).name in model_cache:
        return model_cache[Path(model_path).name]
    model = models.vgg16(pretrained=False)
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(25088, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(512, num_classes)
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    model_cache[Path(model_path).name] = model
    return model
