import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from skimage.feature import hog
import cv2

# === Обработка изображений ===

def prepare_gray_image(img: Image.Image, size=(64, 64)) -> np.ndarray:
    """
    Подготовка grayscale изображения (resize, нормализация, векторизация)
    Используется для моделей: PCA+LogReg, SIFT+HOG, RF.
    """
    img = img.convert("L").resize(size)
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = (img_array - 0.5) / 0.5
    return img_array.flatten().reshape(1, -1)

def extract_sift_hog_features(img: Image.Image, size=(64, 64)) -> np.ndarray:
    """
    Извлечение SIFT + HOG признаков из grayscale изображения.
    Используется для моделей: RandomForest.
    """
    img = img.convert("L").resize(size)
    img_gray = np.array(img).astype(np.uint8)
    sift = cv2.SIFT_create()
    _, sift_desc = sift.detectAndCompute(img_gray, None)
    sift_vec = sift_desc.flatten()[:128] if sift_desc is not None else np.zeros(128)
    hog_vec = hog(img_gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
    return np.concatenate([sift_vec, hog_vec]).reshape(1, -1)

def extract_color_features(img: Image.Image, size=(64, 64)) -> np.ndarray:
    """
    Извлечение цветовой гистограммы + яркость.
    Используется для модели LGBM.
    """
    img = img.convert("RGB").resize(size)
    img_rgb = np.array(img)
    hist_features = []
    for i in range(3):
        hist = cv2.calcHist([img_rgb], [i], None, [16], [0, 256]).flatten()
        hist_features.extend(hist)
    mean_brightness = np.mean(img_rgb)
    return np.array(hist_features + [mean_brightness]).reshape(1, -1)

# === DL трансформации и подготовка тензора ===
dl_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def prepare_dl_tensor(img: Image.Image, device: torch.device) -> torch.Tensor:
    """
    Преобразует RGB-изображение в нормализованный тензор.
    Используется для моделей CNN и VGG.
    """
    return dl_transform(img).unsqueeze(0).to(device)