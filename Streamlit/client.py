import os
import logging
from logging.handlers import TimedRotatingFileHandler
import streamlit as st
import asyncio
import json
import httpx
import pandas as pd
import plotly.express as px

# Папка для логов
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
LOG_FOLDER = os.path.join(CURRENT_DIR, "logs")
os.makedirs(LOG_FOLDER, exist_ok=True)

# Настройка логгера
logger = logging.getLogger("ml_app")
if not logger.hasHandlers():
    logger.setLevel(logging.INFO)
    handler = TimedRotatingFileHandler(
        filename=os.path.join(LOG_FOLDER, "ml_app.log"),
        when="midnight",  # Ротация логов каждую ночь
        interval=1,
        backupCount=7,  # Храним логи за последние 7 дней
        encoding="utf-8",
    )
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def highlight(val):
    if pd.isna(val):
        return ''
    color = '#00FA9A' if val > 0.5 else '#FFC0CB' if val > 0.3 else '#F08080'
    return f'background-color: {color}; color: black; font-weight: bold; border: 1px solid gold;'

# Интерфейс Streamlit
st.markdown(
    """
    <style>
    .stApp {
        background-image: linear-gradient(to bottom, #ffdde1, #ee9ca7); 
        background-attachment: fixed;
        color: black;
    }
    [data-testid="stSidebar"] {
        background-image: linear-gradient(to top, #EE82EE, #DDA0DD, #BA55D3); 
        color: white;
    }
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] label {
        color: white;
        font-size: 30px;
        font-weight: bold;
    }
    [data-testid="stSidebar"] [role="radiogroup"] .stRadio label { 
        color: white;
        font-size: 25px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

async def prediction(file, selected_model):
    """Предсказание модели."""
    async with httpx.AsyncClient(timeout=1000) as client:
        try:
            response = await client.post(f"{BASE_URL}/api/predict", files={"file": (file.name, file.getvalue(), file.type)}, data={"selected_model": selected_model})
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            error_detail = e.response.json().get("detail", "Ошибка на сервере")
            raise RuntimeError(f"HTTP {e.response.status_code}: {error_detail}")
        except Exception as e:
            raise RuntimeError(f"Неизвестная ошибка: {str(e)}")

BASE_URL = "http://0.0.0.0:8000"

d = {"PCA + Logistic Regression🕶": "lgbm_color.joblib", "SIFT-HOG + Random Forest🌸": "logreg_pca.joblib",\
      "Color Histogram + LightGBM👛": "rf_sift_hog.joblib", "Custom CNN (224px)✨": "veg_vgg16_final.pt",\
      "VGG16 Transfer Learning👑": "veg_cnn_final_224.pt"}

# Заголовок
st.title("✨ Slaaaay ML App 💅")
st.subheader("Сделай их жизнь ярче с предсказаниями 🤩")

st.header("🔮 Предсказания - магия данных")
st.write("Скоро здесь будет магия ✨")
st.write("Загрузи свои тестовые данные, милашка 😘")

# Загрузка файла
file = st.file_uploader("Выбери файл", type=["jpg", "png", "jpeg"])
if file is not None:
    st.write("Вот твоё изображение, queen 👑")
    st.image(file, caption="Твое загруженное изображение 💖", use_container_width=True)
selected_model = st.radio(
    "Выбери метод:",
    ["PCA + Logistic Regression🕶", "SIFT-HOG + Random Forest🌸", "Color Histogram + LightGBM👛", "Custom CNN (224px)✨", "VGG16 Transfer Learning👑"],
)
if selected_model == "PCA + Logistic Regression🕶":
    st.info("Логистическая регрессия по признакам после PCA 👠🥑")
elif selected_model == "SIFT-HOG + Random Forest🌸":
    st.info("Классическая ML-модель на фичах изображений 📈💋")
elif selected_model == "Color Histogram + LightGBM👛":
    st.info("Градиентный бустинг на гистограммах цвета 💃👛")
elif selected_model == "Custom CNN (224px)✨":
    st.info("Кастомная сверточная нейросеть (размер 224×224) 💅💋")
elif selected_model == "VGG16 Transfer Learning👑":
    st.info("Дообученная предобученная модель VGG16 🌈😘")

if st.button("💃 Начать магичить", disabled=(file is None or not selected_model)):
    with st.spinner("🔮 Выполнение предсказаний..."):
        container = st.empty()
        try:
            logger.info(f"Начато предсказание для модели {selected_model}")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(prediction(file, d[selected_model]))
            phrase = results.get("phrase", "Магия в деле, bae!")
            prediction = results.get("prediction", "🤷‍♀️ Неизвестно")
            container.success("✅ Предсказания получены!")
            st.markdown(f"### {phrase} **{prediction}** 💫")
            logger.info(f"Предсказание для модели {selected_model} успешно: {phrase} {prediction}")
        except Exception as e:
            logger.error(f"Ошибка при предсказании для модели {selected_model}: {str(e)}")
            container.error(f"⚠️ Ошибка: {str(e)}")