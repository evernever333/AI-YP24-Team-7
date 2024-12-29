import os
import logging
from logging.handlers import TimedRotatingFileHandler
import streamlit as st
import asyncio
import json
import httpx

# Папка для логов
LOG_FOLDER = "logs"
os.makedirs(LOG_FOLDER, exist_ok=True)

# Настройка логгера
logger = logging.getLogger("ml_app")
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


# обучение модели
async def train_model(file, config, update_container):
    async with httpx.AsyncClient(timeout=1000) as client:
        response = await client.post(f"{BASE_URL}/fit", files={"file": (file.name, file.getvalue(), file.type)}, data={"model": json.dumps(config)})
        response.raise_for_status()
        return response.json()


# предсказание модели
async def prediction(file, model_id, update_container):
    async with httpx.AsyncClient(timeout=1000) as client:
        response = await client.post(f"{BASE_URL}/predict", files={"file": (file.name, file.getvalue(), file.type)}, data={"model_id": model_id})
        response.raise_for_status()
        return response.json()

async def list_models():
    """Получение списка моделей."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/list_models")
        response.raise_for_status()
        return response.json()

async def remove_all_models():
    """Удаление всех моделей."""
    async with httpx.AsyncClient() as client:
        response = await client.delete(f"{BASE_URL}/remove_all")
        response.raise_for_status()
        return response.json()

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

BASE_URL = "http://127.0.0.1:8000"

# Заголовок
st.title("✨ Slaaaay ML App 💅")
st.subheader("Сделай их жизнь ярче с предсказаниями 🤩")

st.sidebar.title("🎨 Навигация")
st.sidebar.write("Выбери раздел, bae 🌈")
page = st.sidebar.radio(" ", ["Обучение", "Предсказание", "Список моделей", "Удаление моделей"])

if page == "Обучение":
    st.header("👑 Обучение ML-модели")
    st.write("Загрузи свой датасетик, милашка 😘")

    # Загрузка файла
    file = st.file_uploader("Выбери файл", type=["zip"])

    # Выбор метода обучения
    method = st.radio(
        "Выбери метод:",
        ["SVC🕶", "LogisticRegression🌸", "RandomForestClassifier👛"],
    )
    st.checkbox("Показать описательные статистики", disabled=file is None)

    # Настройка параметров модели
    params = {}
    if method == "SVC🕶":
        st.info("SVC (Support Vector Classifier) - шик для разделения! 👠🥑")
        params["C"] = st.number_input("C", value=1.0)
        params["kernel"] = st.selectbox("Kernel", ["linear", "rbf"], index=1)
        params["class_weight"] = st.selectbox("Class weight", ["None", "balanced"], index=1)
    elif method == "LogisticRegression🌸":
        st.info("Logistic Regression - твой гламурный анализ 📈💋")
        params["C"] = st.number_input("C", value=1.0)
        params["max_iter"] = st.number_input("Max iter", value=500)
    elif method == "RandomForestClassifier👛":
        st.info("RandomForestClassifier👛 - разберётся во всём, как истинная королева 🌳✨")
        params["n_estimators"] = st.number_input("n_estimators", value=100)

    st.write("Напиши своё имя и я назову модель в честь тебя 💋")
    model_id = st.text_input("ID модели", value=f"{method[:-1]}")

    if st.button("💃 Начать обучение модели", disabled=(file is None or not model_id)):
        container = st.empty()
        try:
            config = {
                "model": method[:-1],
                "params": params,
                "model_id": model_id,
            }
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(train_model(file, config, container))
            container.success("✅ Обучение завершено!")
            st.json(results)
        except Exception as e:
            logger.error(f"Ошибка: {str(e)}")
            container.error(f"⚠️ Ошибка: {str(e)}")


elif page == "Предсказание":
    st.header("🔮 Предсказания - магия данных")
    st.write("Скоро здесь будет магия ✨")
    st.write("Загрузи свои тестовые данные, милашка 😘")

    # Загрузка файла
    file = st.file_uploader("Выбери файл", type=["jpg", "png", "jpeg"])
    if file is not None:
        st.write("Вот твоё изображение, queen 👑")
        st.image(file, caption="Твое загруженное изображение 💖", use_container_width=True)
    
    st.write("Напиши имя модели 💋")
    model_id = st.text_input("ID модели", value="SVC")
    if st.button("💃 Начать магичить", disabled=(file is None or not model_id)):
        container = st.empty()
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(prediction(file, model_id, container))
            container.success("✅ Предсказания получены!")
            st.json(results)
        except Exception as e:
            logger.error(f"Ошибка: {str(e)}")
            container.error(f"⚠️ Ошибка: {str(e)}")

elif page == "Список моделей":
    st.header("📂 Список моделей")
    if st.button("📋 Получить список моделей"):
        with st.spinner("📂 Получение списка моделей..."):
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                models = loop.run_until_complete(list_models())
                st.success("✅ Список моделей получен!")
                st.json(models)
            except Exception as e:
                st.error(f"⚠️ Ошибка: {str(e)}")

elif page == "Удаление моделей":
    st.header("🗑️ Удаление всех моделей")
    if st.button("❌ Удалить все модели"):
        with st.spinner("🗑️ Удаление моделей..."):
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                results = loop.run_until_complete(remove_all_models())
                st.success("✅ Все модели удалены!")
                st.json(results)
            except Exception as e:
                st.error(f"⚠️ Ошибка: {str(e)}")

