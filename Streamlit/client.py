import os
import logging
from logging.handlers import TimedRotatingFileHandler
import streamlit as st
import asyncio
import time

# Папка для логов
LOG_FOLDER = "logs"
os.makedirs(LOG_FOLDER, exist_ok=True)

# Настройка логгера
logger = logging.getLogger("ml_app")
logger.setLevel(logging.INFO)
handler = TimedRotatingFileHandler(
    filename=os.path.join(LOG_FOLDER, "ml_app.log"),
    when="midnight",
    interval=1,
    backupCount=7,
    encoding="utf-8",
)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


async def train_model(file, model, model_id):
    logger.info(f"Начало обучения модели {model_id} с параметрами: {model}")
    # Эмуляция долгой задачи
    for i in range(1, 6):
        await asyncio.sleep(1)  # Обновляемся каждую секунду
        logger.info(f"Процесс обучения модели {model_id}: {i * 20}% завершено")
    logger.info(f"Обучение модели {model_id} завершено успешно!")
    return {"message": f"Обучение модели {model_id} завершено!", "accuracy": 98.7}


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
page = st.sidebar.radio(" ", ["Обучение", "Предсказание"])

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

    if method == "SVC🕶":
        st.info("SVC (Support Vector Classifier) - шик для разделения! 👠🥑")
        C = st.number_input("C", value=1.0)
        kernel = st.selectbox("Kernel", ["linear", "rbf"], index=1)
        parameters = f"C={C}, kernel='{kernel}'"
    elif method == "LogisticRegression🌸":
        st.info("Logistic Regression - твой гламурный анализ 📈💋")
        C = st.number_input("C", value=1.0)
        max_iter = st.number_input("Max iter", value=500)
        parameters = f"C={C}, max_iter={max_iter}"
    elif method == "RandomForestClassifier👛":
        st.info("RandomForestClassifier👛 - разберётся во всём, как истинная королева 🌳✨")
        n_estimators = st.number_input("n_estimators", value=100)
        parameters = f"n_estimators={n_estimators}"

    model_id = st.text_input("ID модели", value=f"{method[:-1]}")

    if st.button("💃 Начать обучение модели", disabled=(file is None or not model_id)):
        container = st.empty()
        with container.container():
            st.spinner("✨ Обучение модели...")
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(train_model(file, f"{method[:-1]}({parameters})", model_id))
            container.success("✅ Обучение завершено!")
            st.json(results)
        except Exception as e:
            logger.error(f"Ошибка: {str(e)}")
            container.error(f"⚠️ Ошибка: {str(e)}")

elif page == "Предсказание":
    st.header("🔮 Предсказания - магия данных")
    st.write("Скоро здесь будет магия ✨")
