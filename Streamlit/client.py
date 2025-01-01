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

# получение eda
async def eda(file):
    """Обучение модели."""
    async with httpx.AsyncClient(timeout=1000) as client:
        try:
            response = await client.post(f"{BASE_URL}/eda", files={"file": (file.name, file.getvalue(), file.type)})
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            error_detail = e.response.json().get("detail", "Ошибка на сервере")
            raise RuntimeError(f"HTTP {e.response.status_code}: {error_detail}")
        except Exception as e:
            raise RuntimeError(f"Неизвестная ошибка: {str(e)}")

# обучение модели
async def train_model(config):
    """Обучение модели."""
    async with httpx.AsyncClient(timeout=1000) as client:
        try:
            response = await client.post(f"{BASE_URL}/fit", data={"model_data": config})
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            error_detail = e.response.json().get("detail", "Ошибка на сервере")
            raise RuntimeError(f"HTTP {e.response.status_code}: {error_detail}")
        except Exception as e:
            raise RuntimeError(f"Неизвестная ошибка: {str(e)}")


# предсказание модели
async def prediction(file, model_id):
    """Предсказание модели."""
    async with httpx.AsyncClient(timeout=1000) as client:
        try:
            response = await client.post(f"{BASE_URL}/predict", files={"file": (file.name, file.getvalue(), file.type)}, data={"model_id": model_id})
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            error_detail = e.response.json().get("detail", "Ошибка на сервере")
            raise RuntimeError(f"HTTP {e.response.status_code}: {error_detail}")
        except Exception as e:
            raise RuntimeError(f"Неизвестная ошибка: {str(e)}")
        
# список моделей
async def list_models():
    """Получение списка моделей."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{BASE_URL}/list_models")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            error_detail = e.response.json().get("detail", "Ошибка на сервере")
            raise RuntimeError(f"HTTP {e.response.status_code}: {error_detail}")
        except Exception as e:
            raise RuntimeError(f"Неизвестная ошибка: {str(e)}")

# удаление всех моделей
async def remove_all_models():
    """Удаление всех моделей."""
    async with httpx.AsyncClient(timeout=1000) as client:
        try:
            response = await client.delete(f"{BASE_URL}/remove_all")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            error_detail = e.response.json().get("detail", "Ошибка на сервере")
            raise RuntimeError(f"HTTP {e.response.status_code}: {error_detail}")
        except Exception as e:
            raise RuntimeError(f"Неизвестная ошибка: {str(e)}")

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

BASE_URL = "http://0.0.0.0:8000"

# Заголовок
st.title("✨ Slaaaay ML App 💅")
st.subheader("Сделай их жизнь ярче с предсказаниями 🤩")

st.sidebar.title("🎨 Навигация")
st.sidebar.write("Выбери раздел, bae 🌈")
page = st.sidebar.radio(" ", ["EDA", "Обучение", "Предсказание", "Список моделей", "Удаление моделей"])

if page == "EDA":
    st.header("📊 EDA ML-модели")
    st.write("Загрузи свой датасетик, милашка 😘")

    # Загрузка файла
    file = st.file_uploader("Выбери файл", type=["zip"])

    if st.button("💃 Начать получение EDA", disabled = file is None):
        with st.spinner("✨ Получение EDA..."):
            container = st.empty()
            try:
                logger.info(f"Начато получение EDA")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                results = loop.run_until_complete(eda(file))
                st.success(results["message"])
                st.markdown("### 🚂 Распределение классов в Train:")
                train_class_dist = results["train_class_dist"]
                train_df = pd.DataFrame(
                    list(train_class_dist.items()), 
                    columns=["Класс", "Количество"]
                )
                train_df.index += 1  # Начинаем индексацию с 1
                st.dataframe(train_df)

                st.markdown("### 🧪 Распределение классов в Test:")
                test_class_dist = results["test_class_dist"]
                test_df = pd.DataFrame(
                    list(test_class_dist.items()), 
                    columns=["Класс", "Количество"]
                )
                test_df.index += 1  # Начинаем индексацию с 1
                st.dataframe(test_df)
                
                st.markdown("### 🖼️ Визуализация результата:")
                st.image(results["anomaly_image"], caption="✨ Результат EDA", use_container_width=True)

                logger.info(f"Закончено получение EDA")
            except Exception as e:
                logger.error(f"Ошибка при получении eda: {str(e)}")
                container.error(f"⚠️ Ошибка: {str(e)}")

if page == "Обучение":
    st.header("👑 Обучение ML-модели")

    # Выбор метода обучения
    method = st.radio(
        "Выбери метод:",
        ["SVC🕶", "LogisticRegression🌸", "RandomForestClassifier👛"],
    )

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

    if st.button("💃 Начать обучение модели", disabled = not model_id):
        with st.spinner("✨ Обучение модели..."):
            container = st.empty()
            try:
                config = json.dumps({
                    "model": method[:-1],
                    "params": params,
                    "model_id": model_id,
                })
                logger.info(f"Начато обучение модели с ID: {model_id}, метод: {method[:-1]}, параметры: {params}")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                results = loop.run_until_complete(train_model(config))
                container.success("✅ Обучение завершено!")
                st.markdown(
                    f"""
                    ### 🌟 Результаты обучения:
                    - **ID модели:** `{results['id']}`
                    - **Точность:** `{results['accuracy']:.2%}`
                    """
                )
                df = pd.DataFrame(results['report']).T.reset_index()
                df = df.rename(columns={'index': 'Class'})
                st.title("💅 Slaaaay Таблица Результатов")
                st.write("Взгляни на эту красотку таблицу ✨")
                st.dataframe(
                    df.style.applymap(highlight, subset=['precision', 'recall', 'f1-score'])
                )
                st.subheader("🎨 Визуализация")

                for metric in ['precision', 'recall', 'f1-score']:
                    fig = px.pie(df[:-2], values=metric, names='Class', title=f"{metric.title()} Распределение 💖")
                    fig.update_traces(textinfo='percent+label', pull=[0.05]*len(df))
                    st.plotly_chart(fig)
                logger.info(f"Обучение модели {model_id} завершено успешно. Результаты: {results}")
            except Exception as e:
                logger.error(f"Ошибка при обучении модели {model_id}: {str(e)}")
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
        with st.spinner("🔮 Выполнение предсказаний..."):
            container = st.empty()
            try:
                logger.info(f"Начато предсказание для модели {model_id}")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                results = loop.run_until_complete(prediction(file, model_id))
                phrase = results.get("phrase", "Магия в деле, bae!")
                prediction = results.get("prediction", "🤷‍♀️ Неизвестно")
                container.success("✅ Предсказания получены!")
                st.markdown(f"### {phrase} **{prediction}** 💫")
                logger.info(f"Предсказание для модели {model_id} успешно: {phrase} {prediction}")
            except Exception as e:
                logger.error(f"Ошибка при предсказании для модели {model_id}: {str(e)}")
                container.error(f"⚠️ Ошибка: {str(e)}")

elif page == "Список моделей":
    st.header("📂 Список моделей")
    if st.button("📋 Получить список моделей"):
        with st.spinner("📂 Получение списка моделей..."):
            try:
                logger.info("Запрос списка моделей")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                models = loop.run_until_complete(list_models())
                if models:
                    st.markdown("### 📋 Список моделей:")
                    for model in models:
                        st.markdown(f"- **ID:** `{model['id']}`")
                    st.success("✅ Список моделей получен!")
                    logger.info(f"Список моделей получен: {models}")
                else:
                    st.info("Пока нет моделей в системе. Загрузи данные для обучения! 💖")
                    logger.info(f"На данный момент нет моделей")
            except Exception as e:
                logger.error(f"Ошибка при получении списка моделей: {str(e)}")
                st.error(f"⚠️ Ошибка: {str(e)}")

elif page == "Удаление моделей":
    st.header("🗑️ Удаление всех моделей")
    if st.button("❌ Удалить все модели"):
        with st.spinner("🗑️ Удаление моделей..."):
            try:
                logger.info("Запрос на удаление всех моделей")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                results = loop.run_until_complete(remove_all_models())
                if results:
                    st.markdown("### ❌ Удалено:")
                    for result in results:
                        st.markdown(f"- **ID:** `{result['id']}`")
                    st.success("✅ Все модели удалены!")
                    logger.info(f"Все модели успешно удалены: {results}")
                else:
                    st.info("Пока нет моделей в системе. Загрузи данные для обучения! 💖")
                    logger.info(f"На данный момент нет моделей")
            except Exception as e:
                logger.error(f"Ошибка при удалении всех моделей: {str(e)}")
                st.error(f"⚠️ Ошибка: {str(e)}")

