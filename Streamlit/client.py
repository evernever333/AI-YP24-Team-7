import streamlit as st
import httpx
import asyncio
import pandas as pd
import time


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

async def train_model(file, model, model_id):
    """Отправка данных для обучения модели."""
    time.sleep(2)
    print(model)
    return {"message": "Обучение прошло успешно!", "accuracy": 98.7}

async def describe_data(file):
    pass

# Заголовок
st.title("✨ Slaaaay ML App 💅")
st.subheader("Сделай их жизнь ярче с предсказаниями 🤩")

st.sidebar.title("🎨 Навигация")
st.sidebar.write("Выбери раздел, bae 🌈")
page = st.sidebar.radio("Разделы:", ["Обучение", "Предсказание"])

if page == "Обучение":
    st.header("👑 Обучение ML-модели")
    st.write("Загрузи свой датасетик, милашка 😘")
    
    file = st.file_uploader("Выбери файл", type=["zip"])
    st.checkbox("Показать описательные статистики", disabled=(file is None))
    # if file is not None and st.checkbox("Показать описательные статистики"):
    #     data = describe_data(file)
    #     st.write(data.describe())

    st.write("Выбери метод обучения, my sun 🌈 ")
    method = st.radio("", 
                      ["SVC🕶", "LogisticRegression🌸", "RandomForestClassifier👛"])
    if method == "SVC🕶":
        st.info("SVC (Support Vector Classifier) - шик для разделения! 👠🥑")
        st.write("Напиши свои гиперпараметры 🌹")
        C = st.number_input("C", value=1.0)
        kernel = st.selectbox("Kernel", ["linear", "rbf"], index=1)
        сlass_weight = st.selectbox("Class weight", ["balanced", None])
        parameters = f"C={C}, kernel='{kernel}', class_weight='{сlass_weight}'"
    elif method == "LogisticRegression🌸":
        st.info("Logistic Regression - твой гламурный анализ 📈💋")
        st.write("Напиши свои гиперпараметры 🌹")
        C = st.number_input("C", value=1.0)
        max_iter = st.number_input("Max iter", value=100)
        parameters = f"C={C}, max_iter={max_iter}"
    elif method == "RandomForestClassifier👛":
        st.info("RandomForestClassifier👛 - разберётся во всём, как истинная королева 🌳✨")
        st.write("Напиши свои гиперпараметры 🌹")
        n_estimators = st.number_input("n_estimators", value=100)
        parameters = f"n_estimators={n_estimators}"
    st.write("Напиши своё имя и я назову модель в честь тебя 😘")
    model_id = st.text_input("ID модели", value = f"{method[:-1]}")
    if st.button("💃 Начать обучение модели", disabled=(file is None) or (model_id == "")):
        with st.spinner("✨ Обучение модели..."):
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                results = loop.run_until_complete(train_model(file, f"{method[:-1]}({parameters})", model_id))
                st.success("✅ Обучение завершено!")
                st.json(results)
            except Exception as e:
                st.error(f"⚠️ Ошибка: {str(e)}")

elif page == "Предсказание":
    st.header("🔮 Предсказания - магия данных")
    st.write("Скоро здесь будет магия ✨")