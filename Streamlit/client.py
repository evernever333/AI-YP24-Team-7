import streamlit as st
import httpx
import asyncio
import itertools
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

async def train_model(file, model):
    time.sleep(10)
    return {"message": "Обучение прошло успешно!", "accuracy": 98.7}

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
    st.write("Выбери метод обучения, my sun 🌈 ")
    method = st.radio("", 
                      ["SVC🕶", "LinearRegression🌸", "Tree👛"])
    if method == "SVC🕶":
        st.info("SVC (Support Vector Classifier) - шик для разделения! 👠🥑")
    elif method == "LinearRegression🌸":
        st.info("Linear Regression - твой гламурный анализ 📈💋")
    elif method == "Tree👛":
        st.info("Tree - разберётся во всём, как истинная королева 🌳✨")
    st.write("Напиши своё имя и я назову модель в честь тебя 😘")
    model_id = st.text_input("ID модели", value = f"{method[:-1]}")
    if st.button("💃 Начать обучение модели", disabled=(file is None) or (model_id == "")):
        with st.spinner("✨ Обучение модели..."):
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                results = loop.run_until_complete(train_model(file, method))
                st.success("✅ Обучение завершено!")
                st.json(results)
            except Exception as e:
                st.error(f"⚠️ Ошибка: {str(e)}")

elif page == "Предсказание":
    st.header("🔮 Предсказания - магия данных")
    st.write("Скоро здесь будет магия ✨")