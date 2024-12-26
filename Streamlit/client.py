import streamlit as st
import httpx
import asyncio

# Шикарный CSS стиль для фона и элементов
st.markdown(
    """
    <style>
    /* Фон для всего приложения */
    .stApp {
        background-image: linear-gradient(to bottom, #ffdde1, #ee9ca7); /* розовый */
        background-attachment: fixed;
        color: black;
    }

    /* Сайдбар стиль */
    [data-testid="stSidebar"] {
        background-image: linear-gradient(to top, #EE82EE, #DDA0DD, #BA55D3); /* Гламурный сиреневый градиент */
        color: white;
    }

    /* Заголовки в сайдбаре */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] label {
        color: white;
        font-size: 30px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

page = f'''
<style>
div[data-testid="stRadio"] label {{
        color: white !important; 
        font-size: 18px !important; 
        font-family: "Comic Sans MS", cursive !important;
    }}
</style>
'''
st.markdown(page, unsafe_allow_html = True)

BASE_URL = "http://127.0.0.1:8000"

async def train_model(data):
    """Отправка данных для обучения модели."""
    async with httpx.AsyncClient(timeout=1000) as client:
        response = await client.post(f"{BASE_URL}/fit", json=data)
        response.raise_for_status()
        return response.json()

# Заголовок
st.title("✨ Slaaaay ML App 💅")
st.subheader("Сделай их жизнь ярче с предсказаниями 🤩")

# Навигация
st.sidebar.title("🎨 Навигация")
page = st.sidebar.radio("Выбери раздел, bae 🌈", ["Обучение", "Предсказание"])

if page == "Обучение":
    st.header("👑 Обучение ML-модели")
    st.write("Загрузи свой датасетик, милашка 😘")
    
    # Загружаем датасет
    dataset = st.file_uploader("Выбери файл", type=["zip"])
    
    # Выбор метода
    method = st.radio("Выбери метод обучения: ", 
                      ["SVC!", "LinearRegression!", "Tree!"], label_visibility="visible")
    
    # Подсказки
    if method == "SVC!":
        st.info("SVC (Support Vector Classifier) - шик для разделения! 👠🥑")
    elif method == "LinearRegression!":
        st.info("Linear Regression - твой гламурный анализ 📈💋")
    elif method == "Tree!":
        st.info("Tree - разберётся во всём, как истинная королева 🌳✨")

    if st.button("💃 Начать обучение модели"):
        with st.spinner("✨ Обучение моделей..."):
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    results = loop.run_until_complete(train_model(dataset, method))
                    st.success("✅ Обучение завершено!")
                    st.json(results)
                except Exception as e:
                    st.error(f"⚠️ Ошибка: {str(e)}")
elif page == "Предсказание":
    st.header("🔮 Предсказания - магия данных")
    st.write("Скоро здесь будет магия ✨")
