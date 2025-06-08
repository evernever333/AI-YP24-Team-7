FROM python:3.12

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем requirements.txt из корня проекта
COPY requirements.txt /app/requirements.txt

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем только Streamlit-приложение
COPY ./Streamlit /app

# Открываем порт Streamlit
EXPOSE 8501

# Запускаем Streamlit
ENTRYPOINT ["streamlit", "run"]
CMD ["client.py"]