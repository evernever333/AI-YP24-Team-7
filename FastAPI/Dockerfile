FROM python:3.12

# Установим рабочую директорию
WORKDIR /app

# Копируем requirements.txt из корня проекта
COPY requirements.txt /app/requirements.txt

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Устанавливаем libGL для OpenCV
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Копируем только папку FastAPI внутрь контейнера
COPY ./FastAPI /app

# Открываем порт FastAPI
EXPOSE 8000

# Запускаем приложение
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
