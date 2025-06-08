FROM python:3.12

# Рабочая директория
WORKDIR /app

# Копируем requirements
COPY requirements.txt /app/requirements.txt

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# libGL для OpenCV
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Копируем только папку FastAPI
COPY ./FastAPI /app/FastAPI

# Устанавливаем рабочую директорию внутри FastAPI
WORKDIR /app/FastAPI

# Открываем порт FastAPI
EXPOSE 8000

# Запуск uvicorn изнутри папки FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
