# Руководство пользователя для приложения Slaaaay ML App

## Описание системы

Slaaaay ML App — это удобное приложение для обучения моделей машинного обучения. Оно включает в себя:

1. **Сервис на FastAPI**, который обеспечивает API для взаимодействия с приложением.
2. **Streamlit-приложение**, предоставляющее удобный графический интерфейс для работы с приложением.

Приложение позволяет:

- Решать задачу классификации по изображениям. 
- Загружать пользовательские датасеты. 
- Выполнять анализ данных (EDA) в отдельном разделе. 
- Выбирать метод обучения (SVC, Logistic Regression, Random Forest). 
- Выполнять предсказания по изображениям. 
- Просматривать список обученных моделей. 
- Удалять ненужные модели. 

## Функции приложения

### 1. EDA (Exploratory Data Analysis)

Раздел предназначен для анализа данных:

- Загрузите датасет (поддерживаются файлы в формате ZIP, размером до 1 ГБ).
- Приложение выполнит автоматический анализ данных, включая:
  - Информацию о количестве классов.
  - Количество изображений для обучающего и тестового наборов.
  - Значение средней яркости по калассам.
  - Значение стандартного отклонения яркости по классам.

### 2. Обучение

Раздел предназначен для обучения новых моделей:

- Загрузите датасет (поддерживаются файлы в формате ZIP, размером до 200 МБ). Если вы хотите работать с большими датасетами, настройте параметр загрузки для Streamlit с помощью команды:
  ```bash
  streamlit run client.py --server.maxUploadSize 1000
  ```
- Выберите метод обучения (SVC, Logistic Regression, Random Forest).
- Запустите процесс обучения.
- После обучения отображается информация о модели, включая метрики (precision, recall, f1-score, support) и визуализацию результатов.

### 3. Предсказание

Этот раздел позволяет загрузить изображение и получить предсказание, определяя, что на нем изображено, исходя из обученной модели:

- Загрузите изображение.
- Выберите модель для предсказания (список обученных моделей можно увидеть в разделе "Список моделей").
- Получите результат с вероятностями для каждого класса.

### 4. Список моделей

В этом разделе отображается список всех обученных моделей. Вы можете выбрать любую модель для дальнейшего использования.

### 5. Удаление моделей

Позволяет удалить ненужные модели для освобождения пространства или обновления результатов.

## Сборка и запуск контейнера

Для работы с приложением необходимо создать Docker-образ и запустить его. Следуйте инструкциям ниже.

### 1. Подготовка окружения

Убедитесь, что у вас установлены:

- **Docker** ([https://www.docker.com/](https://www.docker.com/)), а также **Docker Compose**.
- Файл `Dockerfile`, содержащий инструкции для сборки каждого образа(в нашем случае это "FastAPI" в качестве бэкенда и "Streamlit" в качестве фронтенда).
- Файл `docker-compose.yml` содержащий объединенные инстукции для взаимодействия образов друг с другом. 
- Файлы `requirements.txt` с зависимостями Python.

### 2. Сборка Docker-образов, настройка Docker compose и запуск контейнера. 

1. Перейдите в директорию с проектом:
   ```bash
   cd /path/to/project
   ```
2. Соберите Docker-образ(при этом на ваше компьютере на заднем плане должно быть открыто приложение Docker):
   ```bash
   docker-compose up --build
   ```
   По дефолту здесь будут создаваться два образа -  `fastapi` и `streamlit`.
   Также в нашем случае будут использоваться два порта, которые мы подводим к таким же портам на докере:
   - Порт `8501` используется для Streamlit.
   - Порт `8000` используется для FastAPI.

2. После запуска откройте в браузере:

   - **Streamlit-приложение:** `http://localhost:8501`
   - **FastAPI-документация:** `http://localhost:8000/docs`

### 4. Остановка контейнера

Для остановки контейнера выполните:

```bash
docker-compose stop
```

Для воспроизведения работы контейнера выполните:

```bash
docker-compose start
```
Для остановки и удаления контейнеров выполните:

```bash
docker-compose down
```


## Использование приложения

1. **EDA:**

   - Зайдите в Streamlit-приложение.
   - Выберите раздел "EDA".
   - Загрузите ваш датасет.
   - После завершения анализа ознакомьтесь с результатами: распределением классов, структурой данных и значениями яркости изображений.

2. **Обучение:**

   - Перейдите в раздел "Обучение".
   - Загрузите ваш датасет и настройте параметры обучения.
   - Запустите процесс обучения.
   - После завершения обучения проверьте результаты модели (метрики и визуализации).

3. **Предсказание:**

   - Перейдите в раздел "Предсказание".
   - Загрузите изображение для анализа.
   - Выберите обученную модель.
   - Получите предсказание и вероятности для каждого класса.

4. **Список моделей:**

   - Откройте раздел "Список моделей".
   - Получите список уже обученных моделей.

5. **Удаление моделей:**

   - Откройте раздел "Удаление моделей".
   - Удалите все модели и загруженные датасеты из хранилища данных.

## Замечания
- В качестве тестового датасета исползовался https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset, но приложение поддерживает любые другие данные соответсвующего формата.
- Убедитесь, что ваш датасет корректно подготовлен перед загрузкой (имеет папки train и test, каждая папка имеет структуру подпапок с одинаковыми классами. Папка класса содержит изображения в формате .png, .jpg, .jpeg. В названии изображений отсутствуют пробелы и символы).
- При возникновении ошибок проверяйте логи контейнера с помощью команды:
  ```bash
  docker-compose logs
  ```

Приложение разработано для упрощения работы с задачами машинного обучения и предоставляет интуитивно понятный интерфейс.
