FROM python:3.10
COPY requirements.txt app/requirements.txt
WORKDIR /app
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install -y libgl1-mesa-glx
COPY . /app
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
