FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY classifier.py api.py evaluate.py trainingsdaten.json testdaten.json ./
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
