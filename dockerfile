FROM python:3.14-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY api/ ./api/
COPY src/ ./src/ 

COPY models/churn_model_v2.joblib ./models/
COPY models/churn_model_v2_metrics.json ./models/

RUN useradd --create-home appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["python","-m","uvicorn","api.main:app","--host","0.0.0.0","--port","8000"]
