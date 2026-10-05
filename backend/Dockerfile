FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY twin ./twin
COPY backend ./backend
ENV TWIN_DB=/app/data/twin.duckdb TWIN_SCHEDULER=1
EXPOSE 8000
CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000"]
