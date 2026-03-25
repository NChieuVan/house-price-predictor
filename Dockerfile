FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY src/api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/api/ /app/src/api/
COPY models/trained/ /app/models/trained/

EXPOSE 8000

CMD ["uvicorn", "main:app", "--app-dir", "/app/src/api", "--host", "0.0.0.0", "--port", "8000"]