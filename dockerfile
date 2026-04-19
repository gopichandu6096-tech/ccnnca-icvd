FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p data outputs

ENV PYTHONPATH=/app
CMD ["python", "scripts/train.py", "--config", "configs/default.yaml"]