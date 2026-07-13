FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# the host injects $PORT; app.py binds 0.0.0.0:$PORT (defaults to 7860 locally)
CMD ["python", "app.py"]
