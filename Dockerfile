FROM python:3.11-slim

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements-web.txt

CMD ["sh", "-c", "gunicorn watereos_visualizer.app:server --bind 0.0.0.0:$PORT"]
