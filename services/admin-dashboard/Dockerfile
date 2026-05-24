FROM python:3.11-slim

WORKDIR /app

# Install deps separately so layer is cached when only source changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Source is bind-mounted in dev (see docker-compose.yml). Copy as a
# fallback so the image is also runnable standalone.
COPY . .

EXPOSE 8820

ENV ADMIN_HOST=0.0.0.0 \
    ADMIN_PORT=8820 \
    PYTHONUNBUFFERED=1

CMD ["python", "server.py"]
