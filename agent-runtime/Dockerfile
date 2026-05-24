FROM python:3.12-slim

WORKDIR /app

# build-essential needed for some claude-agent-sdk transitive deps that
# compile native extensions; curl kept for the healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

EXPOSE 5000

# Healthcheck hits /health (defined in app.main); 60s start_period gives
# the SDK time to warm up its internal state on cold boot.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5000"]
