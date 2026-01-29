# Data Load Component - BigQuery to Parquet with time-based split
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ /app/src/

# Set entrypoint
ENTRYPOINT ["python", "/app/src/main.py"]
