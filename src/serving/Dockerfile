# Custom sklearn serving container for Vertex AI Prediction
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    flask==3.0.0 \
    gunicorn==21.2.0 \
    joblib==1.3.2 \
    numpy==1.26.2 \
    scikit-learn==1.5.2 \
    google-cloud-storage==2.13.0

# Copy serving script
COPY serve.py /app/serve.py

# Environment variables for Vertex AI
ENV AIP_HTTP_PORT=8080
ENV AIP_HEALTH_ROUTE=/health
ENV AIP_PREDICT_ROUTE=/predict

# Expose port
EXPOSE 8080

# Run gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "4", "serve:app"]
