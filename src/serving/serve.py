"""
Custom sklearn serving container for Vertex AI Prediction.
Loads model from GCS and serves predictions via Flask.
"""

import os
import logging

import joblib
import numpy as np
from flask import Flask, jsonify, request
from google.cloud import storage

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global model variable
model = None


def download_model_from_gcs():
    """Download model from GCS artifact URI."""
    global model

    # AIP_STORAGE_URI is set by Vertex AI with the artifact_uri
    artifact_uri = os.environ.get("AIP_STORAGE_URI", "")

    if not artifact_uri:
        logger.error("AIP_STORAGE_URI environment variable not set")
        return False

    logger.info(f"Loading model from: {artifact_uri}")

    try:
        # Parse GCS URI
        # Format: gs://bucket/path/to/model
        if artifact_uri.startswith("gs://"):
            path = artifact_uri[5:]
            bucket_name = path.split("/")[0]
            prefix = "/".join(path.split("/")[1:])
        else:
            logger.error(f"Invalid artifact URI format: {artifact_uri}")
            return False

        # Download model file
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)

        # Model file path
        model_blob_path = f"{prefix}/model.joblib"
        local_model_path = "/tmp/model.joblib"

        blob = bucket.blob(model_blob_path)
        blob.download_to_filename(local_model_path)

        logger.info(f"Downloaded model from: gs://{bucket_name}/{model_blob_path}")

        # Load model
        model = joblib.load(local_model_path)
        logger.info("Model loaded successfully")

        return True

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False


# Load model on startup
download_model_from_gcs()


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    if model is not None:
        return jsonify({"status": "healthy"}), 200
    else:
        return jsonify({"status": "unhealthy", "error": "Model not loaded"}), 503


@app.route("/predict", methods=["POST"])
def predict():
    """Prediction endpoint."""
    global model

    if model is None:
        return jsonify({"error": "Model not loaded"}), 503

    try:
        # Parse request
        request_json = request.get_json()

        if not request_json:
            return jsonify({"error": "No JSON payload"}), 400

        # Get instances from request
        # Vertex AI sends {"instances": [[...], [...]]}
        instances = request_json.get("instances", [])

        if not instances:
            return jsonify({"error": "No instances in request"}), 400

        # Convert to numpy array
        X = np.array(instances)

        # Predict probabilities
        probabilities = model.predict_proba(X)

        # Return predictions in Vertex AI format
        # [[no_churn_prob, churn_prob], ...]
        predictions = probabilities.tolist()

        return jsonify({"predictions": predictions})

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # For local testing
    app.run(host="0.0.0.0", port=8080, debug=True)
