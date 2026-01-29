#!/bin/bash
set -e

# Configuration
PROJECT_ID="${PROJECT_ID:-heum-alfred-evidence-clf-dev}"
REGION="${REGION:-asia-northeast3}"
REPOSITORY="${REPOSITORY:-vertex-ai-models}"
IMAGE_NAME="${IMAGE_NAME:-sklearn-serving}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

# Full image URI
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}:${IMAGE_TAG}"

echo "=========================================="
echo "  Building Custom Serving Container"
echo "=========================================="
echo ""
echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Image URI: ${IMAGE_URI}"
echo ""

# Navigate to serving directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SERVING_DIR="${SCRIPT_DIR}/../src/serving"

cd "${SERVING_DIR}"

# Create Artifact Registry repository if not exists
echo "[1] Creating Artifact Registry repository (if needed)..."
gcloud artifacts repositories create "${REPOSITORY}" \
    --repository-format=docker \
    --location="${REGION}" \
    --project="${PROJECT_ID}" \
    --quiet 2>/dev/null || echo "Repository already exists or creation failed (continuing...)"

# Configure Docker for Artifact Registry
echo ""
echo "[2] Configuring Docker authentication..."
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

# Build the container
echo ""
echo "[3] Building Docker image..."
docker build --platform linux/amd64 -t "${IMAGE_URI}" .

# Push to Artifact Registry
echo ""
echo "[4] Pushing to Artifact Registry..."
docker push "${IMAGE_URI}"

echo ""
echo "=========================================="
echo "  Container Build Complete!"
echo "=========================================="
echo "Image URI: ${IMAGE_URI}"
echo ""
echo "Use this URI in upload_model.py:"
echo "  serving_container_image_uri=\"${IMAGE_URI}\""
