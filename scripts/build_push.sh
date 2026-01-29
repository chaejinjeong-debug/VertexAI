#!/bin/bash
set -e

# build_push.sh - Build and push a single component to Artifact Registry
#
# Usage:
#   ./scripts/build_push.sh <component_name>
#   ./scripts/build_push.sh data_load
#   ./scripts/build_push.sh train
#   ./scripts/build_push.sh eval
#
# Environment variables:
#   PROJECT_ID  - GCP project ID (default: heum-alfred-evidence-clf-dev)
#   REGION      - GCP region (default: asia-northeast3)
#   REPOSITORY  - Artifact Registry repository (default: vertex-ai-pipelines)
#   IMAGE_TAG   - Image tag (default: git SHA or 'latest')

# Configuration
PROJECT_ID="${PROJECT_ID:-heum-alfred-evidence-clf-dev}"
REGION="${REGION:-asia-northeast3}"
REPOSITORY="${REPOSITORY:-vertex-ai-pipelines}"

# Get git SHA for tagging (fallback to 'latest')
GIT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "latest")
IMAGE_TAG="${IMAGE_TAG:-${GIT_SHA}}"

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="${SCRIPT_DIR}/.."
COMPONENTS_DIR="${PROJECT_ROOT}/src/components"

# Validate component name
COMPONENT_NAME="${1}"
if [ -z "${COMPONENT_NAME}" ]; then
    echo "Error: Component name is required"
    echo ""
    echo "Usage: $0 <component_name>"
    echo ""
    echo "Available components:"
    ls -d "${COMPONENTS_DIR}"/*/ 2>/dev/null | xargs -n1 basename | grep -v "^_" || echo "  (none found)"
    exit 1
fi

# Check if component directory exists
COMPONENT_DIR="${COMPONENTS_DIR}/${COMPONENT_NAME}"
if [ ! -d "${COMPONENT_DIR}" ]; then
    echo "Error: Component directory not found: ${COMPONENT_DIR}"
    echo ""
    echo "Available components:"
    ls -d "${COMPONENTS_DIR}"/*/ 2>/dev/null | xargs -n1 basename | grep -v "^_" || echo "  (none found)"
    exit 1
fi

# Check if Dockerfile exists
if [ ! -f "${COMPONENT_DIR}/Dockerfile" ]; then
    echo "Error: Dockerfile not found in ${COMPONENT_DIR}"
    exit 1
fi

# Image URI
IMAGE_NAME="component-${COMPONENT_NAME}"
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}:${IMAGE_TAG}"
IMAGE_URI_LATEST="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}:latest"

echo "=========================================="
echo "  Building Component: ${COMPONENT_NAME}"
echo "=========================================="
echo ""
echo "Project:    ${PROJECT_ID}"
echo "Region:     ${REGION}"
echo "Repository: ${REPOSITORY}"
echo "Image Tag:  ${IMAGE_TAG}"
echo "Image URI:  ${IMAGE_URI}"
echo ""

# Create Artifact Registry repository if not exists
echo "[1/4] Creating Artifact Registry repository (if needed)..."
gcloud artifacts repositories create "${REPOSITORY}" \
    --repository-format=docker \
    --location="${REGION}" \
    --project="${PROJECT_ID}" \
    --quiet 2>/dev/null || echo "  Repository already exists"

# Configure Docker for Artifact Registry
echo ""
echo "[2/4] Configuring Docker authentication..."
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

# Build the container
echo ""
echo "[3/4] Building Docker image..."
cd "${COMPONENT_DIR}"
docker build --platform linux/amd64 -t "${IMAGE_URI}" -t "${IMAGE_URI_LATEST}" .

# Push to Artifact Registry
echo ""
echo "[4/4] Pushing to Artifact Registry..."
docker push "${IMAGE_URI}"
docker push "${IMAGE_URI_LATEST}"

echo ""
echo "=========================================="
echo "  Build Complete!"
echo "=========================================="
echo "Component:  ${COMPONENT_NAME}"
echo "Image URI:  ${IMAGE_URI}"
echo "Latest URI: ${IMAGE_URI_LATEST}"
echo ""
