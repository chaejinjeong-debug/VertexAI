"""
Model Upload Component

파이프라인 아티팩트에서 직접 Model Registry에 등록하고 Vertex AI Experiments에 메트릭을 로깅합니다.
별도의 GCS 업로드 없이 파이프라인의 아티팩트 경로를 그대로 사용합니다.

Usage:
    python main.py --help
    python main.py \
        --input_model_dir /gcs/bucket/path/to/model \
        --input_metrics_path /gcs/bucket/path/to/metrics.json \
        --project_id my-project \
        --region asia-northeast3 \
        --experiment_name churn-experiment \
        --model_display_name churn-model \
        --output_model_resource /path/to/output.txt
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from google.cloud import aiplatform

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Model Upload Component - Upload to Model Registry with Experiments logging",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input arguments
    parser.add_argument(
        "--input_model_dir",
        type=str,
        required=True,
        help="Input model directory (contains model.pkl, model_meta.json)"
    )
    parser.add_argument(
        "--input_metrics_path",
        type=str,
        required=True,
        help="Input metrics JSON file path"
    )

    # GCP settings
    parser.add_argument(
        "--project_id",
        type=str,
        required=True,
        help="GCP project ID"
    )
    parser.add_argument(
        "--region",
        type=str,
        default="asia-northeast3",
        help="GCP region"
    )

    # Model Registry settings
    parser.add_argument(
        "--model_display_name",
        type=str,
        default="churn-model",
        help="Display name for the model in Model Registry"
    )
    parser.add_argument(
        "--serving_container_image",
        type=str,
        default="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest",
        help="Serving container image URI"
    )

    # Experiment settings
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="churn-experiment",
        help="Vertex AI Experiment name"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Experiment run name (auto-generated if not provided)"
    )

    # Output arguments
    parser.add_argument(
        "--output_model_resource",
        type=str,
        required=True,
        help="Output file path to save model resource name"
    )

    return parser.parse_args()


def load_metrics(metrics_path: Path) -> dict:
    """Load metrics from JSON file."""
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    with open(metrics_path) as f:
        metrics = json.load(f)

    logger.info(f"Loaded metrics from {metrics_path}")
    return metrics


def load_model_meta(model_dir: Path) -> dict:
    """Load model metadata."""
    meta_path = model_dir / "model_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return {}


def convert_gcs_path(local_path: str) -> str:
    """Convert /gcs/bucket/path to gs://bucket/path format.

    Vertex AI Pipelines mounts GCS as /gcs/bucket/path.
    Model Registry requires gs://bucket/path format.
    """
    if local_path.startswith("/gcs/"):
        # /gcs/bucket/path -> gs://bucket/path
        return "gs://" + local_path[5:]
    elif local_path.startswith("gs://"):
        # Already in correct format
        return local_path
    else:
        raise ValueError(
            f"Invalid path format: {local_path}. "
            "Expected /gcs/bucket/path or gs://bucket/path"
        )


def log_to_experiments(
    project_id: str,
    region: str,
    experiment_name: str,
    run_name: str,
    metrics: dict,
    params: dict,
) -> str:
    """Log metrics and parameters to Vertex AI Experiments."""
    logger.info(f"Logging to experiment: {experiment_name}, run: {run_name}")

    # Set experiment in the context (re-init with experiment)
    aiplatform.init(
        project=project_id,
        location=region,
        experiment=experiment_name,
    )

    # Start a run and log
    with aiplatform.start_run(run=run_name) as run:
        # Log metrics (filter to numeric values only)
        numeric_metrics = {
            k: v for k, v in metrics.items()
            if isinstance(v, (int, float)) and k not in ["eval_samples"]
        }
        aiplatform.log_metrics(numeric_metrics)
        logger.info(f"Logged metrics: {list(numeric_metrics.keys())}")

        # Log parameters
        if params:
            # Convert all params to string (Experiments requirement)
            str_params = {k: str(v) for k, v in params.items()}
            aiplatform.log_params(str_params)
            logger.info(f"Logged params: {list(str_params.keys())}")

    return run_name


def upload_to_model_registry(
    model_display_name: str,
    artifact_uri: str,
    serving_container_image: str,
    model_meta: dict,
    metrics: dict,
) -> aiplatform.Model:
    """Upload model to Vertex AI Model Registry."""
    logger.info(f"Uploading model to Model Registry: {model_display_name}")

    # Prepare labels
    labels = {
        "model_type": model_meta.get("model_type", "unknown").replace("_", "-"),
        "created_by": "vertex-ai-pipeline",
    }

    # Prepare description
    description_parts = [
        f"Customer Churn Prediction Model",
        f"Model Type: {model_meta.get('model_type', 'unknown')}",
        f"ROC-AUC: {metrics.get('roc_auc', 'N/A'):.4f}" if isinstance(metrics.get('roc_auc'), float) else "",
        f"PR-AUC: {metrics.get('pr_auc', 'N/A'):.4f}" if isinstance(metrics.get('pr_auc'), float) else "",
    ]
    description = "\n".join([p for p in description_parts if p])

    model = aiplatform.Model.upload(
        display_name=model_display_name,
        artifact_uri=artifact_uri,
        serving_container_image_uri=serving_container_image,
        description=description,
        labels=labels,
    )

    logger.info(f"Model uploaded successfully: {model.resource_name}")
    return model


def save_output(model_resource_name: str, output_path: Path) -> None:
    """Save model resource name to output file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(model_resource_name)
    logger.info(f"Saved model resource name to {output_path}")


def main() -> None:
    """Main entry point."""
    args = parse_args()

    logger.info("=" * 50)
    logger.info("Model Upload Component")
    logger.info("=" * 50)

    # Initialize Vertex AI
    logger.info(f"Initializing Vertex AI (project={args.project_id}, region={args.region})")
    aiplatform.init(
        project=args.project_id,
        location=args.region,
    )

    # Load inputs
    model_dir = Path(args.input_model_dir)
    metrics_path = Path(args.input_metrics_path)

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    metrics = load_metrics(metrics_path)
    model_meta = load_model_meta(model_dir)

    logger.info(f"Model type: {model_meta.get('model_type', 'unknown')}")
    logger.info(f"Metrics: ROC-AUC={metrics.get('roc_auc', 'N/A'):.4f}, PR-AUC={metrics.get('pr_auc', 'N/A'):.4f}")

    # Generate run name
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = args.run_name or f"run-{timestamp}"
    model_display_name = f"{args.model_display_name}-{timestamp}"

    # Step 1: Convert path to GCS URI (no re-upload needed)
    logger.info("-" * 50)
    logger.info("Step 1: Converting artifact path to GCS URI")
    artifact_uri = convert_gcs_path(args.input_model_dir)
    logger.info(f"Artifact URI: {artifact_uri}")

    # Step 2: Log to Experiments
    logger.info("-" * 50)
    logger.info("Step 2: Logging to Vertex AI Experiments")
    params = {
        "model_type": model_meta.get("model_type", "unknown"),
        "n_estimators": model_meta.get("n_estimators", "N/A"),
        "max_depth": model_meta.get("max_depth", "N/A"),
        "feature_count": model_meta.get("feature_count", "N/A"),
    }
    log_to_experiments(
        project_id=args.project_id,
        region=args.region,
        experiment_name=args.experiment_name,
        run_name=run_name,
        metrics=metrics,
        params=params,
    )

    # Step 3: Upload to Model Registry
    logger.info("-" * 50)
    logger.info("Step 3: Uploading to Model Registry")
    model = upload_to_model_registry(
        model_display_name=model_display_name,
        artifact_uri=artifact_uri,
        serving_container_image=args.serving_container_image,
        model_meta=model_meta,
        metrics=metrics,
    )

    # Save output
    output_path = Path(args.output_model_resource)
    save_output(model.resource_name, output_path)

    # Summary
    logger.info("=" * 50)
    logger.info("Model Upload Complete!")
    logger.info("=" * 50)
    logger.info(f"Experiment:     {args.experiment_name}")
    logger.info(f"Run:            {run_name}")
    logger.info(f"Model:          {model_display_name}")
    logger.info(f"Resource:       {model.resource_name}")
    logger.info(f"Artifact URI:   {artifact_uri}")
    logger.info("=" * 50)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Component failed: {e}")
        sys.exit(1)
