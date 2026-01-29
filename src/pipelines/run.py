"""
Pipeline Runner

컴파일된 파이프라인을 Vertex AI Pipelines에 제출합니다.

Usage:
    python -m src.pipelines.run
    python -m src.pipelines.run --pipeline-file pipeline.json
    python -m src.pipelines.run --input-bq-table project.dataset.table
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

import yaml
from google.cloud import aiplatform

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "env.yaml"
DEFAULT_PIPELINE_FILE = Path(__file__).parent / "compiled" / "churn_training_pipeline.json"


def load_config() -> dict:
    """Load configuration from env.yaml."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")

    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    config = load_config()

    parser = argparse.ArgumentParser(
        description="Submit pipeline to Vertex AI Pipelines",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Pipeline file
    parser.add_argument(
        "--pipeline-file",
        type=str,
        default=str(DEFAULT_PIPELINE_FILE),
        help="Compiled pipeline JSON file"
    )

    # GCP settings
    parser.add_argument(
        "--project-id",
        type=str,
        default=config["gcp"]["project_id"],
        help="GCP project ID"
    )
    parser.add_argument(
        "--region",
        type=str,
        default=config["gcp"]["region"],
        help="GCP region"
    )

    # Pipeline parameters
    parser.add_argument(
        "--input-bq-table",
        type=str,
        default=f"{config['gcp']['project_id']}.{config['bigquery']['target_dataset']}.{config['bigquery']['tables']['train_dataset']}",
        help="BigQuery table for training data"
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="label_churn_60d",
        help="Label column name"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="random_forest",
        choices=["random_forest", "logistic_regression"],
        help="Model type"
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of trees (for random_forest)"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=10,
        help="Maximum tree depth"
    )

    # Execution options
    parser.add_argument(
        "--display-name",
        type=str,
        default=None,
        help="Pipeline run display name (auto-generated if not provided)"
    )
    parser.add_argument(
        "--service-account",
        type=str,
        default=None,
        help="Service account for pipeline execution"
    )
    parser.add_argument(
        "--enable-caching",
        action="store_true",
        default=True,
        help="Enable pipeline caching"
    )
    parser.add_argument(
        "--no-caching",
        action="store_true",
        help="Disable pipeline caching"
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Wait for pipeline to complete"
    )

    return parser.parse_args()


def run_pipeline(args: argparse.Namespace) -> str:
    """Submit pipeline to Vertex AI Pipelines."""
    # Validate pipeline file
    pipeline_file = Path(args.pipeline_file)
    if not pipeline_file.exists():
        raise FileNotFoundError(
            f"Pipeline file not found: {pipeline_file}\n"
            "Run 'python -m src.pipelines.compile' first."
        )

    # Initialize Vertex AI
    logger.info(f"Initializing Vertex AI (project={args.project_id}, region={args.region})")
    aiplatform.init(
        project=args.project_id,
        location=args.region,
    )

    # Generate display name
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    display_name = args.display_name or f"churn-training-{timestamp}"

    # Pipeline parameters
    pipeline_params = {
        "input_bq_table": args.input_bq_table,
        "label_column": args.label_column,
        "time_column": "label_timestamp",
        "train_ratio": 0.7,
        "valid_ratio": 0.15,
        "model_type": args.model_type,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "random_state": 42,
    }

    logger.info("Pipeline parameters:")
    for key, value in pipeline_params.items():
        logger.info(f"  {key}: {value}")

    # Create pipeline job
    logger.info(f"Creating pipeline job: {display_name}")

    enable_caching = not args.no_caching and args.enable_caching

    job = aiplatform.PipelineJob(
        display_name=display_name,
        template_path=str(pipeline_file),
        parameter_values=pipeline_params,
        enable_caching=enable_caching,
    )

    # Submit pipeline
    logger.info("Submitting pipeline to Vertex AI...")

    job.submit(
        service_account=args.service_account,
    )

    logger.info(f"Pipeline submitted successfully!")
    logger.info(f"Job name: {job.name}")
    logger.info(f"Job resource name: {job.resource_name}")

    # Get console URL
    console_url = (
        f"https://console.cloud.google.com/vertex-ai/pipelines/runs/"
        f"{job.name.split('/')[-1]}?project={args.project_id}"
    )
    logger.info(f"Console URL: {console_url}")

    # Wait for completion if requested
    if args.sync:
        logger.info("Waiting for pipeline to complete...")
        job.wait()
        logger.info(f"Pipeline completed with state: {job.state}")

    return job.resource_name


def main() -> None:
    """Main entry point."""
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Vertex AI Pipeline Runner")
    logger.info("=" * 60)

    try:
        resource_name = run_pipeline(args)
        logger.info("=" * 60)
        logger.info("Pipeline submitted successfully!")
        logger.info(f"Resource: {resource_name}")
        logger.info("=" * 60)
    except FileNotFoundError as e:
        logger.error(str(e))
        raise SystemExit(1)
    except Exception as e:
        logger.error(f"Pipeline submission failed: {e}")
        raise


if __name__ == "__main__":
    main()
