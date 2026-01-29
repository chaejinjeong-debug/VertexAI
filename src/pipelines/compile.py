"""
Pipeline Compiler

파이프라인을 JSON 형식으로 컴파일합니다.

Usage:
    python -m src.pipelines.compile
    python -m src.pipelines.compile --output pipeline.json
"""

import argparse
import logging
from pathlib import Path

from kfp import compiler

from src.pipelines.pipeline import churn_training_pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Default output path
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "compiled"
DEFAULT_OUTPUT_FILE = "churn_training_pipeline.json"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compile Vertex AI Pipeline to JSON",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR / DEFAULT_OUTPUT_FILE),
        help="Output JSON file path"
    )

    return parser.parse_args()


def compile_pipeline(output_path: str) -> None:
    """Compile the pipeline to JSON format."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Compiling pipeline...")
    logger.info(f"Output: {output_file}")

    compiler.Compiler().compile(
        pipeline_func=churn_training_pipeline,
        package_path=str(output_file),
    )

    logger.info(f"Pipeline compiled successfully: {output_file}")
    logger.info(f"File size: {output_file.stat().st_size:,} bytes")


def main() -> None:
    """Main entry point."""
    args = parse_args()

    logger.info("=" * 50)
    logger.info("Pipeline Compiler")
    logger.info("=" * 50)

    compile_pipeline(args.output)

    logger.info("=" * 50)
    logger.info("Compilation complete")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
