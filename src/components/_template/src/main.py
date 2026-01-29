"""
Pipeline Component Template

Usage:
    python main.py --help
    python main.py --input_path /path/to/input --output_path /path/to/output
"""

import argparse
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Pipeline Component Template",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input arguments (--input_*)
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to input data"
    )

    # Output arguments (--output_*)
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to output data"
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    logger.info("Starting component execution")
    logger.info(f"Input path: {args.input_path}")
    logger.info(f"Output path: {args.output_path}")

    # TODO: Implement component logic here

    logger.info("Component execution completed")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Component failed: {e}")
        sys.exit(1)
