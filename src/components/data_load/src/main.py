"""
Data Load Component

BigQuery 테이블에서 데이터를 로드하고 시간 기반으로 train/valid/test로 분할합니다.

Usage:
    python main.py --help
    python main.py \
        --input_bq_table project.dataset.table \
        --output_dataset_dir /path/to/output \
        --label_column label_churn_60d \
        --time_column label_timestamp
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
from google.cloud import bigquery

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Data Load Component - BigQuery to Parquet with time-based split",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input arguments
    parser.add_argument(
        "--input_bq_table",
        type=str,
        required=True,
        help="BigQuery table path (project.dataset.table)"
    )

    # Output arguments
    parser.add_argument(
        "--output_dataset_dir",
        type=str,
        required=True,
        help="Output directory for parquet files"
    )

    # Configuration arguments
    parser.add_argument(
        "--label_column",
        type=str,
        default="label_churn_60d",
        help="Label column name"
    )
    parser.add_argument(
        "--time_column",
        type=str,
        default="label_timestamp",
        help="Time column for time-based split"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Training data ratio"
    )
    parser.add_argument(
        "--valid_ratio",
        type=float,
        default=0.15,
        help="Validation data ratio"
    )

    return parser.parse_args()


def load_from_bigquery(bq_table: str) -> pd.DataFrame:
    """Load data from BigQuery table."""
    logger.info(f"Loading data from BigQuery: {bq_table}")

    # Extract project ID from table path (project.dataset.table)
    project_id = bq_table.split(".")[0]
    logger.info(f"Using project: {project_id}")

    client = bigquery.Client(project=project_id)
    query = f"SELECT * FROM `{bq_table}`"

    df = client.query(query).to_dataframe()
    logger.info(f"Loaded {len(df):,} rows from BigQuery")

    return df


def time_based_split(
    df: pd.DataFrame,
    time_column: str,
    train_ratio: float,
    valid_ratio: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data based on time column to prevent data leakage.

    Earlier data -> train
    Middle data -> valid
    Later data -> test
    """
    logger.info(f"Splitting data by time column: {time_column}")

    # Sort by time
    df_sorted = df.sort_values(time_column).reset_index(drop=True)

    n = len(df_sorted)
    train_end = int(n * train_ratio)
    valid_end = int(n * (train_ratio + valid_ratio))

    train_df = df_sorted.iloc[:train_end]
    valid_df = df_sorted.iloc[train_end:valid_end]
    test_df = df_sorted.iloc[valid_end:]

    logger.info(f"Split sizes - Train: {len(train_df):,}, Valid: {len(valid_df):,}, Test: {len(test_df):,}")

    return train_df, valid_df, test_df


def save_parquet(df: pd.DataFrame, output_path: Path) -> None:
    """Save DataFrame as parquet file."""
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved {len(df):,} rows to {output_path}")


def create_metadata(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_column: str,
    time_column: str,
    output_dir: Path
) -> dict:
    """Create and save dataset metadata."""
    # Calculate label distribution
    train_positive_rate = train_df[label_column].mean()
    valid_positive_rate = valid_df[label_column].mean()
    test_positive_rate = test_df[label_column].mean()

    # Get time ranges
    all_times = pd.concat([train_df[time_column], valid_df[time_column], test_df[time_column]])

    metadata = {
        "train_samples": len(train_df),
        "valid_samples": len(valid_df),
        "test_samples": len(test_df),
        "total_samples": len(train_df) + len(valid_df) + len(test_df),
        "label_column": label_column,
        "time_column": time_column,
        "train_positive_rate": float(train_positive_rate),
        "valid_positive_rate": float(valid_positive_rate),
        "test_positive_rate": float(test_positive_rate),
        "time_range": {
            "min": str(all_times.min()),
            "max": str(all_times.max())
        },
        "columns": list(train_df.columns)
    }

    # Save metadata
    metadata_path = output_dir / "dataset_meta.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_path}")

    return metadata


def main() -> None:
    """Main entry point."""
    args = parse_args()

    logger.info("=" * 50)
    logger.info("Data Load Component")
    logger.info("=" * 50)

    # Create output directory
    output_dir = Path(args.output_dataset_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data from BigQuery
    df = load_from_bigquery(args.input_bq_table)

    # Validate required columns
    if args.time_column not in df.columns:
        raise ValueError(f"Time column '{args.time_column}' not found in data")
    if args.label_column not in df.columns:
        raise ValueError(f"Label column '{args.label_column}' not found in data")

    # Time-based split
    train_df, valid_df, test_df = time_based_split(
        df=df,
        time_column=args.time_column,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio
    )

    # Save parquet files
    save_parquet(train_df, output_dir / "train.parquet")
    save_parquet(valid_df, output_dir / "valid.parquet")
    save_parquet(test_df, output_dir / "test.parquet")

    # Create and save metadata
    metadata = create_metadata(
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        label_column=args.label_column,
        time_column=args.time_column,
        output_dir=output_dir
    )

    # Log summary
    logger.info("=" * 50)
    logger.info("Summary")
    logger.info("=" * 50)
    logger.info(f"Total samples: {metadata['total_samples']:,}")
    logger.info(f"Train positive rate: {metadata['train_positive_rate']:.2%}")
    logger.info(f"Valid positive rate: {metadata['valid_positive_rate']:.2%}")
    logger.info(f"Test positive rate: {metadata['test_positive_rate']:.2%}")

    # Warn if label is biased
    for split, rate in [
        ("train", metadata['train_positive_rate']),
        ("valid", metadata['valid_positive_rate']),
        ("test", metadata['test_positive_rate'])
    ]:
        if rate < 0.05 or rate > 0.95:
            logger.warning(f"Warning: {split} set has biased labels (positive rate: {rate:.2%})")

    logger.info("Data load completed successfully")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Component failed: {e}")
        sys.exit(1)
