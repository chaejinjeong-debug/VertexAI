"""
Eval Component

모델을 평가하고 메트릭을 계산합니다.

Usage:
    python main.py --help
    python main.py \
        --input_model_dir /path/to/model \
        --input_dataset_dir /path/to/dataset \
        --output_metrics_path /path/to/metrics.json \
        --eval_split valid
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Default feature columns (from training.yaml)
DEFAULT_FEATURE_COLUMNS = [
    "orders_30d",
    "orders_90d",
    "revenue_30d",
    "revenue_90d",
    "avg_order_value_90d",
    "distinct_products_90d",
    "distinct_categories_90d",
    "days_since_last_order"
]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Eval Component - Model Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input arguments
    parser.add_argument(
        "--input_model_dir",
        type=str,
        required=True,
        help="Input model directory (contains model.pkl)"
    )
    parser.add_argument(
        "--input_dataset_dir",
        type=str,
        required=True,
        help="Input dataset directory (contains valid.parquet or test.parquet)"
    )

    # Output arguments
    parser.add_argument(
        "--output_metrics_path",
        type=str,
        required=True,
        help="Output metrics JSON file path"
    )

    # Configuration arguments
    parser.add_argument(
        "--eval_split",
        type=str,
        default="valid",
        choices=["valid", "test"],
        help="Which split to evaluate on"
    )
    parser.add_argument(
        "--feature_columns",
        type=str,
        default=",".join(DEFAULT_FEATURE_COLUMNS),
        help="Comma-separated list of feature column names"
    )
    parser.add_argument(
        "--label_column",
        type=str,
        default="label_churn_60d",
        help="Label column name"
    )

    return parser.parse_args()


def load_model(model_dir: Path):
    """Load model from pickle file."""
    model_path = model_dir / "model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = joblib.load(model_path)
    logger.info(f"Loaded model from {model_path}")

    return model


def load_eval_data(dataset_dir: Path, eval_split: str) -> pd.DataFrame:
    """Load evaluation data from parquet."""
    data_path = dataset_dir / f"{eval_split}.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"Evaluation data not found: {data_path}")

    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df):,} samples from {data_path}")

    return df


def prepare_features(
    df: pd.DataFrame,
    feature_columns: list[str],
    label_column: str
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare features and labels."""
    # Validate columns
    missing_features = set(feature_columns) - set(df.columns)
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")
    if label_column not in df.columns:
        raise ValueError(f"Label column not found: {label_column}")

    X = df[feature_columns].copy()
    y = df[label_column].copy()

    # Fill NaN values with 0
    X = X.fillna(0)

    return X, y


def calculate_metrics(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    eval_split: str
) -> dict:
    """Calculate evaluation metrics."""
    logger.info("Calculating predictions...")
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    # Calculate metrics
    metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, y_prob)),
        "pr_auc": float(average_precision_score(y, y_prob)),
        "positive_rate": float(y.mean()),
        "eval_split": eval_split,
        "eval_samples": len(y)
    }

    return metrics


def save_metrics(metrics: dict, output_path: Path) -> None:
    """Save metrics to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Saved metrics to {output_path}")


def main() -> None:
    """Main entry point."""
    args = parse_args()

    logger.info("=" * 50)
    logger.info("Eval Component")
    logger.info("=" * 50)

    # Parse feature columns
    feature_columns = [c.strip() for c in args.feature_columns.split(",")]
    logger.info(f"Using {len(feature_columns)} features")

    # Load model
    model_dir = Path(args.input_model_dir)
    model = load_model(model_dir)

    # Load model metadata if available
    meta_path = model_dir / "model_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            model_meta = json.load(f)
        logger.info(f"Model type: {model_meta.get('model_type', 'unknown')}")

    # Load evaluation data
    dataset_dir = Path(args.input_dataset_dir)
    df = load_eval_data(dataset_dir, args.eval_split)

    # Prepare features and labels
    X, y = prepare_features(df, feature_columns, args.label_column)

    # Log label distribution
    positive_rate = y.mean()
    logger.info(f"Positive rate in {args.eval_split} set: {positive_rate:.2%}")
    if positive_rate < 0.05 or positive_rate > 0.95:
        logger.warning(f"Warning: Biased label distribution (positive rate: {positive_rate:.2%})")

    # Calculate metrics
    metrics = calculate_metrics(model, X, y, args.eval_split)

    # Save metrics
    output_path = Path(args.output_metrics_path)
    save_metrics(metrics, output_path)

    # Log results
    logger.info("=" * 50)
    logger.info(f"Evaluation Results ({args.eval_split} set)")
    logger.info("=" * 50)
    logger.info(f"Samples:      {metrics['eval_samples']:,}")
    logger.info(f"Positive Rate: {metrics['positive_rate']:.2%}")
    logger.info("-" * 30)
    logger.info(f"Accuracy:     {metrics['accuracy']:.4f}")
    logger.info(f"Precision:    {metrics['precision']:.4f}")
    logger.info(f"Recall:       {metrics['recall']:.4f}")
    logger.info(f"F1 Score:     {metrics['f1']:.4f}")
    logger.info(f"ROC-AUC:      {metrics['roc_auc']:.4f}")
    logger.info(f"PR-AUC:       {metrics['pr_auc']:.4f}")
    logger.info("=" * 50)
    logger.info("Evaluation completed successfully")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Component failed: {e}")
        sys.exit(1)
