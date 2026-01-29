"""
Train Component

Parquet 데이터에서 모델을 학습합니다.

Usage:
    python main.py --help
    python main.py \
        --input_dataset_dir /path/to/dataset \
        --output_model_dir /path/to/model \
        --feature_columns "orders_30d,orders_90d,revenue_30d" \
        --label_column label_churn_60d
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

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
        description="Train Component - Model Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input arguments
    parser.add_argument(
        "--input_dataset_dir",
        type=str,
        required=True,
        help="Input dataset directory (contains train.parquet)"
    )

    # Output arguments
    parser.add_argument(
        "--output_model_dir",
        type=str,
        required=True,
        help="Output model directory"
    )

    # Model configuration
    parser.add_argument(
        "--model_type",
        type=str,
        default="random_forest",
        choices=["random_forest", "logistic_regression"],
        help="Model type to train"
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

    # Hyperparameters
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=100,
        help="Number of trees (for random_forest)"
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=10,
        help="Maximum tree depth"
    )
    parser.add_argument(
        "--min_samples_split",
        type=int,
        default=5,
        help="Minimum samples to split a node"
    )
    parser.add_argument(
        "--min_samples_leaf",
        type=int,
        default=2,
        help="Minimum samples in a leaf node"
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    return parser.parse_args()


def load_train_data(dataset_dir: Path) -> pd.DataFrame:
    """Load training data from parquet."""
    train_path = dataset_dir / "train.parquet"
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")

    df = pd.read_parquet(train_path)
    logger.info(f"Loaded {len(df):,} training samples from {train_path}")

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

    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Label distribution: {y.value_counts().to_dict()}")

    return X, y


def create_model(args: argparse.Namespace):
    """Create model based on configuration."""
    if args.model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.random_state,
            n_jobs=-1
        )
        logger.info(f"Created RandomForestClassifier with {args.n_estimators} estimators")
    else:
        model = LogisticRegression(
            random_state=args.random_state,
            max_iter=1000,
            n_jobs=-1
        )
        logger.info("Created LogisticRegression model")

    return model


def train_model(model, X: pd.DataFrame, y: pd.Series):
    """Train the model."""
    logger.info("Starting model training...")
    model.fit(X, y)
    logger.info("Model training completed")
    return model


def save_model(
    model,
    feature_columns: list[str],
    label_column: str,
    model_type: str,
    output_dir: Path
) -> None:
    """Save model and metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = output_dir / "model.pkl"
    joblib.dump(model, model_path)
    logger.info(f"Saved model to {model_path}")

    # Create metadata
    metadata = {
        "model_type": model_type,
        "feature_columns": feature_columns,
        "label_column": label_column,
        "sklearn_version": "1.5.2"
    }

    # Add model-specific metadata
    if model_type == "random_forest":
        metadata["n_estimators"] = model.n_estimators
        metadata["max_depth"] = model.max_depth
        metadata["feature_importances"] = dict(zip(
            feature_columns,
            [float(x) for x in model.feature_importances_]
        ))

    # Save metadata
    meta_path = output_dir / "model_meta.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {meta_path}")


def main() -> None:
    """Main entry point."""
    args = parse_args()

    logger.info("=" * 50)
    logger.info("Train Component")
    logger.info("=" * 50)

    # Parse feature columns
    feature_columns = [c.strip() for c in args.feature_columns.split(",")]
    logger.info(f"Using {len(feature_columns)} features")

    # Load training data
    dataset_dir = Path(args.input_dataset_dir)
    df = load_train_data(dataset_dir)

    # Prepare features and labels
    X, y = prepare_features(df, feature_columns, args.label_column)

    # Log label distribution
    positive_rate = y.mean()
    logger.info(f"Positive rate: {positive_rate:.2%}")
    if positive_rate < 0.05 or positive_rate > 0.95:
        logger.warning(f"Warning: Biased label distribution (positive rate: {positive_rate:.2%})")

    # Create and train model
    model = create_model(args)
    model = train_model(model, X, y)

    # Save model
    output_dir = Path(args.output_model_dir)
    save_model(
        model=model,
        feature_columns=feature_columns,
        label_column=args.label_column,
        model_type=args.model_type,
        output_dir=output_dir
    )

    # Log feature importances for RandomForest
    if args.model_type == "random_forest":
        logger.info("Feature importances:")
        importances = sorted(
            zip(feature_columns, model.feature_importances_),
            key=lambda x: x[1],
            reverse=True
        )
        for feature, importance in importances:
            logger.info(f"  {feature}: {importance:.4f}")

    logger.info("=" * 50)
    logger.info("Training completed successfully")
    logger.info(f"Model saved to: {output_dir}")
    logger.info("=" * 50)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Component failed: {e}")
        sys.exit(1)
