"""
Vertex AI Pipeline Definition

Customer Churn 예측 모델 학습 파이프라인을 정의합니다.

Components:
1. data_load: BigQuery → Parquet (train/valid/test split)
2. train: Model training
3. eval: Model evaluation (valid set)
4. eval_test: Model evaluation (test set)
"""

from kfp import dsl
from kfp.dsl import Artifact, Input, Output

# Default configuration
DEFAULT_PROJECT_ID = "heum-alfred-evidence-clf-dev"
DEFAULT_REGION = "asia-northeast3"
DEFAULT_REPOSITORY = "vertex-ai-pipelines"

# Feature columns
FEATURE_COLUMNS = [
    "orders_30d",
    "orders_90d",
    "revenue_30d",
    "revenue_90d",
    "avg_order_value_90d",
    "distinct_products_90d",
    "distinct_categories_90d",
    "days_since_last_order",
]


def get_image_uri(
    component_name: str,
    project_id: str = DEFAULT_PROJECT_ID,
    region: str = DEFAULT_REGION,
    repository: str = DEFAULT_REPOSITORY,
    tag: str = "latest",
) -> str:
    """Get the container image URI for a component."""
    return f"{region}-docker.pkg.dev/{project_id}/{repository}/component-{component_name}:{tag}"


@dsl.component(base_image="python:3.11-slim")
def data_load_op(
    input_bq_table: str,
    label_column: str,
    time_column: str,
    train_ratio: float,
    valid_ratio: float,
    output_dataset: Output[Artifact],
) -> None:
    """Load data from BigQuery and split into train/valid/test."""
    import json
    import subprocess
    import sys

    # Install dependencies
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "google-cloud-bigquery", "pandas", "pyarrow", "db-dtypes"])

    from pathlib import Path
    import pandas as pd
    from google.cloud import bigquery

    output_dir = Path(output_dataset.path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load from BigQuery
    print(f"Loading data from BigQuery: {input_bq_table}")
    client = bigquery.Client()
    query = f"SELECT * FROM `{input_bq_table}`"
    df = client.query(query).to_dataframe()
    print(f"Loaded {len(df):,} rows")

    # Time-based split
    df_sorted = df.sort_values(time_column).reset_index(drop=True)
    n = len(df_sorted)
    train_end = int(n * train_ratio)
    valid_end = int(n * (train_ratio + valid_ratio))

    train_df = df_sorted.iloc[:train_end]
    valid_df = df_sorted.iloc[train_end:valid_end]
    test_df = df_sorted.iloc[valid_end:]

    print(f"Split: train={len(train_df):,}, valid={len(valid_df):,}, test={len(test_df):,}")

    # Save parquet files
    train_df.to_parquet(output_dir / "train.parquet", index=False)
    valid_df.to_parquet(output_dir / "valid.parquet", index=False)
    test_df.to_parquet(output_dir / "test.parquet", index=False)

    # Save metadata
    metadata = {
        "train_samples": len(train_df),
        "valid_samples": len(valid_df),
        "test_samples": len(test_df),
        "label_column": label_column,
        "time_column": time_column,
        "train_positive_rate": float(train_df[label_column].mean()),
        "valid_positive_rate": float(valid_df[label_column].mean()),
        "test_positive_rate": float(test_df[label_column].mean()),
    }
    with open(output_dir / "dataset_meta.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("Data load completed")


@dsl.component(base_image="python:3.11-slim")
def train_op(
    input_dataset: Input[Artifact],
    feature_columns: str,
    label_column: str,
    model_type: str,
    n_estimators: int,
    max_depth: int,
    random_state: int,
    output_model: Output[Artifact],
) -> None:
    """Train a classification model."""
    import json
    import subprocess
    import sys

    # Install dependencies
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "scikit-learn", "pandas", "pyarrow", "joblib"])

    from pathlib import Path
    import joblib
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    input_dir = Path(input_dataset.path)
    output_dir = Path(output_model.path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load training data
    train_df = pd.read_parquet(input_dir / "train.parquet")
    print(f"Loaded {len(train_df):,} training samples")

    # Prepare features
    features = [c.strip() for c in feature_columns.split(",")]
    X = train_df[features].fillna(0)
    y = train_df[label_column]

    print(f"Features: {features}")
    print(f"Label distribution: {y.value_counts().to_dict()}")

    # Create model
    if model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        model = LogisticRegression(random_state=random_state, max_iter=1000)

    # Train
    print(f"Training {model_type}...")
    model.fit(X, y)
    print("Training completed")

    # Save model
    joblib.dump(model, output_dir / "model.pkl")

    # Save metadata
    metadata = {
        "model_type": model_type,
        "feature_columns": features,
        "label_column": label_column,
    }
    if model_type == "random_forest":
        metadata["feature_importances"] = dict(zip(features, [float(x) for x in model.feature_importances_]))

    with open(output_dir / "model_meta.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Model saved to {output_dir}")


@dsl.component(base_image="python:3.11-slim")
def eval_op(
    input_model: Input[Artifact],
    input_dataset: Input[Artifact],
    feature_columns: str,
    label_column: str,
    eval_split: str,
    output_metrics: Output[Artifact],
) -> None:
    """Evaluate the model on validation or test set."""
    import json
    import subprocess
    import sys

    # Install dependencies
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "scikit-learn", "pandas", "pyarrow", "joblib"])

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

    model_dir = Path(input_model.path)
    dataset_dir = Path(input_dataset.path)
    output_path = Path(output_metrics.path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load model
    model = joblib.load(model_dir / "model.pkl")
    print(f"Loaded model from {model_dir}")

    # Load data
    df = pd.read_parquet(dataset_dir / f"{eval_split}.parquet")
    print(f"Loaded {len(df):,} {eval_split} samples")

    # Prepare features
    features = [c.strip() for c in feature_columns.split(",")]
    X = df[features].fillna(0)
    y = df[label_column]

    # Predict
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    # Calculate metrics
    metrics = {
        "eval_split": eval_split,
        "eval_samples": len(y),
        "positive_rate": float(y.mean()),
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, y_prob)),
        "pr_auc": float(average_precision_score(y, y_prob)),
    }

    # Save metrics
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Evaluation Results ({eval_split}):")
    print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC:  {metrics['pr_auc']:.4f}")
    print(f"  F1:      {metrics['f1']:.4f}")


@dsl.pipeline(
    name="customer-churn-training-pipeline",
    description="Customer Churn 예측 모델 학습 파이프라인",
)
def churn_training_pipeline(
    input_bq_table: str = "heum-alfred-evidence-clf-dev.featurestore_demo.train_dataset",
    label_column: str = "label_churn_60d",
    time_column: str = "label_timestamp",
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
    model_type: str = "random_forest",
    n_estimators: int = 100,
    max_depth: int = 10,
    random_state: int = 42,
) -> None:
    """
    End-to-end training pipeline for customer churn prediction.

    Args:
        input_bq_table: BigQuery table path for training data
        label_column: Name of the label column
        time_column: Name of the timestamp column for time-based split
        train_ratio: Proportion of data for training
        valid_ratio: Proportion of data for validation
        model_type: Type of model (random_forest or logistic_regression)
        n_estimators: Number of trees for RandomForest
        max_depth: Maximum tree depth
        random_state: Random seed for reproducibility
    """
    feature_columns_str = ",".join(FEATURE_COLUMNS)

    # Step 1: Load and split data
    data_load_task = data_load_op(
        input_bq_table=input_bq_table,
        label_column=label_column,
        time_column=time_column,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
    )
    data_load_task.set_display_name("Data Load")

    # Step 2: Train model
    train_task = train_op(
        input_dataset=data_load_task.outputs["output_dataset"],
        feature_columns=feature_columns_str,
        label_column=label_column,
        model_type=model_type,
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )
    train_task.set_display_name("Train Model")

    # Step 3: Evaluate on validation set
    eval_valid_task = eval_op(
        input_model=train_task.outputs["output_model"],
        input_dataset=data_load_task.outputs["output_dataset"],
        feature_columns=feature_columns_str,
        label_column=label_column,
        eval_split="valid",
    )
    eval_valid_task.set_display_name("Evaluate (Valid)")

    # Step 4: Evaluate on test set
    eval_test_task = eval_op(
        input_model=train_task.outputs["output_model"],
        input_dataset=data_load_task.outputs["output_dataset"],
        feature_columns=feature_columns_str,
        label_column=label_column,
        eval_split="test",
    )
    eval_test_task.set_display_name("Evaluate (Test)")


if __name__ == "__main__":
    # For testing: print pipeline info
    print("Pipeline: customer-churn-training-pipeline")
    print(f"Feature columns: {FEATURE_COLUMNS}")
