"""
Vertex AI Pipeline Definition (Container-based Components)

Customer Churn 예측 모델 학습 파이프라인을 정의합니다.
src/components/ 디렉토리의 컨테이너 이미지를 사용합니다.

Components:
1. data_load: BigQuery → Parquet (train/valid/test split)
2. train: Model training
3. eval: Model evaluation (valid set)
4. eval_test: Model evaluation (test set)
5. model_upload: Model Registry 업로드 + Experiments 로깅
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


@dsl.container_component
def data_load_op(
    input_bq_table: str,
    label_column: str,
    time_column: str,
    train_ratio: float,
    valid_ratio: float,
    output_dataset: Output[Artifact],
):
    """Load data from BigQuery and split into train/valid/test."""
    return dsl.ContainerSpec(
        image=get_image_uri("data_load"),
        command=["python", "/app/src/main.py"],
        args=[
            "--input_bq_table", input_bq_table,
            "--output_dataset_dir", output_dataset.path,
            "--label_column", label_column,
            "--time_column", time_column,
            "--train_ratio", str(train_ratio),
            "--valid_ratio", str(valid_ratio),
        ],
    )


@dsl.container_component
def train_op(
    input_dataset: Input[Artifact],
    feature_columns: str,
    label_column: str,
    model_type: str,
    n_estimators: int,
    max_depth: int,
    random_state: int,
    output_model: Output[Artifact],
):
    """Train a classification model."""
    return dsl.ContainerSpec(
        image=get_image_uri("train"),
        command=["python", "/app/src/main.py"],
        args=[
            "--input_dataset_dir", input_dataset.path,
            "--output_model_dir", output_model.path,
            "--feature_columns", feature_columns,
            "--label_column", label_column,
            "--model_type", model_type,
            "--n_estimators", str(n_estimators),
            "--max_depth", str(max_depth),
            "--random_state", str(random_state),
        ],
    )


@dsl.container_component
def eval_op(
    input_model: Input[Artifact],
    input_dataset: Input[Artifact],
    feature_columns: str,
    label_column: str,
    eval_split: str,
    output_metrics: Output[Artifact],
):
    """Evaluate the model on validation or test set."""
    return dsl.ContainerSpec(
        image=get_image_uri("eval"),
        command=["python", "/app/src/main.py"],
        args=[
            "--input_model_dir", input_model.path,
            "--input_dataset_dir", input_dataset.path,
            "--output_metrics_path", output_metrics.path,
            "--feature_columns", feature_columns,
            "--label_column", label_column,
            "--eval_split", eval_split,
        ],
    )


@dsl.container_component
def model_upload_op(
    input_model: Input[Artifact],
    input_metrics: Input[Artifact],
    project_id: str,
    region: str,
    experiment_name: str,
    model_display_name: str,
    output_model_resource: Output[Artifact],
):
    """Upload model to Model Registry and log to Experiments.

    Uses pipeline artifact path directly (no separate GCS upload needed).
    """
    return dsl.ContainerSpec(
        image=get_image_uri("model_upload"),
        command=["python", "/app/src/main.py"],
        args=[
            "--input_model_dir", input_model.path,
            "--input_metrics_path", input_metrics.path,
            "--project_id", project_id,
            "--region", region,
            "--experiment_name", experiment_name,
            "--model_display_name", model_display_name,
            "--output_model_resource", output_model_resource.path,
        ],
    )


@dsl.pipeline(
    name="customer-churn-training-pipeline",
    description="Customer Churn 예측 모델 학습 파이프라인 (Container-based)",
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
    # Model Registry & Experiments settings
    project_id: str = DEFAULT_PROJECT_ID,
    region: str = DEFAULT_REGION,
    experiment_name: str = "churn-experiment",
    model_display_name: str = "churn-model",
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
        project_id: GCP project ID
        region: GCP region
        experiment_name: Vertex AI Experiment name for tracking
        model_display_name: Display name for Model Registry
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

    # Step 5: Upload model to Model Registry and log to Experiments
    model_upload_task = model_upload_op(
        input_model=train_task.outputs["output_model"],
        input_metrics=eval_valid_task.outputs["output_metrics"],
        project_id=project_id,
        region=region,
        experiment_name=experiment_name,
        model_display_name=model_display_name,
    )
    model_upload_task.set_display_name("Model Upload")


if __name__ == "__main__":
    # For testing: print pipeline info
    print("Pipeline: customer-churn-training-pipeline (Container-based)")
    print(f"Feature columns: {FEATURE_COLUMNS}")
    print(f"Data Load Image: {get_image_uri('data_load')}")
    print(f"Train Image: {get_image_uri('train')}")
    print(f"Eval Image: {get_image_uri('eval')}")
    print(f"Model Upload Image: {get_image_uri('model_upload')}")
