"""
모델 학습 스크립트
BigQuery train_dataset에서 데이터를 로드하여 scikit-learn 모델을 학습합니다.
"""

import argparse
import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from google.cloud import bigquery
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def load_configs() -> tuple[dict, dict]:
    """환경 설정 및 학습 설정 로드"""
    with open(CONFIGS_DIR / "env.yaml", "r") as f:
        env_config = yaml.safe_load(f)
    with open(CONFIGS_DIR / "training.yaml", "r") as f:
        training_config = yaml.safe_load(f)
    return env_config, training_config


def load_training_data(
    client: bigquery.Client,
    project_id: str,
    dataset_id: str,
    table_id: str
) -> pd.DataFrame:
    """BigQuery에서 학습 데이터 로드"""
    query = f"""
    SELECT *
    FROM `{project_id}.{dataset_id}.{table_id}`
    """

    logger.info(f"학습 데이터 로드 중: {project_id}.{dataset_id}.{table_id}")
    df = client.query(query).to_dataframe()
    logger.info(f"로드된 데이터: {len(df):,} rows")

    return df


def prepare_features(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str
) -> tuple[pd.DataFrame, pd.Series]:
    """피처와 타겟 분리"""
    X = df[feature_columns].copy()
    y = df[target_column].copy()

    # NaN 처리
    X = X.fillna(0)

    return X, y


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    hyperparameters: dict
) -> RandomForestClassifier:
    """모델 학습"""
    model = RandomForestClassifier(
        n_estimators=hyperparameters.get("n_estimators", 100),
        max_depth=hyperparameters.get("max_depth", 10),
        min_samples_split=hyperparameters.get("min_samples_split", 5),
        min_samples_leaf=hyperparameters.get("min_samples_leaf", 2),
        random_state=42,
        n_jobs=-1
    )

    logger.info("모델 학습 시작...")
    model.fit(X_train, y_train)
    logger.info("모델 학습 완료")

    return model


def evaluate_model(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> dict:
    """모델 평가"""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "pr_auc": average_precision_score(y_test, y_prob),
    }

    return metrics


def print_evaluation_results(
    metrics: dict,
    y_train: pd.Series,
    y_test: pd.Series
):
    """평가 결과 출력"""
    logger.info("\n=== 학습 데이터 분포 ===")
    train_positive_rate = y_train.mean() * 100
    test_positive_rate = y_test.mean() * 100
    logger.info(f"  Train Churn Rate: {train_positive_rate:.2f}%")
    logger.info(f"  Test Churn Rate: {test_positive_rate:.2f}%")

    # 라벨 편향 경고
    if train_positive_rate < 5 or train_positive_rate > 95:
        logger.warning(f"  경고: 라벨이 편향되어 있습니다! (Churn Rate: {train_positive_rate:.2f}%)")

    logger.info("\n=== 모델 평가 결과 ===")
    logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall:    {metrics['recall']:.4f}")
    logger.info(f"  F1 Score:  {metrics['f1']:.4f}")
    logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    logger.info(f"  PR-AUC:    {metrics['pr_auc']:.4f}")


def save_model(
    model: RandomForestClassifier,
    feature_columns: list[str],
    output_dir: Path
) -> Path:
    """모델 및 메타데이터 저장"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 모델 저장
    model_path = output_dir / "model.joblib"
    joblib.dump(model, model_path)
    logger.info(f"모델 저장: {model_path}")

    # 피처 목록 저장 (서빙 시 사용)
    features_path = output_dir / "feature_columns.yaml"
    with open(features_path, "w") as f:
        yaml.dump({"feature_columns": feature_columns}, f)
    logger.info(f"피처 목록 저장: {features_path}")

    return model_path


def main():
    parser = argparse.ArgumentParser(description="Churn 예측 모델 학습")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ARTIFACTS_DIR / "model"),
        help="모델 저장 경로"
    )
    args = parser.parse_args()

    # 설정 로드
    env_config, training_config = load_configs()

    project_id = env_config["gcp"]["project_id"]
    target_dataset = env_config["bigquery"]["target_dataset"]

    feature_columns = training_config["training"]["feature_columns"]
    target_column = training_config["training"]["target_column"]
    test_size = training_config["training"]["test_size"]
    random_state = training_config["training"]["random_state"]
    hyperparameters = training_config["training"]["hyperparameters"]

    # BigQuery 클라이언트 초기화
    client = bigquery.Client(project=project_id)

    # 데이터 로드
    df = load_training_data(
        client=client,
        project_id=project_id,
        dataset_id=target_dataset,
        table_id="train_dataset"
    )

    # 피처/타겟 분리
    X, y = prepare_features(df, feature_columns, target_column)

    # Train/Test 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    logger.info(f"Train set: {len(X_train):,} samples")
    logger.info(f"Test set: {len(X_test):,} samples")

    # 모델 학습
    model = train_model(X_train, y_train, hyperparameters)

    # 모델 평가
    metrics = evaluate_model(model, X_test, y_test)
    print_evaluation_results(metrics, y_train, y_test)

    # 피처 중요도 출력
    logger.info("\n=== 피처 중요도 ===")
    importances = sorted(
        zip(feature_columns, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True
    )
    for feature, importance in importances:
        logger.info(f"  {feature}: {importance:.4f}")

    # 모델 저장
    output_dir = Path(args.output_dir)
    model_path = save_model(model, feature_columns, output_dir)

    # 메트릭 저장
    metrics_path = output_dir / "metrics.yaml"
    with open(metrics_path, "w") as f:
        yaml.dump(metrics, f)
    logger.info(f"메트릭 저장: {metrics_path}")

    print(f"\n모델 학습 완료!")
    print(f"  모델 경로: {model_path}")
    print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC: {metrics['pr_auc']:.4f}")


if __name__ == "__main__":
    main()
