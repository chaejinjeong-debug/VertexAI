"""
모델 업로드 스크립트
학습된 모델을 Vertex AI Model Registry에 업로드합니다.
"""

import argparse
import logging
from pathlib import Path

import yaml
from google.cloud import aiplatform, storage

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


def upload_to_gcs(
    local_dir: Path,
    bucket_name: str,
    gcs_prefix: str
) -> str:
    """로컬 디렉토리를 GCS에 업로드"""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    uploaded_files = []
    for local_file in local_dir.glob("**/*"):
        if local_file.is_file():
            relative_path = local_file.relative_to(local_dir)
            gcs_path = f"{gcs_prefix}/{relative_path}"
            blob = bucket.blob(gcs_path)
            blob.upload_from_filename(str(local_file))
            uploaded_files.append(gcs_path)
            logger.info(f"업로드: {local_file} -> gs://{bucket_name}/{gcs_path}")

    return f"gs://{bucket_name}/{gcs_prefix}"


def check_model_exists(
    project_id: str,
    region: str,
    model_name: str
) -> aiplatform.Model | None:
    """기존 모델 존재 여부 확인"""
    aiplatform.init(project=project_id, location=region)

    models = aiplatform.Model.list(
        filter=f'display_name="{model_name}"',
        order_by="create_time desc"
    )

    if models:
        return models[0]
    return None


def upload_model(
    project_id: str,
    region: str,
    model_config: dict,
    artifact_uri: str,
    skip_if_exists: bool = False
) -> aiplatform.Model:
    """Vertex AI Model Registry에 모델 업로드"""
    aiplatform.init(project=project_id, location=region)

    model_name = model_config["display_name"]

    # 기존 모델 확인
    if skip_if_exists:
        existing_model = check_model_exists(project_id, region, model_name)
        if existing_model:
            logger.info(f"모델이 이미 존재합니다: {existing_model.resource_name}")
            return existing_model

    logger.info(f"모델 업로드 시작: {model_name}")
    logger.info(f"  Artifact URI: {artifact_uri}")

    # 모델 업로드
    model = aiplatform.Model.upload(
        display_name=model_name,
        description=model_config.get("description", ""),
        artifact_uri=artifact_uri,
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest",
        labels={
            "framework": "sklearn",
            "task": "classification",
            "use_case": "churn_prediction"
        }
    )

    logger.info(f"모델 업로드 완료: {model.resource_name}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Vertex AI Model Registry에 모델 업로드")
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(ARTIFACTS_DIR / "model"),
        help="로컬 모델 디렉토리"
    )
    parser.add_argument(
        "--gcs-bucket",
        type=str,
        help="GCS 버킷 이름 (기본: {project_id}-vertex-models)"
    )
    parser.add_argument(
        "--skip-if-exists",
        action="store_true",
        help="이미 존재하면 스킵"
    )
    args = parser.parse_args()

    # 설정 로드
    env_config, training_config = load_configs()

    project_id = env_config["gcp"]["project_id"]
    region = env_config["gcp"]["region"]
    model_config = training_config["model"]

    # GCS 버킷 설정
    gcs_bucket = args.gcs_bucket or f"{project_id}-vertex-models"

    # 모델 디렉토리 확인
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        logger.error(f"모델 디렉토리가 없습니다: {model_dir}")
        logger.error("먼저 'uv run src/training/train.py'를 실행하세요.")
        return

    model_file = model_dir / "model.joblib"
    if not model_file.exists():
        logger.error(f"모델 파일이 없습니다: {model_file}")
        return

    # GCS 버킷 생성 (없으면)
    try:
        storage_client = storage.Client(project=project_id)
        bucket = storage_client.bucket(gcs_bucket)
        if not bucket.exists():
            logger.info(f"GCS 버킷 생성: {gcs_bucket}")
            bucket = storage_client.create_bucket(
                gcs_bucket,
                location=env_config["bigquery"]["location"]
            )
    except Exception as e:
        logger.warning(f"버킷 확인/생성 중 오류 (무시 가능): {e}")

    # GCS에 모델 업로드
    gcs_prefix = f"models/{model_config['name']}"
    artifact_uri = upload_to_gcs(model_dir, gcs_bucket, gcs_prefix)

    # Vertex AI Model Registry에 업로드
    model = upload_model(
        project_id=project_id,
        region=region,
        model_config=model_config,
        artifact_uri=artifact_uri,
        skip_if_exists=args.skip_if_exists
    )

    print(f"\n모델 업로드 완료!")
    print(f"  Model Name: {model.display_name}")
    print(f"  Model ID: {model.resource_name}")
    print(f"  Artifact URI: {artifact_uri}")


if __name__ == "__main__":
    main()
