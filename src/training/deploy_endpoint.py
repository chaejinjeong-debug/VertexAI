"""
엔드포인트 배포 스크립트
Vertex AI Model Registry의 모델을 Endpoint에 배포합니다.
"""

import argparse
import logging
from pathlib import Path

import yaml
from google.cloud import aiplatform

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"


def load_configs() -> tuple[dict, dict]:
    """환경 설정 및 학습 설정 로드"""
    with open(CONFIGS_DIR / "env.yaml", "r") as f:
        env_config = yaml.safe_load(f)
    with open(CONFIGS_DIR / "training.yaml", "r") as f:
        training_config = yaml.safe_load(f)
    return env_config, training_config


def get_latest_model(
    project_id: str,
    region: str,
    model_display_name: str
) -> aiplatform.Model | None:
    """최신 모델 조회"""
    aiplatform.init(project=project_id, location=region)

    models = aiplatform.Model.list(
        filter=f'display_name="{model_display_name}"',
        order_by="create_time desc"
    )

    if models:
        return models[0]
    return None


def get_or_create_endpoint(
    project_id: str,
    region: str,
    endpoint_name: str,
    endpoint_display_name: str
) -> aiplatform.Endpoint:
    """Endpoint 조회 또는 생성"""
    aiplatform.init(project=project_id, location=region)

    # 기존 Endpoint 조회
    endpoints = aiplatform.Endpoint.list(
        filter=f'display_name="{endpoint_display_name}"'
    )

    if endpoints:
        logger.info(f"기존 Endpoint 사용: {endpoints[0].resource_name}")
        return endpoints[0]

    # 새 Endpoint 생성
    logger.info(f"Endpoint 생성 시작: {endpoint_display_name}")
    endpoint = aiplatform.Endpoint.create(
        display_name=endpoint_display_name,
        labels={
            "task": "churn_prediction",
            "environment": "demo"
        }
    )

    logger.info(f"Endpoint 생성 완료: {endpoint.resource_name}")
    return endpoint


def deploy_model_to_endpoint(
    model: aiplatform.Model,
    endpoint: aiplatform.Endpoint,
    deployed_model_config: dict
) -> None:
    """모델을 Endpoint에 배포"""
    # 현재 배포된 모델 확인
    deployed_models = endpoint.list_models()

    # 기존 배포 undeploy (선택적)
    for deployed_model in deployed_models:
        logger.info(f"기존 배포 해제 중: {deployed_model.id}")
        endpoint.undeploy(deployed_model_id=deployed_model.id)

    # 새 모델 배포
    logger.info(f"모델 배포 시작: {model.display_name}")
    logger.info(f"  Machine Type: {deployed_model_config.get('machine_type', 'n1-standard-2')}")

    model.deploy(
        endpoint=endpoint,
        deployed_model_display_name=f"{model.display_name}-deployed",
        machine_type=deployed_model_config.get("machine_type", "n1-standard-2"),
        min_replica_count=deployed_model_config.get("min_replica_count", 1),
        max_replica_count=deployed_model_config.get("max_replica_count", 3),
        traffic_percentage=100,
        sync=True
    )

    logger.info("모델 배포 완료")


def main():
    parser = argparse.ArgumentParser(description="Vertex AI Endpoint에 모델 배포")
    parser.add_argument(
        "--model-name",
        type=str,
        help="배포할 모델 이름 (기본: config에서 로드)"
    )
    parser.add_argument(
        "--undeploy-existing",
        action="store_true",
        default=True,
        help="기존 배포 해제 후 새 모델 배포"
    )
    args = parser.parse_args()

    # 설정 로드
    env_config, training_config = load_configs()

    project_id = env_config["gcp"]["project_id"]
    region = env_config["gcp"]["region"]

    model_config = training_config["model"]
    endpoint_config = training_config["endpoint"]
    deployed_model_config = endpoint_config.get("deployed_model", {})

    # 모델 이름
    model_display_name = args.model_name or model_config["display_name"]

    # Vertex AI 초기화
    aiplatform.init(project=project_id, location=region)

    # 모델 조회
    model = get_latest_model(project_id, region, model_display_name)
    if not model:
        logger.error(f"모델을 찾을 수 없습니다: {model_display_name}")
        logger.error("먼저 'uv run src/training/upload_model.py'를 실행하세요.")
        return

    logger.info(f"모델 발견: {model.resource_name}")

    # Endpoint 조회/생성
    endpoint = get_or_create_endpoint(
        project_id=project_id,
        region=region,
        endpoint_name=endpoint_config["name"],
        endpoint_display_name=endpoint_config["display_name"]
    )

    # 모델 배포
    deploy_model_to_endpoint(
        model=model,
        endpoint=endpoint,
        deployed_model_config=deployed_model_config
    )

    print(f"\n배포 완료!")
    print(f"  Model: {model.display_name}")
    print(f"  Endpoint ID: {endpoint.resource_name}")
    print(f"  Endpoint Name: {endpoint.display_name}")


if __name__ == "__main__":
    main()
