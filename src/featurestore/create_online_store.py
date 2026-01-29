"""
Online Store 생성
Vertex AI Feature Store의 온라인 서빙 저장소를 생성합니다.
"""

import argparse
import logging
from pathlib import Path

import yaml
from google.cloud import aiplatform
from google.cloud.aiplatform_v1 import (
    FeatureOnlineStoreAdminServiceClient,
    FeatureOnlineStore,
    CreateFeatureOnlineStoreRequest,
)
from google.api_core import exceptions as gcp_exceptions

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
    """환경 설정 및 Feature Store 설정 로드"""
    with open(CONFIGS_DIR / "env.yaml", "r") as f:
        env_config = yaml.safe_load(f)
    with open(CONFIGS_DIR / "featurestore.yaml", "r") as f:
        fs_config = yaml.safe_load(f)
    return env_config, fs_config


def get_online_store_client(region: str) -> FeatureOnlineStoreAdminServiceClient:
    """Feature Online Store Admin 클라이언트 생성"""
    api_endpoint = f"{region}-aiplatform.googleapis.com"
    return FeatureOnlineStoreAdminServiceClient(
        client_options={"api_endpoint": api_endpoint}
    )


def check_online_store_exists(
    client: FeatureOnlineStoreAdminServiceClient,
    project_id: str,
    region: str,
    online_store_name: str
) -> bool:
    """Online Store 존재 여부 확인"""
    parent = f"projects/{project_id}/locations/{region}"

    try:
        stores = client.list_feature_online_stores(parent=parent)
        for store in stores:
            if store.name.endswith(f"/{online_store_name}"):
                return True
        return False
    except Exception as e:
        logger.error(f"Online Store 목록 조회 실패: {e}")
        return False


def create_online_store(
    project_id: str,
    region: str,
    online_store_config: dict,
    skip_if_exists: bool = True
) -> str:
    """Online Store 생성"""
    client = get_online_store_client(region)
    online_store_name = online_store_config["name"]

    # 존재 여부 확인
    if skip_if_exists and check_online_store_exists(client, project_id, region, online_store_name):
        logger.info(f"Online Store 이미 존재: {online_store_name}")
        return f"projects/{project_id}/locations/{region}/featureOnlineStores/{online_store_name}"

    parent = f"projects/{project_id}/locations/{region}"

    # Bigtable 기반 Online Store 설정
    bigtable_config = online_store_config.get("bigtable", {})
    auto_scaling = bigtable_config.get("auto_scaling", {})

    online_store = FeatureOnlineStore(
        bigtable=FeatureOnlineStore.Bigtable(
            auto_scaling=FeatureOnlineStore.Bigtable.AutoScaling(
                min_node_count=auto_scaling.get("min_node_count", 1),
                max_node_count=auto_scaling.get("max_node_count", 3),
                cpu_utilization_target=auto_scaling.get("cpu_utilization_target", 50),
            )
        )
    )

    request = CreateFeatureOnlineStoreRequest(
        parent=parent,
        feature_online_store=online_store,
        feature_online_store_id=online_store_name,
    )

    logger.info(f"Online Store 생성 시작: {online_store_name}")
    logger.info(f"  - 리전: {region}")
    logger.info(f"  - Bigtable 노드: {auto_scaling.get('min_node_count', 1)} ~ {auto_scaling.get('max_node_count', 3)}")

    try:
        operation = client.create_feature_online_store(request=request)
        logger.info("Online Store 생성 중... (수 분 소요될 수 있음)")

        result = operation.result()
        logger.info(f"Online Store 생성 완료: {result.name}")
        return result.name

    except gcp_exceptions.AlreadyExists:
        logger.info(f"Online Store 이미 존재: {online_store_name}")
        return f"projects/{project_id}/locations/{region}/featureOnlineStores/{online_store_name}"
    except Exception as e:
        logger.error(f"Online Store 생성 실패: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Vertex AI Online Store 생성")
    parser.add_argument(
        "--force",
        action="store_true",
        help="이미 존재해도 재생성 시도"
    )
    args = parser.parse_args()

    # 설정 로드
    env_config, fs_config = load_configs()

    project_id = env_config["gcp"]["project_id"]
    region = env_config["gcp"]["region"]
    online_store_config = fs_config["online_store"]

    # Vertex AI 초기화
    aiplatform.init(project=project_id, location=region)

    # Online Store 생성
    online_store_name = create_online_store(
        project_id=project_id,
        region=region,
        online_store_config=online_store_config,
        skip_if_exists=not args.force
    )

    print(f"\nOnline Store: {online_store_name}")


if __name__ == "__main__":
    main()
