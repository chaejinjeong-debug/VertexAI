"""
Feature View 생성
Feature Group을 기반으로 Online Store에 연결된 Feature View를 생성합니다.
"""

import argparse
import logging
from pathlib import Path

import yaml
from google.cloud import aiplatform
from google.cloud.aiplatform_v1 import (
    FeatureOnlineStoreAdminServiceClient,
    FeatureView,
    CreateFeatureViewRequest,
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


def check_feature_view_exists(
    client: FeatureOnlineStoreAdminServiceClient,
    project_id: str,
    region: str,
    online_store_name: str,
    feature_view_name: str
) -> bool:
    """Feature View 존재 여부 확인"""
    parent = f"projects/{project_id}/locations/{region}/featureOnlineStores/{online_store_name}"

    try:
        views = client.list_feature_views(parent=parent)
        for view in views:
            if view.name.endswith(f"/{feature_view_name}"):
                return True
        return False
    except Exception as e:
        logger.error(f"Feature View 목록 조회 실패: {e}")
        return False


def create_feature_view(
    project_id: str,
    region: str,
    online_store_name: str,
    feature_group_name: str,
    feature_view_config: dict,
    skip_if_exists: bool = True
) -> str:
    """Feature View 생성"""
    client = get_online_store_client(region)
    feature_view_name = feature_view_config["name"]

    # 존재 여부 확인
    if skip_if_exists and check_feature_view_exists(
        client, project_id, region, online_store_name, feature_view_name
    ):
        logger.info(f"Feature View 이미 존재: {feature_view_name}")
        return f"projects/{project_id}/locations/{region}/featureOnlineStores/{online_store_name}/featureViews/{feature_view_name}"

    parent = f"projects/{project_id}/locations/{region}/featureOnlineStores/{online_store_name}"

    # 피처 목록
    features = feature_view_config.get("features", [])

    # Sync 설정
    sync_config = feature_view_config.get("sync_config", {})
    cron_schedule = sync_config.get("cron", "0 2 * * *")

    # Feature View 생성
    # NOTE: feature_group_id는 간단한 이름만 사용 (전체 경로 아님)
    logger.info(f"  - Feature Group ID: {feature_group_name}")
    feature_view = FeatureView(
        feature_registry_source=FeatureView.FeatureRegistrySource(
            feature_groups=[
                FeatureView.FeatureRegistrySource.FeatureGroup(
                    feature_group_id=feature_group_name,
                    feature_ids=features,
                )
            ]
        ),
        sync_config=FeatureView.SyncConfig(
            cron=cron_schedule,
        ),
    )

    request = CreateFeatureViewRequest(
        parent=parent,
        feature_view=feature_view,
        feature_view_id=feature_view_name,
    )

    logger.info(f"Feature View 생성 시작: {feature_view_name}")
    logger.info(f"  - Online Store: {online_store_name}")
    logger.info(f"  - Feature Group: {feature_group_name}")
    logger.info(f"  - 피처: {features}")
    logger.info(f"  - Sync 스케줄: {cron_schedule}")

    try:
        operation = client.create_feature_view(request=request)
        logger.info("Feature View 생성 중...")

        result = operation.result()
        logger.info(f"Feature View 생성 완료: {result.name}")
        return result.name

    except gcp_exceptions.AlreadyExists:
        logger.info(f"Feature View 이미 존재: {feature_view_name}")
        return f"projects/{project_id}/locations/{region}/featureOnlineStores/{online_store_name}/featureViews/{feature_view_name}"
    except Exception as e:
        logger.error(f"Feature View 생성 실패: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Vertex AI Feature View 생성")
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
    online_store_name = fs_config["online_store"]["name"]
    feature_group_name = fs_config["feature_group"]["name"]
    feature_view_config = fs_config["feature_view"]

    # Vertex AI 초기화
    aiplatform.init(project=project_id, location=region)

    # Feature View 생성
    feature_view_path = create_feature_view(
        project_id=project_id,
        region=region,
        online_store_name=online_store_name,
        feature_group_name=feature_group_name,
        feature_view_config=feature_view_config,
        skip_if_exists=not args.force
    )

    print(f"\nFeature View: {feature_view_path}")


if __name__ == "__main__":
    main()
