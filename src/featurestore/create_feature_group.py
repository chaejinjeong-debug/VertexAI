"""
Feature Group 생성
BigQuery 테이블을 소스로 하는 Feature Group을 생성합니다.
"""

import argparse
import logging
from pathlib import Path

import yaml
from google.cloud import aiplatform
from google.cloud.aiplatform_v1 import (
    FeatureRegistryServiceClient,
    FeatureGroup,
    CreateFeatureGroupRequest,
)
from google.cloud.aiplatform_v1.types import io as aiplatform_io
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


def get_feature_registry_client(region: str) -> FeatureRegistryServiceClient:
    """Feature Registry 클라이언트 생성"""
    api_endpoint = f"{region}-aiplatform.googleapis.com"
    return FeatureRegistryServiceClient(
        client_options={"api_endpoint": api_endpoint}
    )


def check_feature_group_exists(
    client: FeatureRegistryServiceClient,
    project_id: str,
    region: str,
    feature_group_name: str
) -> bool:
    """Feature Group 존재 여부 확인"""
    parent = f"projects/{project_id}/locations/{region}"

    try:
        groups = client.list_feature_groups(parent=parent)
        for group in groups:
            if group.name.endswith(f"/{feature_group_name}"):
                return True
        return False
    except Exception as e:
        logger.error(f"Feature Group 목록 조회 실패: {e}")
        return False


def create_feature_group(
    project_id: str,
    region: str,
    feature_group_config: dict,
    bigquery_config: dict,
    skip_if_exists: bool = True
) -> str:
    """Feature Group 생성"""
    client = get_feature_registry_client(region)
    feature_group_name = feature_group_config["name"]

    # 존재 여부 확인
    if skip_if_exists and check_feature_group_exists(client, project_id, region, feature_group_name):
        logger.info(f"Feature Group 이미 존재: {feature_group_name}")
        return f"projects/{project_id}/locations/{region}/featureGroups/{feature_group_name}"

    parent = f"projects/{project_id}/locations/{region}"

    # BigQuery 소스 테이블 URI 구성
    target_dataset = bigquery_config["target_dataset"]
    features_table = bigquery_config["tables"]["features_customer"]
    bq_uri = f"bq://{project_id}.{target_dataset}.{features_table}"

    # entity_id_columns 설정
    entity_id_columns = feature_group_config["big_query_source"]["entity_id_columns"]

    # Feature Group 생성
    feature_group = FeatureGroup(
        big_query=FeatureGroup.BigQuery(
            big_query_source=aiplatform_io.BigQuerySource(
                input_uri=bq_uri,
            ),
            entity_id_columns=entity_id_columns,
        ),
        description=feature_group_config.get("description", ""),
    )

    request = CreateFeatureGroupRequest(
        parent=parent,
        feature_group=feature_group,
        feature_group_id=feature_group_name,
    )

    logger.info(f"Feature Group 생성 시작: {feature_group_name}")
    logger.info(f"  - BigQuery 소스: {bq_uri}")
    logger.info(f"  - Entity ID 컬럼: {entity_id_columns}")

    try:
        operation = client.create_feature_group(request=request)
        logger.info("Feature Group 생성 중...")

        result = operation.result()
        logger.info(f"Feature Group 생성 완료: {result.name}")
        return result.name

    except gcp_exceptions.AlreadyExists:
        logger.info(f"Feature Group 이미 존재: {feature_group_name}")
        return f"projects/{project_id}/locations/{region}/featureGroups/{feature_group_name}"
    except Exception as e:
        logger.error(f"Feature Group 생성 실패: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Vertex AI Feature Group 생성")
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
    feature_group_config = fs_config["feature_group"]
    bigquery_config = env_config["bigquery"]

    # Vertex AI 초기화
    aiplatform.init(project=project_id, location=region)

    # Feature Group 생성
    feature_group_name = create_feature_group(
        project_id=project_id,
        region=region,
        feature_group_config=feature_group_config,
        bigquery_config=bigquery_config,
        skip_if_exists=not args.force
    )

    print(f"\nFeature Group: {feature_group_name}")


if __name__ == "__main__":
    main()
