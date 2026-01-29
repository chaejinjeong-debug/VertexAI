"""
온라인 피처 조회
Feature View에서 실시간으로 피처를 조회합니다.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import yaml
from google.cloud import aiplatform
from google.cloud.aiplatform_v1 import (
    FeatureOnlineStoreServiceClient,
    FetchFeatureValuesRequest,
)
from google.cloud.aiplatform_v1.types import FeatureViewDataKey

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


def get_feature_online_store_client(region: str) -> FeatureOnlineStoreServiceClient:
    """Feature Online Store Service 클라이언트 생성"""
    api_endpoint = f"{region}-aiplatform.googleapis.com"
    return FeatureOnlineStoreServiceClient(
        client_options={"api_endpoint": api_endpoint}
    )


def fetch_features(
    project_id: str,
    region: str,
    online_store_name: str,
    feature_view_name: str,
    entity_id: str,
    feature_names: Optional[list[str]] = None
) -> dict:
    """
    온라인 피처 조회

    Args:
        project_id: GCP 프로젝트 ID
        region: 리전
        online_store_name: Online Store 이름
        feature_view_name: Feature View 이름
        entity_id: 엔티티 ID (customer_id)
        feature_names: 조회할 피처 목록 (None이면 전체)

    Returns:
        피처 딕셔너리
    """
    client = get_feature_online_store_client(region)

    feature_view_path = (
        f"projects/{project_id}/locations/{region}/"
        f"featureOnlineStores/{online_store_name}/featureViews/{feature_view_name}"
    )

    # 요청 구성
    request = FetchFeatureValuesRequest(
        feature_view=feature_view_path,
        data_key=FeatureViewDataKey(
            key=str(entity_id),
        ),
    )

    logger.debug(f"피처 조회: entity_id={entity_id}")

    try:
        response = client.fetch_feature_values(request=request)

        # 응답에서 피처 값 추출
        features = {}

        # key_values 형식으로 응답 파싱 (features 필드 사용)
        if response.key_values and response.key_values.features:
            for kv in response.key_values.features:
                feature_name = kv.name

                # 피처 필터링
                if feature_names and feature_name not in feature_names:
                    continue

                # 값 타입에 따라 추출 (proto의 oneof 필드 확인)
                value = kv.value
                pb = value._pb

                # proto의 HasField로 어떤 필드가 설정되었는지 확인
                if pb.HasField("int64_value"):
                    features[feature_name] = value.int64_value
                elif pb.HasField("double_value"):
                    features[feature_name] = value.double_value
                elif pb.HasField("string_value"):
                    features[feature_name] = value.string_value
                elif pb.HasField("bool_value"):
                    features[feature_name] = value.bool_value
                else:
                    features[feature_name] = None

        return features

    except Exception as e:
        logger.error(f"피처 조회 실패: {e}")
        raise


def fetch_features_batch(
    project_id: str,
    region: str,
    online_store_name: str,
    feature_view_name: str,
    entity_ids: list[str],
    feature_names: Optional[list[str]] = None
) -> list[dict]:
    """
    여러 엔티티의 피처를 일괄 조회

    Args:
        entity_ids: 엔티티 ID 목록

    Returns:
        피처 딕셔너리 목록
    """
    results = []

    for entity_id in entity_ids:
        try:
            features = fetch_features(
                project_id=project_id,
                region=region,
                online_store_name=online_store_name,
                feature_view_name=feature_view_name,
                entity_id=entity_id,
                feature_names=feature_names
            )
            results.append({
                "entity_id": entity_id,
                "features": features,
                "status": "success"
            })
        except Exception as e:
            results.append({
                "entity_id": entity_id,
                "features": None,
                "status": "error",
                "error": str(e)
            })

    return results


def main():
    parser = argparse.ArgumentParser(description="Vertex AI Feature View에서 피처 조회")
    parser.add_argument(
        "--customer-id",
        type=str,
        required=True,
        help="조회할 고객 ID"
    )
    parser.add_argument(
        "--features",
        type=str,
        nargs="+",
        help="조회할 피처 목록 (지정하지 않으면 전체)"
    )
    args = parser.parse_args()

    # 설정 로드
    env_config, fs_config = load_configs()

    project_id = env_config["gcp"]["project_id"]
    region = env_config["gcp"]["region"]
    online_store_name = fs_config["online_store"]["name"]
    feature_view_name = fs_config["feature_view"]["name"]

    # Vertex AI 초기화
    aiplatform.init(project=project_id, location=region)

    # 피처 조회
    features = fetch_features(
        project_id=project_id,
        region=region,
        online_store_name=online_store_name,
        feature_view_name=feature_view_name,
        entity_id=args.customer_id,
        feature_names=args.features
    )

    print(f"\n고객 ID: {args.customer_id}")
    print("피처:")
    for name, value in features.items():
        print(f"  {name}: {value}")


if __name__ == "__main__":
    main()
