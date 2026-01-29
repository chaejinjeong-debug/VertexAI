"""
Feature View 동기화 트리거
Feature View의 데이터를 Online Store로 동기화합니다.
"""

import argparse
import logging
import time
from pathlib import Path

import yaml
from google.cloud import aiplatform
from google.cloud.aiplatform_v1 import (
    FeatureOnlineStoreAdminServiceClient,
    SyncFeatureViewRequest,
    GetFeatureViewSyncRequest,
)

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


def trigger_sync(
    project_id: str,
    region: str,
    online_store_name: str,
    feature_view_name: str,
    wait_for_completion: bool = True,
    timeout_minutes: int = 30
) -> str:
    """Feature View 동기화 트리거"""
    client = get_online_store_client(region)

    feature_view_path = (
        f"projects/{project_id}/locations/{region}/"
        f"featureOnlineStores/{online_store_name}/featureViews/{feature_view_name}"
    )

    request = SyncFeatureViewRequest(
        feature_view=feature_view_path,
    )

    logger.info(f"Feature View 동기화 시작: {feature_view_name}")
    logger.info(f"  - Online Store: {online_store_name}")

    try:
        response = client.sync_feature_view(request=request)
        sync_name = response.feature_view_sync
        logger.info(f"동기화 작업 시작됨: {sync_name}")

        if wait_for_completion:
            logger.info("동기화 완료 대기 중...")
            sync_status = wait_for_sync_completion(
                client, sync_name, timeout_minutes
            )
            return sync_status
        else:
            return sync_name

    except Exception as e:
        logger.error(f"Feature View 동기화 실패: {e}")
        raise


def wait_for_sync_completion(
    client: FeatureOnlineStoreAdminServiceClient,
    sync_name: str,
    timeout_minutes: int = 30
) -> str:
    """동기화 완료 대기"""
    start_time = time.time()
    timeout_seconds = timeout_minutes * 60

    while True:
        elapsed = time.time() - start_time
        if elapsed > timeout_seconds:
            logger.warning(f"동기화 타임아웃 ({timeout_minutes}분)")
            return "TIMEOUT"

        try:
            request = GetFeatureViewSyncRequest(name=sync_name)
            sync = client.get_feature_view_sync(request=request)

            # 동기화 상태 확인
            if sync.run_time.end_time.seconds > 0:
                # 동기화 완료
                row_synced = sync.sync_summary.row_synced
                total_slot = sync.sync_summary.total_slot

                logger.info(f"동기화 완료!")
                logger.info(f"  - 동기화된 행: {row_synced:,}")
                logger.info(f"  - 총 슬롯: {total_slot}")

                return "COMPLETED"

            # 진행 중
            minutes_elapsed = int(elapsed / 60)
            logger.info(f"  동기화 진행 중... ({minutes_elapsed}분 경과)")

        except Exception as e:
            logger.warning(f"상태 조회 중 오류: {e}")

        time.sleep(30)  # 30초 간격으로 확인


def get_sync_status(
    project_id: str,
    region: str,
    online_store_name: str,
    feature_view_name: str
) -> dict:
    """최근 동기화 상태 조회"""
    client = get_online_store_client(region)

    feature_view_path = (
        f"projects/{project_id}/locations/{region}/"
        f"featureOnlineStores/{online_store_name}/featureViews/{feature_view_name}"
    )

    try:
        syncs = client.list_feature_view_syncs(parent=feature_view_path)
        sync_list = list(syncs)

        if not sync_list:
            return {"status": "NO_SYNC", "message": "동기화 기록 없음"}

        # 가장 최근 동기화
        latest_sync = sync_list[0]

        return {
            "status": "COMPLETED" if latest_sync.run_time.end_time.seconds > 0 else "IN_PROGRESS",
            "name": latest_sync.name,
            "row_synced": latest_sync.sync_summary.row_synced if latest_sync.run_time.end_time.seconds > 0 else 0,
            "start_time": latest_sync.run_time.start_time,
            "end_time": latest_sync.run_time.end_time if latest_sync.run_time.end_time.seconds > 0 else None,
        }

    except Exception as e:
        return {"status": "ERROR", "message": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Vertex AI Feature View 동기화")
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="동기화 완료를 기다리지 않음"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="동기화 완료 대기 타임아웃 (분, 기본값: 30)"
    )
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="동기화 트리거 없이 현재 상태만 확인"
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

    if args.status_only:
        # 상태만 확인
        status = get_sync_status(
            project_id=project_id,
            region=region,
            online_store_name=online_store_name,
            feature_view_name=feature_view_name
        )
        print(f"\n동기화 상태: {status}")
    else:
        # 동기화 트리거
        result = trigger_sync(
            project_id=project_id,
            region=region,
            online_store_name=online_store_name,
            feature_view_name=feature_view_name,
            wait_for_completion=not args.no_wait,
            timeout_minutes=args.timeout
        )
        print(f"\n동기화 결과: {result}")


if __name__ == "__main__":
    main()
