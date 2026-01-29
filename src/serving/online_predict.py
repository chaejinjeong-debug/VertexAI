"""
온라인 추론 스크립트
Feature Store에서 피처를 조회하고 Endpoint에서 예측을 수행합니다.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import yaml
from google.cloud import aiplatform

# 피처 조회 모듈 임포트
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.featurestore.fetch_features import fetch_features, fetch_features_batch

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"


def load_configs() -> tuple[dict, dict, dict]:
    """환경 설정 및 관련 설정 로드"""
    with open(CONFIGS_DIR / "env.yaml", "r") as f:
        env_config = yaml.safe_load(f)
    with open(CONFIGS_DIR / "featurestore.yaml", "r") as f:
        fs_config = yaml.safe_load(f)
    with open(CONFIGS_DIR / "training.yaml", "r") as f:
        training_config = yaml.safe_load(f)
    return env_config, fs_config, training_config


def get_endpoint(
    project_id: str,
    region: str,
    endpoint_display_name: str
) -> Optional[aiplatform.Endpoint]:
    """Endpoint 조회"""
    aiplatform.init(project=project_id, location=region)

    endpoints = aiplatform.Endpoint.list(
        filter=f'display_name="{endpoint_display_name}"'
    )

    if endpoints:
        return endpoints[0]
    return None


def prepare_prediction_input(
    features: dict,
    feature_columns: list[str]
) -> list[list[float]]:
    """예측 입력 형식으로 변환"""
    # 피처 순서 맞추기
    instance = []
    for col in feature_columns:
        value = features.get(col, 0)
        # None 처리
        if value is None:
            value = 0
        instance.append(float(value))

    return [instance]


def predict(
    endpoint: aiplatform.Endpoint,
    instances: list[list[float]]
) -> dict:
    """Endpoint 예측 호출"""
    response = endpoint.predict(instances=instances)

    # 응답 파싱
    predictions = response.predictions
    if predictions:
        # sklearn RandomForest의 predict_proba 결과
        # [[no_churn_prob, churn_prob], ...]
        return {
            "predictions": predictions,
            "churn_probability": predictions[0][1] if len(predictions[0]) > 1 else predictions[0]
        }

    return {"predictions": [], "churn_probability": None}


def online_predict_single(
    customer_id: str,
    env_config: dict,
    fs_config: dict,
    training_config: dict
) -> dict:
    """단일 고객 예측"""
    project_id = env_config["gcp"]["project_id"]
    region = env_config["gcp"]["region"]

    online_store_name = fs_config["online_store"]["name"]
    feature_view_name = fs_config["feature_view"]["name"]
    feature_columns = training_config["training"]["feature_columns"]
    endpoint_display_name = training_config["endpoint"]["display_name"]

    # 1. Feature Store에서 피처 조회
    logger.info(f"피처 조회 중: customer_id={customer_id}")
    features = fetch_features(
        project_id=project_id,
        region=region,
        online_store_name=online_store_name,
        feature_view_name=feature_view_name,
        entity_id=customer_id,
        feature_names=feature_columns
    )

    if not features:
        return {
            "customer_id": customer_id,
            "status": "error",
            "error": "피처 조회 실패 또는 해당 고객 없음"
        }

    logger.info(f"피처 조회 완료: {len(features)} features")

    # 2. Endpoint 조회
    endpoint = get_endpoint(project_id, region, endpoint_display_name)
    if not endpoint:
        return {
            "customer_id": customer_id,
            "status": "error",
            "error": f"Endpoint를 찾을 수 없음: {endpoint_display_name}"
        }

    # 3. 예측 입력 준비
    instances = prepare_prediction_input(features, feature_columns)

    # 4. 예측 수행
    logger.info("예측 수행 중...")
    prediction_result = predict(endpoint, instances)

    return {
        "customer_id": customer_id,
        "features": features,
        "churn_probability": prediction_result["churn_probability"],
        "status": "success"
    }


def online_predict_batch(
    customer_ids: list[str],
    env_config: dict,
    fs_config: dict,
    training_config: dict
) -> list[dict]:
    """배치 고객 예측"""
    project_id = env_config["gcp"]["project_id"]
    region = env_config["gcp"]["region"]

    online_store_name = fs_config["online_store"]["name"]
    feature_view_name = fs_config["feature_view"]["name"]
    feature_columns = training_config["training"]["feature_columns"]
    endpoint_display_name = training_config["endpoint"]["display_name"]

    # Endpoint 조회
    endpoint = get_endpoint(project_id, region, endpoint_display_name)
    if not endpoint:
        logger.error(f"Endpoint를 찾을 수 없음: {endpoint_display_name}")
        return []

    results = []

    for customer_id in customer_ids:
        try:
            # 피처 조회
            features = fetch_features(
                project_id=project_id,
                region=region,
                online_store_name=online_store_name,
                feature_view_name=feature_view_name,
                entity_id=customer_id,
                feature_names=feature_columns
            )

            if not features:
                results.append({
                    "customer_id": customer_id,
                    "status": "error",
                    "error": "피처 조회 실패"
                })
                continue

            # 예측
            instances = prepare_prediction_input(features, feature_columns)
            prediction_result = predict(endpoint, instances)

            results.append({
                "customer_id": customer_id,
                "churn_probability": prediction_result["churn_probability"],
                "status": "success"
            })

        except Exception as e:
            results.append({
                "customer_id": customer_id,
                "status": "error",
                "error": str(e)
            })

    return results


def main():
    parser = argparse.ArgumentParser(description="Feature Store + Endpoint 온라인 예측")
    parser.add_argument(
        "--customer-id",
        type=str,
        help="예측할 고객 ID (단일)"
    )
    parser.add_argument(
        "--customer-ids",
        type=str,
        nargs="+",
        help="예측할 고객 ID 목록 (배치)"
    )
    parser.add_argument(
        "--show-features",
        action="store_true",
        help="피처 값도 출력"
    )
    args = parser.parse_args()

    if not args.customer_id and not args.customer_ids:
        parser.error("--customer-id 또는 --customer-ids 중 하나는 필수입니다.")

    # 설정 로드
    env_config, fs_config, training_config = load_configs()

    # Vertex AI 초기화
    project_id = env_config["gcp"]["project_id"]
    region = env_config["gcp"]["region"]
    aiplatform.init(project=project_id, location=region)

    if args.customer_id:
        # 단일 예측
        result = online_predict_single(
            customer_id=args.customer_id,
            env_config=env_config,
            fs_config=fs_config,
            training_config=training_config
        )

        print(f"\n=== 예측 결과 ===")
        print(f"Customer ID: {result['customer_id']}")
        print(f"Status: {result['status']}")

        if result['status'] == 'success':
            churn_prob = result['churn_probability']
            print(f"Churn Probability: {churn_prob:.4f} ({churn_prob*100:.2f}%)")

            if args.show_features:
                print("\n피처:")
                for name, value in result.get('features', {}).items():
                    print(f"  {name}: {value}")
        else:
            print(f"Error: {result.get('error', 'Unknown')}")

    else:
        # 배치 예측
        results = online_predict_batch(
            customer_ids=args.customer_ids,
            env_config=env_config,
            fs_config=fs_config,
            training_config=training_config
        )

        print(f"\n=== 배치 예측 결과 ({len(results)}건) ===")
        success_count = 0

        for result in results:
            status = result['status']
            customer_id = result['customer_id']

            if status == 'success':
                churn_prob = result['churn_probability']
                print(f"  {customer_id}: {churn_prob:.4f} ({churn_prob*100:.2f}%)")
                success_count += 1
            else:
                print(f"  {customer_id}: ERROR - {result.get('error', 'Unknown')}")

        print(f"\n성공: {success_count}/{len(results)}")


if __name__ == "__main__":
    main()
