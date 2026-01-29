#!/bin/bash
# Vertex AI Feature Store Demo - End-to-End 실행 스크립트
# theLook eCommerce 데이터로 Customer Churn Prediction 데모

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 스크립트 디렉토리
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Vertex AI Feature Store Demo${NC}"
echo -e "${BLUE}  Customer Churn Prediction${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 단계 실행 함수
run_step() {
    local step_num=$1
    local step_name=$2
    local command=$3

    echo -e "${YELLOW}[$step_num] $step_name${NC}"
    echo -e "${BLUE}실행: $command${NC}"
    echo ""

    if eval "$command"; then
        echo -e "${GREEN}✓ $step_name 완료${NC}"
    else
        echo -e "${RED}✗ $step_name 실패${NC}"
        exit 1
    fi
    echo ""
    echo "-------------------------------------------"
    echo ""
}

# 환경 확인
echo -e "${YELLOW}[0] 환경 확인${NC}"
echo ""

# uv 확인
if ! command -v uv &> /dev/null; then
    echo -e "${RED}Error: uv가 설치되어 있지 않습니다.${NC}"
    echo "설치: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# gcloud 확인
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}Error: gcloud CLI가 설치되어 있지 않습니다.${NC}"
    exit 1
fi

# 인증 확인
if ! gcloud auth application-default print-access-token &> /dev/null 2>&1; then
    echo -e "${YELLOW}Google Cloud 인증이 필요합니다.${NC}"
    gcloud auth application-default login
fi

echo -e "${GREEN}✓ 환경 확인 완료${NC}"
echo ""
echo "-------------------------------------------"
echo ""

# 인자 파싱
SKIP_DATA=${SKIP_DATA:-false}
SKIP_FEATURESTORE=${SKIP_FEATURESTORE:-false}
SKIP_TRAINING=${SKIP_TRAINING:-false}
SKIP_DEPLOY=${SKIP_DEPLOY:-false}
CUSTOMER_IDS=${CUSTOMER_IDS:-""}

# 사용법 출력
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --skip-data           BigQuery 데이터 가공 단계 스킵"
    echo "  --skip-featurestore   Feature Store 리소스 생성 단계 스킵"
    echo "  --skip-training       모델 학습 단계 스킵"
    echo "  --skip-deploy         모델 배포 단계 스킵"
    echo "  --customer-ids IDS    예측할 고객 ID (쉼표 구분)"
    echo ""
    echo "Environment variables:"
    echo "  SKIP_DATA=true          BigQuery 단계 스킵"
    echo "  SKIP_FEATURESTORE=true  Feature Store 단계 스킵"
    echo "  SKIP_TRAINING=true      학습 단계 스킵"
    echo "  SKIP_DEPLOY=true        배포 단계 스킵"
    echo "  CUSTOMER_IDS='1,2,3'    예측할 고객 ID"
    echo ""
    exit 0
fi

# 인자 처리
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-data)
            SKIP_DATA=true
            shift
            ;;
        --skip-featurestore)
            SKIP_FEATURESTORE=true
            shift
            ;;
        --skip-training)
            SKIP_TRAINING=true
            shift
            ;;
        --skip-deploy)
            SKIP_DEPLOY=true
            shift
            ;;
        --customer-ids)
            CUSTOMER_IDS="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# ============================================
# Phase 1: BigQuery 데이터 가공
# ============================================
if [ "$SKIP_DATA" != "true" ]; then
    run_step "1" "BigQuery 데이터 가공 (features, labels, train_dataset)" \
        "uv run python -m src.data.run_sql --all"
else
    echo -e "${YELLOW}[1] BigQuery 데이터 가공 - 스킵${NC}"
    echo ""
fi

# ============================================
# Phase 2: Feature Store 리소스 생성
# ============================================
if [ "$SKIP_FEATURESTORE" != "true" ]; then
    run_step "2-1" "Online Store 생성" \
        "uv run python -m src.featurestore.create_online_store"

    run_step "2-2" "Feature Group 생성" \
        "uv run python -m src.featurestore.create_feature_group"

    run_step "2-3" "Feature View 생성" \
        "uv run python -m src.featurestore.create_feature_view"

    run_step "2-4" "Feature View 동기화" \
        "uv run python -m src.featurestore.sync_feature_view"
else
    echo -e "${YELLOW}[2] Feature Store 리소스 생성 - 스킵${NC}"
    echo ""
fi

# ============================================
# Phase 3: 모델 학습
# ============================================
if [ "$SKIP_TRAINING" != "true" ]; then
    run_step "3" "모델 학습 (scikit-learn RandomForest)" \
        "uv run python -m src.training.train"
else
    echo -e "${YELLOW}[3] 모델 학습 - 스킵${NC}"
    echo ""
fi

# ============================================
# Phase 4: 모델 등록 및 배포
# ============================================
if [ "$SKIP_DEPLOY" != "true" ]; then
    run_step "4-1" "Vertex AI Model Registry 업로드" \
        "uv run python -m src.training.upload_model"

    run_step "4-2" "Vertex AI Endpoint 배포" \
        "uv run python -m src.training.deploy_endpoint"
else
    echo -e "${YELLOW}[4] 모델 배포 - 스킵${NC}"
    echo ""
fi

# ============================================
# Phase 5: 온라인 예측 테스트
# ============================================
echo -e "${YELLOW}[5] 온라인 예측 테스트${NC}"
echo ""

# 테스트용 고객 ID 조회 (지정되지 않은 경우)
if [ -z "$CUSTOMER_IDS" ]; then
    echo "테스트용 고객 ID를 BigQuery에서 조회 중..."

    # BigQuery에서 최근 활성 고객 10명 조회
    CUSTOMER_IDS=$(gcloud sql query --project="$(grep 'project_id' configs/env.yaml | awk -F'"' '{print $2}')" \
        "SELECT DISTINCT customer_id FROM \`$(grep 'project_id' configs/env.yaml | awk -F'"' '{print $2}').$(grep 'target_dataset' configs/env.yaml | awk -F'"' '{print $2}').train_dataset\` LIMIT 10" \
        --format="csv(customer_id)" 2>/dev/null | tail -n +2 | tr '\n' ',' | sed 's/,$//' || true)

    # 조회 실패 시 기본값
    if [ -z "$CUSTOMER_IDS" ]; then
        echo "BigQuery 조회 실패. 기본 고객 ID 사용 중..."
        CUSTOMER_IDS="1,2,3,4,5"
    fi
fi

echo "테스트 고객 ID: $CUSTOMER_IDS"
echo ""

# 쉼표를 공백으로 변환하여 배열로
IFS=',' read -ra CUSTOMER_ARRAY <<< "$CUSTOMER_IDS"

# 예측 실행
echo -e "${BLUE}예측 실행 중...${NC}"
echo ""

success_count=0
total_count=0

for customer_id in "${CUSTOMER_ARRAY[@]}"; do
    customer_id=$(echo "$customer_id" | tr -d ' ')
    if [ -n "$customer_id" ]; then
        total_count=$((total_count + 1))
        echo -n "  Customer $customer_id: "

        result=$(uv run python -m src.serving.online_predict --customer-id "$customer_id" 2>&1 || true)

        if echo "$result" | grep -q "Churn Probability"; then
            prob=$(echo "$result" | grep "Churn Probability" | awk -F': ' '{print $2}')
            echo -e "${GREEN}$prob${NC}"
            success_count=$((success_count + 1))
        else
            echo -e "${RED}실패${NC}"
        fi
    fi
done

echo ""
echo -e "${GREEN}예측 성공: $success_count / $total_count${NC}"
echo ""

# ============================================
# 완료
# ============================================
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}  데모 완료!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "다음 명령어로 개별 고객 예측을 수행할 수 있습니다:"
echo ""
echo "  uv run python -m src.serving.online_predict --customer-id <CUSTOMER_ID>"
echo ""
echo "피처까지 출력하려면:"
echo ""
echo "  uv run python -m src.serving.online_predict --customer-id <CUSTOMER_ID> --show-features"
echo ""
