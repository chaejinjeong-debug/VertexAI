# Vertex AI Feature Store Demo

theLook eCommerce 데이터를 활용한 Vertex AI Feature Store End-to-End 데모

## 개요

고객(customer_id)의 **60일 내 이탈(Churn) 확률**을 예측하는 ML 파이프라인 데모입니다.

- **오프라인(학습)**: 라벨 시점 기준 과거 행동/구매를 집계한 피처로 학습
- **온라인(추론)**: Feature Store에서 최신 피처를 조회하여 실시간 예측

### 아키텍처

```
BigQuery (theLook) → BigQuery (가공) → Feature Store → Online Serving
                           ↓
                    Model Training → Model Registry → Endpoint
                                                          ↓
                                      Feature Fetch → Predict → Response
```

## 사전 요구사항

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (패키지 관리)
- GCP 프로젝트 (BigQuery, Vertex AI 활성화)
- 인증: `gcloud auth application-default login`

### 필요 권한

| 서비스 | 권한 |
|--------|------|
| BigQuery | `bigquery.datasets.create`, `bigquery.tables.create`, `bigquery.jobs.create` |
| Feature Store | `aiplatform.featureOnlineStores.create`, `aiplatform.featureGroups.create` |
| Vertex AI | `aiplatform.models.upload`, `aiplatform.endpoints.create`, `aiplatform.endpoints.deploy` |
| Cloud Storage | `storage.buckets.create`, `storage.objects.create` |

## 설치

```bash
# uv 설치 (아직 없다면)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 의존성 설치
uv sync
```

## 설정

`configs/env.yaml`에서 GCP 프로젝트 ID 설정:

```yaml
gcp:
  project_id: "your-gcp-project-id"  # ← 실제 프로젝트 ID로 변경
  region: "asia-northeast3"           # Vertex AI 리전
```

---

## Quick Start (End-to-End 데모)

한 번의 명령으로 전체 파이프라인 실행:

```bash
./scripts/demo.sh
```

### 데모 옵션

```bash
# 특정 단계 스킵
./scripts/demo.sh --skip-data           # BigQuery 가공 스킵
./scripts/demo.sh --skip-featurestore   # Feature Store 생성 스킵
./scripts/demo.sh --skip-training       # 모델 학습 스킵
./scripts/demo.sh --skip-deploy         # 모델 배포 스킵

# 특정 고객 예측
./scripts/demo.sh --customer-ids "1234,5678,9012"

# 환경 변수 사용
SKIP_DATA=true SKIP_FEATURESTORE=true ./scripts/demo.sh
```

---

## 수동 실행 (Step-by-Step)

### Phase 1: BigQuery 데이터 가공

```bash
uv run python -m src.data.run_sql --all
```

생성 테이블:
- `features_customer`: 고객 피처 스냅샷 (30일/90일 윈도우)
- `labels_customer_churn`: 이탈 라벨 (60일 기준)
- `train_dataset`: Point-in-time join된 학습 데이터

### Phase 2: Feature Store 리소스 생성

```bash
# Online Store 생성
uv run python -m src.featurestore.create_online_store

# Feature Group 생성
uv run python -m src.featurestore.create_feature_group

# Feature View 생성
uv run python -m src.featurestore.create_feature_view

# 최초 동기화
uv run python -m src.featurestore.sync_feature_view
```

### Phase 3: 모델 학습

```bash
uv run python -m src.training.train
```

출력:
- `artifacts/model/model.joblib`: 학습된 모델
- `artifacts/model/metrics.yaml`: 평가 메트릭 (ROC-AUC, PR-AUC 등)

### Phase 4: 모델 등록 및 배포

```bash
# Model Registry 업로드
uv run python -m src.training.upload_model

# Endpoint 배포
uv run python -m src.training.deploy_endpoint
```

### Phase 5: 온라인 예측

```bash
# 단일 고객 예측
uv run python -m src.serving.online_predict --customer-id 12345

# 피처 포함 출력
uv run python -m src.serving.online_predict --customer-id 12345 --show-features

# 배치 예측
uv run python -m src.serving.online_predict --customer-ids 12345 67890 11111
```

---

## 피처 목록

| 피처 | 설명 |
|------|------|
| `orders_30d` | 최근 30일 주문 수 |
| `orders_90d` | 최근 90일 주문 수 |
| `revenue_30d` | 최근 30일 매출 |
| `revenue_90d` | 최근 90일 매출 |
| `avg_order_value_90d` | 최근 90일 평균 주문 금액 |
| `distinct_products_90d` | 최근 90일 구매 상품 수 (고유) |
| `distinct_categories_90d` | 최근 90일 구매 카테고리 수 (고유) |
| `days_since_last_order` | 마지막 주문 후 경과일 |

---

## 프로젝트 구조

```
.
├── configs/
│   ├── env.yaml              # GCP 환경 설정
│   ├── featurestore.yaml     # Feature Store 리소스 설정
│   └── training.yaml         # 학습 파라미터 설정
├── src/
│   ├── data/                 # BigQuery SQL 및 실행 유틸리티
│   │   ├── 01_prepare_bq.sql
│   │   ├── 02_build_features.sql
│   │   ├── 03_build_labels_churn.sql
│   │   ├── 04_build_train.sql
│   │   ├── 05_sanity_checks.sql
│   │   └── run_sql.py
│   ├── featurestore/         # Feature Store 관리
│   │   ├── create_online_store.py
│   │   ├── create_feature_group.py
│   │   ├── create_feature_view.py
│   │   ├── sync_feature_view.py
│   │   └── fetch_features.py
│   ├── training/             # 모델 학습/배포
│   │   ├── train.py
│   │   ├── upload_model.py
│   │   └── deploy_endpoint.py
│   └── serving/              # 온라인 추론
│       └── online_predict.py
├── scripts/
│   └── demo.sh               # End-to-End 데모 스크립트
├── artifacts/                # 모델 아티팩트 (gitignore)
└── TODO.md                   # 구현 진행 상황
```

---

## 검증

### Sanity Check

```bash
uv run python -m src.data.run_sql --sql-file=05_sanity_checks.sql
```

확인 사항:
- Churn rate이 20~80% 범위 (극단적 편향 없음)
- NULL 피처 비율
- 데이터 기간 범위

### Feature Store 동기화 상태

```bash
uv run python -m src.featurestore.sync_feature_view --status
```

---

## 문제 해결

### 권한 오류

```bash
# 현재 인증 확인
gcloud auth application-default print-access-token

# 재인증
gcloud auth application-default login
```

### Feature Store 리소스가 이미 존재

기존 리소스가 있으면 자동으로 재사용됩니다. 강제 재생성이 필요하면:

```bash
uv run python -m src.featurestore.create_online_store --force
```

### 모델 배포 실패

1. Model Registry에 모델이 업로드되었는지 확인
2. Endpoint 할당량 확인 (리전별 제한)
3. 머신 타입 가용성 확인

---

## 라이선스

MIT
