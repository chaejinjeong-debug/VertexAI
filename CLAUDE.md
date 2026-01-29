# PRD (Updated): Vertex AI Feature Store(BigQuery 기반) + theLook eCommerce로 End-to-End 데모 (Customer Churn)

1. 배경과 문제 정의

팀/조직 내에서 “Feature Store가 왜 필요하고, 실제로 학습/서빙 일관성(offline/online consistency)을 어떻게 보장하는지”를 짧은 시간에 설득력 있게 보여줄 데모가 필요하다.
단순 모델 학습 예제(iris 등)는 엔티티/시점/온라인 서빙/point-in-time join의 핵심 가치를 드러내기 어렵다.

또한 theLook eCommerce처럼 과거 기간 데이터가 있는 경우, 현재 날짜 기반 date spine을 쓰면 라벨이 전부 0/1로 치우치는 문제가 발생할 수 있으므로, 라벨/피처 생성 시 주문 데이터 범위 기반 spine을 사용해야 한다.

2. 목표

theLook eCommerce 데이터를 BigQuery에서 가공하여

Feature Group 등록

Online Store + Feature View 생성

Feature View Sync(스케줄 기반)

학습 데이터(point-in-time join) 생성

Vertex AI 학습 → Model Registry 업로드

Endpoint 배포

온라인 요청에서 Feature fetch + Predict

까지 한 번에 재현 가능한 데모를 구축한다.

3. Non-goals

초고성능 모델/최고 성능 달성(모델은 scikit-learn 수준으로 단순화)

실시간 스트리밍 피처 파이프라인(Dataflow/Kafka 등) 구축

고급 피처 엔지니어링 자동화(Feast 수준의 복잡한 변환 DAG)

(범위 밖) 모델 모니터링/드리프트 감지 운영 고도화

4. 사용자(Stakeholders)

ML 엔지니어/DS: 피처 재사용, 학습-서빙 일관성 확인

MLOps/플랫폼 엔지니어: 리소스 구성, 권한/배포/운영 흐름 확인

PM/리더십: “Feature Store 도입 가치”를 빠르게 이해

5. 성공 지표 / 완료 정의(Definition of Done)

make demo 또는 scripts/demo.sh 한 번으로 아래가 실행됨:

BigQuery에 features_customer, labels_customer_churn, train_dataset 생성

Feature Group / Online Store / Feature View 생성

Feature View 최초 sync 완료(온라인 서빙 가능 상태)

학습 수행 및 모델 업로드, 엔드포인트 배포 완료

customer_id 1건 입력 시 (Feature fetch → 예측값 반환) 동작

실행 로그에 각 단계 리소스 ID/이름 + 핵심 metric(AUC/PR-AUC/Accuracy 등) 출력

(추가) 라벨 분포(positive rate)가 로그에 출력되어 라벨이 한쪽으로 쏠리지 않음을 확인

6. 데모 시나리오(스토리)

“어떤 고객(customer_id)이 기준 시점 이후 N일 동안 구매가 없을(=이탈할) 확률을 예측한다.”

오프라인(학습): label_timestamp(스냅샷 날짜) 기준으로 과거 행동/구매를 집계한 피처를 생성해 학습

온라인(추론): 요청 시점 customer_id로 Feature Store에서 최신 피처를 조회하고 모델에 입력

Churn 라벨 정의(데모 표준)

label_churn_Nd = 1
if (label_timestamp, label_timestamp + N] 구간에 주문이 없음

else 0

N은 데모 안정성을 위해 60일(기본) 또는 90일(옵션) 을 사용한다.
(14일은 “재구매”에 가깝고 이탈로는 너무 단기라 라벨 편향 위험이 큼)

7. 데이터셋: theLook eCommerce

소스: BigQuery Public Dataset (theLook eCommerce)

엔티티: customer_id

이벤트/주문 시점 컬럼을 활용하여 시점 기반 학습 데이터(point-in-time) 생성 가능

8. 데이터 모델(가공 결과)
8.1 features_customer (피처 스냅샷)

키: customer_id

시점: feature_timestamp (예: 하루 1회 스냅샷)

피처(간단 버전, churn에 적합하게 조정)

orders_30d, orders_90d

revenue_30d, revenue_90d, avg_order_value_90d

days_since_last_order

distinct_products_90d, distinct_categories_90d

(옵션) return_rate_90d (order_items returned_at 활용 가능 시)

목적: Feature Store Online/Offline에서 동일 피처 정의를 재사용

8.2 labels_customer_churn (라벨)

키: customer_id

시점: label_timestamp

라벨: label_churn_60d (기본) / label_churn_90d (옵션)

중요: 라벨 생성 시 label_timestamp spine은 CURRENT_DATE 기반이 아니라 orders 데이터 범위 기반으로 생성한다.

spine 상한: MAX(order_date) - N

8.3 train_dataset (학습 데이터)

labels_customer_churn의 각 row에 대해

features_customer에서 feature_timestamp <= label_timestamp 인 것 중 가장 최신 스냅샷을 조인

누수 방지(point-in-time join) 알려주기 좋은 구조

9. 시스템 구성(아키텍처 개요)

BigQuery (raw/public) → BigQuery (가공 테이블/뷰)

Vertex AI Feature Store(Latest)

Feature Group: BigQuery features_customer 등록

Online Store: online serving 저장소

Feature View: Feature Group 기반 서빙 뷰 + Scheduled sync

Vertex AI Training

BigQuery train_dataset로 학습

Vertex AI Model Registry + Endpoint

Online Serving Client

customer_id → Feature View fetch → Endpoint predict

10. 기능 요구사항(Functional Requirements)
FR-1: BigQuery 가공 파이프라인

01_prepare_bq.sql: raw 테이블에서 필요한 컬럼만 정리(또는 뷰 생성)

02_build_features.sql: features_customer 스냅샷 생성(날짜 단위)

03_build_labels_churn.sql: labels_customer_churn 생성

orders 범위 기반 date spine 사용

N=60(기본), N=90(옵션)

04_build_train.sql: point-in-time join으로 train_dataset 생성

05_sanity_checks.sql(추가): 라벨/피처/조인 결과 검증

라벨 positive rate, null rate, 기간 범위 체크 등

FR-2: Feature Store 리소스 생성

Online Store 생성

Feature Group 생성 (source = BigQuery, entity_id_columns = customer_id)

Feature View 생성 (source = Feature Group, online_store 지정)

Sync 설정: Scheduled sync(서울 리전에서 continuous sync 제약 가능성 고려)

최초 1회 sync 트리거(데모 시 반드시 포함)

(추가) sync 완료 여부 확인 로직(상태 polling) 포함

FR-3: 모델 학습 및 등록

학습 스크립트가 train_dataset를 로드해 학습

metric 출력(예: ROC-AUC + PR-AUC 권장)

모델 artifact 저장 및 Vertex AI Model 업로드

FR-4: 배포 및 온라인 추론

Endpoint 생성/배포(또는 기존 endpoint 재사용)

Online request 흐름:

입력: customer_id

Feature View에서 피처 조회

모델 입력 벡터 구성

Endpoint predict 호출

출력: churn_probability

FR-5: 재현성/자동화

단일 커맨드 실행으로 end-to-end 수행(최소: 단계별 스크립트 + 상위 runner)

각 단계는 “이미 존재하면 재사용/스킵” 옵션 제공(데모 반복 실행용)

(추가) 이미지/리소스 네이밍 규칙 및 환경(dev/prod) config 분리

FR-6: Pipeline 컴포넌트 구성 (컨테이너 기반)

Pipeline 컴포넌트는 3개로 구성:

1. **data_load**
   - 입력: BQ 테이블 경로, 라벨 컬럼, split 비율
   - 출력: train/valid/test parquet (artifact dir)
   - 시간 기준 split (time_col 기반) 지원

2. **train**
   - 입력: dataset artifact
   - 출력: model artifact (model.pkl, model_meta.json)
   - MVP: logistic regression / RandomForest

3. **eval**
   - 입력: model + dataset (valid/test)
   - 출력: metrics.json (ROC-AUC, PR-AUC, accuracy, positive_rate)

각 컴포넌트는 컨테이너 이미지로 Artifact Registry에 배포

### 컴포넌트 표준 인터페이스

#### 6.1 실행 규약
- 컨테이너 단독 실행 가능: `docker run IMAGE --help`
- 입력/출력은 경로 기반: `--input_*`, `--output_*`
- 출력은 반드시 파일로 남김 (다음 스텝 재사용)

#### 6.2 아티팩트 규칙
- dataset: `dataset_dir/{train,valid,test}.parquet`
- model: `model_dir/model.pkl`, `model_dir/model_meta.json`
- metrics: `metrics.json`

#### 6.3 이미지 버전 규칙
- 이미지 태그: `:<gitsha>` (기본)
- 릴리스 태그: `:<version>` (옵션)

### DevEx 요구사항

#### 템플릿 제공
- `src/components/_template/` 제공
  - Dockerfile
  - src/main.py
  - (옵션) component.yaml

#### 표준 스크립트
- `scripts/build_push.sh <component>`: 단일 컴포넌트 build/push
- `scripts/build_push_all.sh`: 전체 build/push
- `scripts/smoke_test.sh <component>`: 로컬 검증

#### 파이프라인 스크립트
- `src/pipelines/pipeline.py`: 파이프라인 정의
- `src/pipelines/compile.py`: JSON 컴파일
- `src/pipelines/run.py`: Vertex AI 제출

### 디렉토리 구조

```
src/
├── data/           # (유지) SQL 스크립트
├── featurestore/   # (유지) Feature Store
├── training/       # (유지) 학습
├── serving/        # (유지) 서빙
├── components/     # (추가) 파이프라인 컴포넌트
│   ├── _template/
│   │   ├── Dockerfile
│   │   └── src/main.py
│   ├── data_load/
│   ├── train/
│   └── eval/
└── pipelines/      # (추가) 파이프라인 정의
    ├── pipeline.py
    ├── compile.py
    └── run.py

scripts/
├── demo.sh                    # (유지)
├── build_serving_container.sh # (유지)
├── build_push.sh              # (추가) 컴포넌트 빌드
├── build_push_all.sh          # (추가) 전체 빌드
└── smoke_test.sh              # (추가) 로컬 검증
```

11. 비기능 요구사항(Non-functional Requirements)

데모 실행 시간: 전체 10~20분 내(데이터 범위/샘플링으로 조절)

비용 통제: BigQuery 쿼리 비용 최소화를 위해 기간 제한/샘플링 적용

권한 최소화: Feature View용 서비스 계정 분리 가능(옵션)

관측성: 단계별 리소스명/상태/동기화 완료 여부를 로그로 명확히 출력

(추가) 라벨 분포가 한쪽으로 쏠릴 경우 경고 출력(데모 실패 방지)

12. 권한/보안 요구사항

BigQuery:

가공 dataset에 대한 create table 권한

public dataset read 권한(기본)

Feature Store / Vertex AI:

Feature Store 리소스 생성 권한

Endpoint 배포 권한

(옵션) Feature View 전용 서비스 계정:

BigQuery Data Viewer (소스 조회)

Online Store 접근 권한(필요 시)

13. 리전/동기화 전략

Feature View sync는 Scheduled sync를 기본으로 한다.

데모에서는 “sync 한번 실행해서 온라인 서빙 가능 상태”를 반드시 보여준다.

BigQuery 데이터셋 위치와 Vertex 리전을 맞추거나(가능하면), 교차 리전으로 인한 제약/지연을 데모 범위에서 피한다.

(추가) 라벨 생성은 CURRENT_DATE() 의존을 피하고 orders 데이터 범위 기반으로 처리한다.

14. 리스크 및 대응

쿼리 비용/시간 과다: 기간 제한(최근 3~6개월), 샘플링, 스냅샷 granularity 조절

라벨 편향/전부 0/1: orders 범위 기반 spine + sanity check로 사전 차단

sync 지연/실패: 최초 sync를 별도 단계로 분리하고 상태 확인 로직 추가

권한 문제: 최소 권한 체크리스트 + 사전 검증 스크립트 제공

데이터 스키마 변경/테이블 경로 혼동: config로 테이블 URI를 단일 관리

15. 수용 기준(Acceptance Criteria)

새 프로젝트/새 계정에서도(필수 권한만 갖추면) README대로 실행 시 재현 가능

Feature fetch + predict가 최소 10건 고객에 대해 정상 동작

라벨 positive rate가 0% 또는 100%가 아니며(경고 기준 설정), sanity check를 통과

실행 실패 시(권한/리전/테이블 미존재) “어떤 단계에서 무엇이 필요한지”가 로그로 드러남

16. 마일스톤 (Phase)

### Phase 1-6: 기존 구현 (완료)
- Feature Store 기반 End-to-End 데모 구현 완료

### Phase 7: Component Containerization
- [ ] 컴포넌트 템플릿 생성 (`src/components/_template/`)
- [ ] data_load 컴포넌트 로컬 실행
- [ ] train 컴포넌트 로컬 실행
- [ ] eval 컴포넌트 로컬 실행

### Phase 8: Pipeline Wiring
- [ ] build_push.sh 스크립트 작성
- [ ] pipeline.py 작성
- [ ] compile.py / run.py 작성
- [ ] Vertex AI Pipelines 실행 성공

### Phase 9: DevEx & Guardrails
- [ ] build_push_all.sh 스크립트
- [ ] smoke_test.sh 스크립트
- [ ] CI에서 변경분만 build/push

### Phase 10 (옵션): Advanced
- [ ] CustomJob 옵션화
- [ ] Image digest pinning

# Rules
1. 코드 생성 시, Phase의 수행 내역들에 대해 TODO.md에서 Check Box를 갱신한다.
2. Phase 완료 시 github으로 commit push 한다.