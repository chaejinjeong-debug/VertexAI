# TODO.md - Vertex AI Feature Store Demo 구현 작업 목록

## Phase 1: 프로젝트 초기 설정

- [x] 디렉토리 구조 생성 (`configs/`, `src/data/`, `src/featurestore/`, `src/training/`, `src/serving/`, `scripts/`)
- [x] `configs/env.yaml` 작성 (GCP 프로젝트, 리전, BigQuery 데이터셋 설정)
- [x] `configs/featurestore.yaml` 작성 (Feature Store 리소스 설정)
- [x] `configs/training.yaml` 작성 (학습 파라미터 설정)
- [x] `requirements.txt` 작성

## Phase 2: BigQuery 데이터 가공 (FR-1)

- [x] `src/data/01_prepare_bq.sql` - theLook 원본 테이블 준비/뷰 생성
- [x] `src/data/02_build_features.sql` - features_customer 스냅샷 생성 (30일/90일 윈도우)
- [x] `src/data/03_build_labels_churn.sql` - labels_customer_churn 생성 (60일 이탈 라벨, orders 범위 기반 date spine)
- [x] `src/data/04_build_train.sql` - point-in-time join으로 train_dataset 생성
- [x] `src/data/05_sanity_checks.sql` - 라벨 분포/NULL rate/기간 범위 검증
- [x] `src/data/run_sql.py` - SQL 실행 유틸리티

## Phase 3: Feature Store 리소스 생성 (FR-2)

- [x] `src/featurestore/create_online_store.py` - Online Store 생성
- [x] `src/featurestore/create_feature_group.py` - Feature Group 생성
- [x] `src/featurestore/create_feature_view.py` - Feature View 생성
- [x] `src/featurestore/sync_feature_view.py` - Feature View 동기화 트리거
- [x] `src/featurestore/fetch_features.py` - 온라인 피처 조회

## Phase 4: 모델 학습 및 등록 (FR-3)

- [x] `src/training/train.py` - 학습 스크립트 (scikit-learn)
- [x] `src/training/upload_model.py` - Vertex AI Model Registry 업로드
- [x] `src/training/deploy_endpoint.py` - Endpoint 생성 및 배포

## Phase 5: 온라인 추론 (FR-4)

- [x] `src/serving/online_predict.py` - Feature fetch + Predict 통합

## Phase 6: 자동화 및 문서화 (FR-5)

- [x] `scripts/demo.sh` - End-to-End 데모 실행 스크립트
- [x] `README.md` 업데이트 - 설치, 실행, 권한 가이드

## Phase 7: Component Containerization (FR-6)

- [x] `src/components/_template/` - 컴포넌트 템플릿 생성
  - [x] Dockerfile
  - [x] src/main.py
- [x] `src/components/data_load/` - data_load 컴포넌트
  - [x] Dockerfile
  - [x] src/main.py (BQ → parquet split)
  - [x] 로컬 실행 테스트 (docker run --help 확인)
- [x] `src/components/train/` - train 컴포넌트
  - [x] Dockerfile
  - [x] src/main.py (model training)
  - [x] 로컬 실행 테스트 (docker run --help 확인)
- [x] `src/components/eval/` - eval 컴포넌트
  - [x] Dockerfile
  - [x] src/main.py (metrics 계산)
  - [x] 로컬 실행 테스트 (docker run --help 확인)

## Phase 8: Pipeline Wiring

- [ ] `scripts/build_push.sh` - 단일 컴포넌트 build/push 스크립트
- [ ] `src/pipelines/pipeline.py` - 파이프라인 정의
- [ ] `src/pipelines/compile.py` - JSON 컴파일
- [ ] `src/pipelines/run.py` - Vertex AI 제출
- [ ] Vertex AI Pipelines 실행 성공

## Phase 9: DevEx & Guardrails

- [ ] `scripts/build_push_all.sh` - 전체 컴포넌트 빌드 스크립트
- [ ] `scripts/smoke_test.sh` - 로컬 검증 스크립트
- [ ] CI 설정 - 변경분만 build/push

## Phase 10 (옵션): Advanced

- [ ] CustomJob 옵션화
- [ ] Image digest pinning

---

## 검증 방법

1. `scripts/demo.sh` 실행으로 전체 파이프라인 동작 확인
2. `python -m src.serving.online_predict --customer_id=XXXX` 로 예측 결과 확인
3. Sanity check 결과 확인:
   - `uv run src/data/run_sql.py --sql-file=05_sanity_checks.sql`
   - Churn rate이 20~80% 범위인지 확인

## 완료 기준 (Definition of Done)

- [x] 새 프로젝트/새 계정에서 README대로 실행 시 재현 가능
- [x] Feature fetch + predict가 최소 10건 고객에 대해 정상 동작
- [x] 라벨 positive rate가 0% 또는 100%가 아니며, sanity check 통과
- [x] 실행 실패 시 "어떤 단계에서 무엇이 필요한지" 로그로 출력
