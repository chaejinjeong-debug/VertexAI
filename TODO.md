# TODO.md - Vertex AI Feature Store Demo 구현 작업 목록

## Phase 1: 프로젝트 초기 설정

- [x] 디렉토리 구조 생성 (`configs/`, `src/data/`, `src/featurestore/`, `src/training/`, `src/serving/`, `scripts/`)
- [x] `configs/env.yaml` 작성 (GCP 프로젝트, 리전, BigQuery 데이터셋 설정)
- [x] `configs/featurestore.yaml` 작성 (Feature Store 리소스 설정)
- [x] `configs/training.yaml` 작성 (학습 파라미터 설정)
- [x] `requirements.txt` 작성

## Phase 2: BigQuery 데이터 가공 (FR-1)

- [x] `src/data/01_prepare_bq.sql` - theLook 원본 테이블 준비/뷰 생성
- [x] `src/data/02_build_features.sql` - features_customer 스냅샷 생성
- [x] `src/data/03_build_labels.sql` - labels_customer 생성
- [x] `src/data/04_build_train.sql` - point-in-time join으로 train_dataset 생성
- [x] `src/data/run_sql.py` - SQL 실행 유틸리티

## Phase 3: Feature Store 리소스 생성 (FR-2)

- [ ] `src/featurestore/create_online_store.py` - Online Store 생성
- [ ] `src/featurestore/create_feature_group.py` - Feature Group 생성
- [ ] `src/featurestore/create_feature_view.py` - Feature View 생성
- [ ] `src/featurestore/sync_feature_view.py` - Feature View 동기화 트리거
- [ ] `src/featurestore/fetch_features.py` - 온라인 피처 조회

## Phase 4: 모델 학습 및 등록 (FR-3)

- [ ] `src/training/train.py` - 학습 스크립트 (scikit-learn)
- [ ] `src/training/upload_model.py` - Vertex AI Model Registry 업로드
- [ ] `src/training/deploy_endpoint.py` - Endpoint 생성 및 배포

## Phase 5: 온라인 추론 (FR-4)

- [ ] `src/serving/online_predict.py` - Feature fetch + Predict 통합

## Phase 6: 자동화 및 문서화 (FR-5)

- [ ] `scripts/demo.sh` - End-to-End 데모 실행 스크립트
- [ ] `README.md` 업데이트 - 설치, 실행, 권한 가이드

---

## 검증 방법

1. `scripts/demo.sh` 실행으로 전체 파이프라인 동작 확인
2. `python -m src.serving.online_predict --customer_id=XXXX` 로 예측 결과 확인

## 완료 기준 (Definition of Done)

- [ ] 새 프로젝트/새 계정에서 README대로 실행 시 재현 가능
- [ ] Feature fetch + predict가 최소 10건 고객에 대해 정상 동작
- [ ] 실행 실패 시 "어떤 단계에서 무엇이 필요한지" 로그로 출력
