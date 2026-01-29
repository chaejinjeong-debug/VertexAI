"""
SQL 실행 유틸리티
BigQuery에서 SQL 파일을 실행하고 결과를 반환합니다.
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml
from google.cloud import bigquery
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
SQL_DIR = PROJECT_ROOT / "src" / "data"


def load_config() -> dict:
    """환경 설정 로드"""
    config_path = CONFIGS_DIR / "env.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_dataset_if_not_exists(
    client: bigquery.Client,
    project_id: str,
    dataset_id: str,
    location: str
) -> None:
    """데이터셋이 없으면 생성"""
    dataset_ref = f"{project_id}.{dataset_id}"

    try:
        client.get_dataset(dataset_ref)
        logger.info(f"데이터셋 이미 존재: {dataset_ref}")
    except gcp_exceptions.NotFound:
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = location
        client.create_dataset(dataset)
        logger.info(f"데이터셋 생성 완료: {dataset_ref}")


def format_sql(sql: str, config: dict) -> str:
    """SQL 템플릿에 설정값 적용"""
    return sql.format(
        project_id=config["gcp"]["project_id"],
        target_dataset=config["bigquery"]["target_dataset"],
        source_dataset=config["bigquery"]["source_dataset"],
        lookback_months=config["data"]["lookback_months"],
        sample_rate=config["data"]["sample_rate"],
    )


def execute_sql_file(
    client: bigquery.Client,
    sql_file: Path,
    config: dict,
    dry_run: bool = False
) -> None:
    """SQL 파일 실행"""
    logger.info(f"SQL 파일 실행: {sql_file.name}")

    # SQL 파일 읽기
    with open(sql_file, "r") as f:
        sql_content = f.read()

    # 설정값 적용
    formatted_sql = format_sql(sql_content, config)

    if dry_run:
        logger.info("=== DRY RUN MODE ===")
        print(formatted_sql)
        return

    # SQL 문장 분리 (세미콜론 기준, 주석 제외)
    statements = []
    current_statement = []

    for line in formatted_sql.split("\n"):
        stripped = line.strip()

        # 주석만 있는 줄 스킵
        if stripped.startswith("--"):
            continue

        current_statement.append(line)

        # 세미콜론으로 끝나면 문장 완료
        if stripped.endswith(";"):
            statement = "\n".join(current_statement).strip()
            if statement and not statement.startswith("--"):
                statements.append(statement)
            current_statement = []

    # 마지막 문장 처리 (세미콜론 없는 경우)
    if current_statement:
        statement = "\n".join(current_statement).strip()
        if statement and not statement.startswith("--"):
            statements.append(statement)

    # 각 문장 실행
    for i, statement in enumerate(statements, 1):
        if not statement.strip():
            continue

        logger.info(f"  문장 {i}/{len(statements)} 실행 중...")

        try:
            query_job = client.query(statement)
            result = query_job.result()

            # 결과 정보 출력
            if query_job.statement_type in ("CREATE_TABLE", "CREATE_VIEW"):
                logger.info(f"    생성 완료: {query_job.destination}")
            elif query_job.total_bytes_processed:
                mb_processed = query_job.total_bytes_processed / (1024 * 1024)
                logger.info(f"    처리된 데이터: {mb_processed:.2f} MB")

        except Exception as e:
            logger.error(f"    실행 실패: {e}")
            raise


def get_table_info(
    client: bigquery.Client,
    project_id: str,
    dataset_id: str,
    table_id: str
) -> dict:
    """테이블 정보 조회"""
    table_ref = f"{project_id}.{dataset_id}.{table_id}"

    try:
        table = client.get_table(table_ref)
        return {
            "table_id": table_id,
            "num_rows": table.num_rows,
            "num_bytes": table.num_bytes,
            "created": table.created,
            "modified": table.modified,
        }
    except gcp_exceptions.NotFound:
        return None


def run_all_sql(dry_run: bool = False) -> None:
    """모든 SQL 파일 순차 실행"""
    config = load_config()

    # BigQuery 클라이언트 초기화
    client = bigquery.Client(project=config["gcp"]["project_id"])

    # 데이터셋 생성
    if not dry_run:
        create_dataset_if_not_exists(
            client,
            config["gcp"]["project_id"],
            config["bigquery"]["target_dataset"],
            config["bigquery"]["location"]
        )

    # SQL 파일 목록 (순서대로)
    sql_files = [
        "01_prepare_bq.sql",
        "02_build_features.sql",
        "03_build_labels.sql",
        "04_build_train.sql",
    ]

    for sql_file in sql_files:
        sql_path = SQL_DIR / sql_file
        if sql_path.exists():
            execute_sql_file(client, sql_path, config, dry_run)
        else:
            logger.warning(f"SQL 파일 없음: {sql_path}")

    # 결과 테이블 정보 출력
    if not dry_run:
        logger.info("\n=== 생성된 테이블 정보 ===")
        for table_id in ["features_customer", "labels_customer", "train_dataset"]:
            info = get_table_info(
                client,
                config["gcp"]["project_id"],
                config["bigquery"]["target_dataset"],
                table_id
            )
            if info:
                logger.info(f"  {table_id}: {info['num_rows']:,} rows")
            else:
                logger.warning(f"  {table_id}: 테이블 없음")


def run_single_sql(sql_file: str, dry_run: bool = False) -> None:
    """단일 SQL 파일 실행"""
    config = load_config()
    client = bigquery.Client(project=config["gcp"]["project_id"])

    sql_path = SQL_DIR / sql_file
    if not sql_path.exists():
        logger.error(f"SQL 파일 없음: {sql_path}")
        sys.exit(1)

    # 데이터셋 생성 (01 파일인 경우)
    if not dry_run and sql_file.startswith("01"):
        create_dataset_if_not_exists(
            client,
            config["gcp"]["project_id"],
            config["bigquery"]["target_dataset"],
            config["bigquery"]["location"]
        )

    execute_sql_file(client, sql_path, config, dry_run)


def main():
    parser = argparse.ArgumentParser(description="BigQuery SQL 실행 유틸리티")
    parser.add_argument(
        "--sql-file",
        type=str,
        help="실행할 SQL 파일 (예: 01_prepare_bq.sql). 지정하지 않으면 모든 SQL 실행"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="SQL을 실행하지 않고 출력만 (템플릿 치환 확인용)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="모든 SQL 파일 순차 실행"
    )

    args = parser.parse_args()

    if args.sql_file:
        run_single_sql(args.sql_file, args.dry_run)
    else:
        run_all_sql(args.dry_run)


if __name__ == "__main__":
    main()
