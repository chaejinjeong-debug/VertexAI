-- 01_prepare_bq.sql
-- theLook eCommerce 원본 데이터에서 필요한 테이블/뷰 준비
-- 데이터셋 생성 및 기본 뷰 설정

-- 참고: 데이터셋 생성은 Python에서 수행 (CREATE SCHEMA IF NOT EXISTS는 제한적)
-- 이 SQL은 데이터셋이 이미 존재한다고 가정

-- 1. 주문 정보 뷰 (최근 데이터만 필터링)
CREATE OR REPLACE VIEW `{project_id}.{target_dataset}.v_orders` AS
SELECT
    order_id,
    user_id AS customer_id,
    status,
    created_at,
    num_of_item,
    EXTRACT(DATE FROM created_at) AS order_date
FROM
    `bigquery-public-data.thelook_ecommerce.orders`
WHERE
    status = 'Complete'
    AND created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {lookback_months} MONTH);

-- 2. 주문 상세 뷰 (매출 계산용)
CREATE OR REPLACE VIEW `{project_id}.{target_dataset}.v_order_items` AS
SELECT
    oi.id AS item_id,
    oi.order_id,
    oi.user_id AS customer_id,
    oi.product_id,
    oi.sale_price,
    oi.created_at,
    EXTRACT(DATE FROM oi.created_at) AS order_date
FROM
    `bigquery-public-data.thelook_ecommerce.order_items` oi
INNER JOIN
    `bigquery-public-data.thelook_ecommerce.orders` o
    ON oi.order_id = o.order_id
WHERE
    o.status = 'Complete'
    AND oi.created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {lookback_months} MONTH);

-- 3. 고객 정보 뷰
CREATE OR REPLACE VIEW `{project_id}.{target_dataset}.v_customers` AS
SELECT
    id AS customer_id,
    first_name,
    last_name,
    email,
    age,
    gender,
    country,
    city,
    created_at AS registered_at
FROM
    `bigquery-public-data.thelook_ecommerce.users`;

-- 4. 피처 계산용 날짜 스파인 (feature_timestamp 기준점)
-- 최근 N개월 동안 매일 스냅샷 생성
CREATE OR REPLACE VIEW `{project_id}.{target_dataset}.v_date_spine` AS
SELECT
    date AS snapshot_date
FROM
    UNNEST(GENERATE_DATE_ARRAY(
        DATE_SUB(CURRENT_DATE(), INTERVAL {lookback_months} MONTH),
        CURRENT_DATE(),
        INTERVAL 1 DAY
    )) AS date;
