-- 03_build_labels_churn.sql
-- labels_customer_churn 테이블 생성
-- 각 고객별, 날짜별 이탈 라벨 생성 (N일 내 구매 없음 = 이탈)
-- 중요: Date spine은 orders 데이터 범위 기반 (CURRENT_DATE() 의존 제거)

CREATE OR REPLACE TABLE `{project_id}.{target_dataset}.labels_customer_churn` AS

WITH order_date_range AS (
    -- 주문 데이터의 실제 날짜 범위 계산
    SELECT
        MIN(order_date) AS min_order_date,
        MAX(order_date) AS max_order_date
    FROM `{project_id}.{target_dataset}.v_orders`
),

-- Orders 범위 기반 date spine (라벨 생성용)
-- 상한: MAX(order_date) - churn_days (라벨 계산에 미래 데이터 필요)
label_date_spine AS (
    SELECT
        date AS label_date
    FROM
        order_date_range,
        UNNEST(GENERATE_DATE_ARRAY(
            min_order_date,
            DATE_SUB(max_order_date, INTERVAL {churn_days} DAY),
            INTERVAL 1 DAY
        )) AS date
),

customer_labels AS (
    SELECT
        ds.label_date,
        c.customer_id,

        -- 해당 날짜 이후 N일 내 주문이 없으면 이탈(1)
        CASE
            WHEN EXISTS (
                SELECT 1
                FROM `{project_id}.{target_dataset}.v_orders` o
                WHERE o.customer_id = c.customer_id
                  AND o.order_date > ds.label_date
                  AND o.order_date <= DATE_ADD(ds.label_date, INTERVAL {churn_days} DAY)
            )
            THEN 0  -- 구매 있음 = 이탈 아님
            ELSE 1  -- 구매 없음 = 이탈
        END AS label_churn_{churn_days}d

    FROM
        label_date_spine ds
    CROSS JOIN
        (SELECT DISTINCT customer_id FROM `{project_id}.{target_dataset}.v_orders`) c
    WHERE
        -- 해당 날짜 이전에 최소 1회 주문이 있는 고객만 (기존 고객)
        EXISTS (
            SELECT 1
            FROM `{project_id}.{target_dataset}.v_orders` o
            WHERE o.customer_id = c.customer_id
              AND o.order_date <= ds.label_date
        )
)

SELECT
    customer_id,
    TIMESTAMP(label_date) AS label_timestamp,
    label_churn_{churn_days}d
FROM
    customer_labels
ORDER BY
    customer_id, label_timestamp;
