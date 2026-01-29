-- 02_build_features.sql
-- features_customer 스냅샷 테이블 생성
-- 각 고객별, 날짜별 피처 스냅샷 생성 (point-in-time 학습용)

CREATE OR REPLACE TABLE `{project_id}.{target_dataset}.features_customer` AS

WITH customer_order_stats AS (
    -- 각 고객의 주문 내역 집계 (날짜별)
    SELECT
        ds.snapshot_date,
        c.customer_id,

        -- 최근 7일 주문 수
        COUNTIF(
            o.order_date > DATE_SUB(ds.snapshot_date, INTERVAL 7 DAY)
            AND o.order_date <= ds.snapshot_date
        ) AS orders_7d,

        -- 최근 30일 주문 수
        COUNTIF(
            o.order_date > DATE_SUB(ds.snapshot_date, INTERVAL 30 DAY)
            AND o.order_date <= ds.snapshot_date
        ) AS orders_30d,

        -- 최근 30일 매출
        COALESCE(SUM(
            CASE WHEN oi.order_date > DATE_SUB(ds.snapshot_date, INTERVAL 30 DAY)
                      AND oi.order_date <= ds.snapshot_date
                 THEN oi.sale_price
                 ELSE 0
            END
        ), 0) AS revenue_30d,

        -- 최근 30일 구매 상품 수 (고유)
        COUNT(DISTINCT
            CASE WHEN oi.order_date > DATE_SUB(ds.snapshot_date, INTERVAL 30 DAY)
                      AND oi.order_date <= ds.snapshot_date
                 THEN oi.product_id
            END
        ) AS distinct_products_30d,

        -- 마지막 주문일
        MAX(CASE WHEN o.order_date <= ds.snapshot_date THEN o.order_date END) AS last_order_date

    FROM
        `{project_id}.{target_dataset}.v_date_spine` ds
    CROSS JOIN
        (SELECT DISTINCT customer_id FROM `{project_id}.{target_dataset}.v_orders`) c
    LEFT JOIN
        `{project_id}.{target_dataset}.v_orders` o
        ON c.customer_id = o.customer_id
    LEFT JOIN
        `{project_id}.{target_dataset}.v_order_items` oi
        ON c.customer_id = oi.customer_id
    GROUP BY
        ds.snapshot_date, c.customer_id
)

SELECT
    customer_id,
    TIMESTAMP(snapshot_date) AS feature_timestamp,

    -- 피처들
    CAST(orders_7d AS INT64) AS orders_7d,
    CAST(orders_30d AS INT64) AS orders_30d,
    ROUND(revenue_30d, 2) AS revenue_30d,
    ROUND(
        SAFE_DIVIDE(revenue_30d, NULLIF(orders_30d, 0)),
        2
    ) AS avg_order_value_30d,
    CAST(distinct_products_30d AS INT64) AS distinct_products_30d,

    -- 마지막 주문 후 경과일 (없으면 999)
    COALESCE(
        DATE_DIFF(snapshot_date, last_order_date, DAY),
        999
    ) AS days_since_last_order

FROM
    customer_order_stats
WHERE
    -- 주문 이력이 있는 고객만 (최소 1회 주문)
    last_order_date IS NOT NULL
ORDER BY
    customer_id, feature_timestamp;
