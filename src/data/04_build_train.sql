-- 04_build_train.sql
-- train_dataset 생성 (point-in-time join)
-- labels_customer의 각 row에 대해, feature_timestamp <= label_timestamp인 가장 최신 피처 조인

CREATE OR REPLACE TABLE `{project_id}.{target_dataset}.train_dataset` AS

WITH labels_with_features AS (
    SELECT
        l.customer_id,
        l.label_timestamp,
        l.label_repurchase_14d,

        -- Point-in-time: 라벨 시점 이전의 가장 최신 피처
        f.feature_timestamp,
        f.orders_7d,
        f.orders_30d,
        f.revenue_30d,
        f.avg_order_value_30d,
        f.distinct_products_30d,
        f.days_since_last_order,

        -- 가장 최신 피처를 선택하기 위한 순위
        ROW_NUMBER() OVER (
            PARTITION BY l.customer_id, l.label_timestamp
            ORDER BY f.feature_timestamp DESC
        ) AS rn

    FROM
        `{project_id}.{target_dataset}.labels_customer` l
    INNER JOIN
        `{project_id}.{target_dataset}.features_customer` f
        ON l.customer_id = f.customer_id
        AND f.feature_timestamp <= l.label_timestamp  -- Point-in-time 조건 (누수 방지)
)

SELECT
    customer_id,
    label_timestamp,
    feature_timestamp,

    -- 피처들
    orders_7d,
    orders_30d,
    revenue_30d,
    COALESCE(avg_order_value_30d, 0) AS avg_order_value_30d,
    distinct_products_30d,
    days_since_last_order,

    -- 라벨
    label_repurchase_14d

FROM
    labels_with_features
WHERE
    rn = 1  -- 가장 최신 피처만 선택
    AND orders_30d > 0  -- 최근 30일 내 주문이 있는 경우만 (활성 고객)
ORDER BY
    customer_id, label_timestamp;
