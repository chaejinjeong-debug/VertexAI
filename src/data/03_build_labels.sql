-- 03_build_labels.sql
-- labels_customer 테이블 생성
-- 각 고객별, 날짜별 라벨 생성 (14일 내 재구매 여부)

CREATE OR REPLACE TABLE `{project_id}.{target_dataset}.labels_customer` AS

WITH customer_labels AS (
    SELECT
        ds.snapshot_date AS label_date,
        c.customer_id,

        -- 해당 날짜 이후 14일 내 주문이 있는지 확인
        CASE
            WHEN EXISTS (
                SELECT 1
                FROM `{project_id}.{target_dataset}.v_orders` o
                WHERE o.customer_id = c.customer_id
                  AND o.order_date > ds.snapshot_date
                  AND o.order_date <= DATE_ADD(ds.snapshot_date, INTERVAL 14 DAY)
            )
            THEN 1
            ELSE 0
        END AS label_repurchase_14d

    FROM
        `{project_id}.{target_dataset}.v_date_spine` ds
    CROSS JOIN
        (SELECT DISTINCT customer_id FROM `{project_id}.{target_dataset}.v_orders`) c
    WHERE
        -- 라벨 생성은 14일 전까지만 (미래 데이터 필요하므로)
        ds.snapshot_date <= DATE_SUB(CURRENT_DATE(), INTERVAL 14 DAY)
)

SELECT
    customer_id,
    TIMESTAMP(label_date) AS label_timestamp,
    label_repurchase_14d
FROM
    customer_labels
ORDER BY
    customer_id, label_timestamp;
