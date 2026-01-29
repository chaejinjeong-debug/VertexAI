-- 05_sanity_checks.sql
-- 데이터 품질 검증 쿼리
-- 라벨 분포, NULL rate, 기간 범위 등 확인

-- 1. 라벨 분포 확인 (Churn Rate)
-- positive rate가 0% 또는 100%면 경고 필요
SELECT
    '1_label_distribution' AS check_name,
    COUNT(*) AS total_rows,
    SUM(label_churn_{churn_days}d) AS churned_count,
    COUNT(*) - SUM(label_churn_{churn_days}d) AS retained_count,
    ROUND(AVG(label_churn_{churn_days}d) * 100, 2) AS churn_rate_pct,
    CASE
        WHEN AVG(label_churn_{churn_days}d) = 0 THEN 'WARNING: All labels are 0 (no churn)'
        WHEN AVG(label_churn_{churn_days}d) = 1 THEN 'WARNING: All labels are 1 (all churn)'
        WHEN AVG(label_churn_{churn_days}d) < 0.05 THEN 'WARNING: Churn rate < 5%'
        WHEN AVG(label_churn_{churn_days}d) > 0.95 THEN 'WARNING: Churn rate > 95%'
        ELSE 'OK: Balanced distribution'
    END AS status
FROM `{project_id}.{target_dataset}.labels_customer_churn`;

-- 2. 피처 테이블 NULL rate 확인
SELECT
    '2_feature_null_rate' AS check_name,
    COUNT(*) AS total_rows,
    ROUND(COUNTIF(orders_30d IS NULL) / COUNT(*) * 100, 2) AS orders_30d_null_pct,
    ROUND(COUNTIF(orders_90d IS NULL) / COUNT(*) * 100, 2) AS orders_90d_null_pct,
    ROUND(COUNTIF(revenue_30d IS NULL) / COUNT(*) * 100, 2) AS revenue_30d_null_pct,
    ROUND(COUNTIF(revenue_90d IS NULL) / COUNT(*) * 100, 2) AS revenue_90d_null_pct,
    ROUND(COUNTIF(avg_order_value_90d IS NULL) / COUNT(*) * 100, 2) AS avg_order_value_90d_null_pct,
    ROUND(COUNTIF(days_since_last_order IS NULL) / COUNT(*) * 100, 2) AS days_since_last_order_null_pct
FROM `{project_id}.{target_dataset}.features_customer`;

-- 3. Train dataset NULL rate 및 조인 결과 확인
SELECT
    '3_train_data_quality' AS check_name,
    COUNT(*) AS total_rows,
    COUNT(DISTINCT customer_id) AS unique_customers,
    ROUND(COUNTIF(orders_90d IS NULL) / COUNT(*) * 100, 2) AS feature_null_pct,
    ROUND(COUNTIF(label_churn_{churn_days}d IS NULL) / COUNT(*) * 100, 2) AS label_null_pct,
    ROUND(AVG(label_churn_{churn_days}d) * 100, 2) AS train_churn_rate_pct
FROM `{project_id}.{target_dataset}.train_dataset`;

-- 4. 기간 범위 확인
SELECT
    '4_date_range' AS check_name,
    'features_customer' AS table_name,
    MIN(feature_timestamp) AS min_date,
    MAX(feature_timestamp) AS max_date,
    DATE_DIFF(DATE(MAX(feature_timestamp)), DATE(MIN(feature_timestamp)), DAY) AS date_range_days
FROM `{project_id}.{target_dataset}.features_customer`
UNION ALL
SELECT
    '4_date_range' AS check_name,
    'labels_customer_churn' AS table_name,
    MIN(label_timestamp) AS min_date,
    MAX(label_timestamp) AS max_date,
    DATE_DIFF(DATE(MAX(label_timestamp)), DATE(MIN(label_timestamp)), DAY) AS date_range_days
FROM `{project_id}.{target_dataset}.labels_customer_churn`
UNION ALL
SELECT
    '4_date_range' AS check_name,
    'train_dataset' AS table_name,
    MIN(label_timestamp) AS min_date,
    MAX(label_timestamp) AS max_date,
    DATE_DIFF(DATE(MAX(label_timestamp)), DATE(MIN(label_timestamp)), DAY) AS date_range_days
FROM `{project_id}.{target_dataset}.train_dataset`;

-- 5. 피처 분포 통계
SELECT
    '5_feature_stats' AS check_name,
    ROUND(AVG(orders_30d), 2) AS avg_orders_30d,
    ROUND(AVG(orders_90d), 2) AS avg_orders_90d,
    ROUND(AVG(revenue_30d), 2) AS avg_revenue_30d,
    ROUND(AVG(revenue_90d), 2) AS avg_revenue_90d,
    ROUND(AVG(avg_order_value_90d), 2) AS avg_order_value_90d,
    ROUND(AVG(days_since_last_order), 2) AS avg_days_since_last_order,
    ROUND(AVG(distinct_products_90d), 2) AS avg_distinct_products_90d,
    ROUND(AVG(distinct_categories_90d), 2) AS avg_distinct_categories_90d
FROM `{project_id}.{target_dataset}.train_dataset`;

-- 6. 고객별 라벨 변화 확인 (이탈 후 복귀 패턴)
SELECT
    '6_customer_label_patterns' AS check_name,
    COUNT(DISTINCT customer_id) AS total_customers,
    -- 이탈 경험 있는 고객 수
    COUNT(DISTINCT CASE WHEN label_churn_{churn_days}d = 1 THEN customer_id END) AS customers_with_churn,
    -- 항상 유지된 고객 수
    COUNT(DISTINCT customer_id) - COUNT(DISTINCT CASE WHEN label_churn_{churn_days}d = 1 THEN customer_id END) AS always_retained_customers
FROM `{project_id}.{target_dataset}.labels_customer_churn`;
