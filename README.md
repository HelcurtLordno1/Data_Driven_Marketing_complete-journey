# Project Apollo: The Complete Journey Analysis

10-day blueprint for a multi-tier hybrid personalization and campaign uplift engine using the dunnhumby Complete Journey dataset.

## Executive Summary

This project builds a full analytics-to-action pipeline that answers one critical business question:

Is observed customer behavior true loyalty, or campaign-induced behavior?

The strategy is to combine:

- Behavioral truth: what households buy.
- Promotional truth: what campaigns and coupons they receive and redeem.
- Environmental truth: what in-store exposure they have (display and mailer).

The output is a practical decision engine for:

- Who to target.
- What to recommend.
- How to maximize incremental margin while minimizing wasted promotional spend.

## Project Philosophy: The Ground Truth Advantage

Unlike synthetic datasets, dunnhumby provides three simultaneous views of customer reality:

1. Behavioral truth: transaction_data (basket-level purchase behavior).
2. Promotional truth: campaign_table, coupon, coupon_redempt (offer delivery and redemption behavior).
3. Environmental truth: causal_data (display and mailer exposure context).

This enables deconfounding marketing impact:

- Did a customer buy because they are truly loyal?
- Or because the product was featured on an end-cap and mailer during campaign days?

## Repository Structure

```text
project_ddm_complete_journey/
|
|-- README.md
|-- requirements.txt
|-- .gitignore
|
|-- data/
|   |-- 01_raw/
|   |   |-- campaign_desc.csv
|   |   |-- campaign_table.csv
|   |   |-- causal_data.csv
|   |   |-- coupon.csv
|   |   |-- coupon_redempt.csv
|   |   |-- hh_demographic.csv
|   |   |-- product.csv
|   |   `-- transaction_data.csv
|   |
|   `-- 02_processed/
|       |-- master_transactions.parquet
|       |-- rfm_lrfmc_features.parquet
|       |-- cluster_assignments.parquet
|       |-- user_item_matrix.npz
|       `-- association_rules.csv
|
|-- notebooks/
|   |-- 01_member1_data_audit.ipynb
|   |-- 02_member2_segmentation_rfm.ipynb
|   |-- 03_member3_mba_analysis.ipynb
|   |-- 04_member3_recommender_svd.ipynb
|   |-- 05_member4_clv_bgnbd.ipynb
|   `-- 06_member4_ab_test_design.ipynb
|
|-- src/
|   |-- __init__.py
|   |-- data_loader.py
|   |-- financial_utils.py
|   |-- clustering_pipeline.py
|   |-- recsys_engine.py
|   `-- uplift_scorer.py
|
|-- reports/
|   |-- figures/
|   |   |-- cohort_retention_heatmap.png
|   |   |-- cluster_radar_chart.png
|   |   |-- network_graph_mba.html
|   |   `-- uplift_quadrant_scatter.png
|   |-- apollo_final_report.pdf
|   `-- apollo_dashboard.pbix
|
|-- tests/
|   |-- test_financial_utils.py
|   `-- test_cluster_stability.py
|
`-- archive/
    `-- old_apriori_attempt.ipynb
```

## Setup Instructions

1. Create and activate a Python environment (recommended Python 3.10+).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Place raw dunnhumby files in data/01_raw.
4. Run notebooks in order from 01 to 06.
5. Move reusable logic into src modules and lock behavior with tests.

## Core File Manifest and Purpose

### Root Files

- README.md: project narrative, architecture, setup, and module plan.
- requirements.txt: pandas, scikit-learn, mlxtend, lifetimes, implicit, powerbiclient.
- .gitignore: ignores raw data, cache files, and local environments.

### Data Layer

- data/01_raw: immutable source files from the dunnhumby zip.
- data/02_processed: cleaned, joined, model-ready outputs.

Critical processed file:

- master_transactions.parquet
  - Why Parquet: joining 138 MB transactions with 679 MB exposure data in CSV can become multi-GB and slow.
  - Benefit: columnar compression and fast column reads for analysis and modeling.
  - Typical columns:
    - household_key
    - BASKET_ID
    - DAY
    - COMMODITY_DESC
    - Revenue_Retailer
    - Price_Paid_Customer
    - display_flag
    - mailer_flag

### Notebooks Layer

- 01_member1_data_audit.ipynb: financial sanity checks and cohort retention heatmap.
- 02_member2_segmentation_rfm.ipynb: LRFMC feature engineering and clustering visuals.
- 03_member3_mba_analysis.ipynb: FP-Growth on organic baskets and rule mining.
- 04_member3_recommender_svd.ipynb: ALS training and ranking evaluation.
- 05_member4_clv_bgnbd.ipynb: BG/NBD, CLV projection, and uplift filtering.
- 06_member4_ab_test_design.ipynb: power analysis and sample-size design.

### Source Code Layer

- src/data_loader.py
  - Class: DunnhumbyLoader for robust ingestion and chunking of large files (especially causal_data.csv).
- src/financial_utils.py
  - Single source of truth for financial formulas.
  - Functions: calculate_true_price, calculate_margin.
- src/clustering_pipeline.py
  - Class: RFMTransformer for segmentation preparation (including skew handling and tie-safe workflow).
- src/recsys_engine.py
  - Class: HybridRecommender to combine MBA signals with ALS fallback and reranking.
- src/uplift_scorer.py
  - Functions: calculate_expected_profit and filter_persuadables for campaign optimization.

### Reports Layer

- reports/figures/network_graph_mba.html
  - Interactive product graph (pyvis or plotly style) to explore affinity links and lift strength.
  - Designed for high-impact presentation beyond static tables.
- reports/apollo_final_report.pdf: final narrative report.
- reports/apollo_dashboard.pbix: executive Power BI dashboard.

### Test Layer

- tests/test_financial_utils.py: validates true price and margin formulas against specification.
- tests/test_cluster_stability.py: validates clustering quality threshold (for example silhouette > 0.2).

## Key Business Logic and Code Anchors

### 1) Sales_Value Trap Resolution (Financial Grounding)

Sales value is retailer revenue, not direct customer paid price. Keep both views explicitly.

```python
import pandas as pd

df = pd.read_csv("data/01_raw/transaction_data.csv")

# Normalize discount signs for stable formula behavior.
df["RETAIL_DISC"] = df["RETAIL_DISC"].abs() * -1
df["COUPON_MATCH_DISC"] = df["COUPON_MATCH_DISC"].abs() * -1

df["Price_Paid_Customer"] = (
    df["SALES_VALUE"]
    - df["RETAIL_DISC"].fillna(0)
    - df["COUPON_MATCH_DISC"].fillna(0)
) / df["QUANTITY"]

df["Revenue_Retailer"] = df["SALES_VALUE"] / df["QUANTITY"]
```

Recommended team memo outcome:

- Use Revenue_Retailer for CLV and retailer-side economics.
- Use Price_Paid_Customer for price sensitivity and elasticity analysis.

### 2) Product Hierarchy Roll-Up

Join transactions with product metadata via PRODUCT_ID, then model at COMMODITY_DESC granularity to avoid both over-fragmentation and over-aggregation.

### 3) Cohort Baseline KPI

Build cohort retention heatmap to reveal retention heterogeneity hidden by single averages.

- Rows: cohort start month.
- Columns: months since first purchase.
- Use this as the business urgency slide.

### 4) Uplift Filter Logic (Sure Things Removal)

```python
# The Moment of Truth: filtering out sure-things
df["target"] = (df["prob_alive"] < 0.95) & (df["campaign_responsive"] == True)

saved = cost_of_campaign * len(df[~df["target"]])
print(
    f"Filtered out {len(df[~df['target']])} Sure Things. "
    f"Saved ${saved:,.0f} in wasted marketing."
)
```

## 10-Day Execution Blueprint (5 Modules)

### Module 1 (Days 1-2): Financial Foundation and Data Sanitization

Role: Data Architect and KPI Strategist

Focus files:

- transaction_data.csv
- product.csv
- hh_demographic.csv

Steps:

1. Resolve Sales_Value trap and compute dual monetary views.
2. Build master transaction table at commodity level.
3. Build acquisition cohorts and retention baseline.

Primary deliverables:

- Financial grounding memo.
- cohort_retention_heatmap.png.

### Module 2 (Days 3-4): Advanced Segmentation and Campaign Intensity

Role: Lead Data Scientist (Clustering)

Feature framework: LRFMC

- L (Length): max(DAY) - min(DAY)
- R (Recency): snapshot_day - max(DAY)
- F (Frequency): distinct basket count
- M (Monetary): sum of retailer revenue
- C (Campaign Intensity): campaign count or exposure proxy

Preprocessing:

- Apply Yeo-Johnson transform on skewed dimensions such as Frequency and Monetary.

Actionable cluster framing:

| Cluster | RFM Profile           | Campaign Intensity | Demographic Signal | Strategic Name        |
|--------:|-----------------------|--------------------|--------------------|-----------------------|
| 0       | High F, High M, Low R | Low                | Level 10+, Group 5 | Organic Loyalists     |
| 1       | High F, High M, Low R | High               | Level 8-9          | Promo-Triggered Elite |
| 2       | High F, Low M, Low R  | Medium             | Level 3-5          | Fill-In Trippers      |
| 3       | Low F, High R         | Low                | Level 1-2          | Dormant/Lost          |

Primary deliverables:

- rfm_lrfmc_features.parquet
- cluster_assignments.parquet
- cluster_radar_chart.png

### Module 3 (Days 5-6): Campaign Uplift Lab and Hybrid Recommendations

Role: AI/ML Engineer (Causal Inference + Recommender)

Focus files:

- campaign_desc.csv
- coupon.csv
- coupon_redempt.csv
- causal_data.csv

Key guidance for large causal file:

- Filter to relevant product_id universe before merging.
- Consider out-of-core tools (Dask or Polars) if full scan is required.

Uplift segment taxonomy:

- Sure Things: buy regardless, do not discount.
- Persuadables: campaign-responsive, high ROI target.
- Lost Causes: low responsiveness, deprioritize.
- Sleeping Dogs: may react negatively to unnecessary intervention.

Hybrid recommender logic:

- MBA on organic baskets (coupon_disc == 0) for true affinity rules.
- ALS on user-item matrix for broader collaborative signal.
- Strategy:
  - Cluster 0: prioritize discovery and margin-aware ALS adjacencies.
  - Cluster 1: prioritize familiar high-lift MBA cross-sell rules.

Primary deliverables:

- association_rules.csv
- user_item_matrix.npz
- network_graph_mba.html

### Module 4 (Days 7-8): CLV and Decision Optimization

Role: Quantitative Marketer

Modeling core:

- BG/NBD for predicted purchases.
- Convert to value estimate:

CLV_12m = predicted_purchases_12m * avg_revenue_per_basket * estimated_margin

Expected profit scoring:

Expected_Profit = (Prob_Conversion * CLV_12m * Margin) - Cost_of_Campaign

Policy rule:

- Remove Prob_Conversion > 0.95 (sure-things).
- Prioritize 0.4 < Prob_Conversion < 0.8 and Uplift_Segment == Persuadable.

A/B test design assumptions:

- Baseline conversion: 15%
- MDE target: 5% uplift (or 10% given sample constraints)
- Power: 80%
- Alpha: 0.05
- Suggested duration: 21 days

### Module 5 (Days 9-10): Executive BI and Boardroom Narrative

Role: Creative Lead and BI Architect

Case narrative:

- Household journey deep dive (for example household 1301).
- Timeline events:
  - Organic purchases
  - Campaign exposures
  - Coupon redemptions

Dashboard concept:

1. Monitor screen: ROMI gauge (for example 1.8x to 2.5x trajectory).
2. Diagnose screen: CLV vs propensity scatter with uplift segment coloring.
3. Execute screen: household lookup with cluster, predicted CLV, and top recommendations.

Final report narrative target:

From mass mailer to precision grocery: increase incremental margin via targeted persuasion, not blanket discounting.

## Suggested Coding Workflow for Team Consistency

1. Prototype logic in notebooks.
2. Move stable logic into src modules.
3. Write tests for critical formulas and model quality gates.
4. Save intermediate artifacts into data/02_processed.
5. Export visuals and executive assets into reports.

## Risks and Guardrails

- Financial misinterpretation risk:
  - Guardrail: centralize formulas in src/financial_utils.py and test them.
- Deconfounding risk:
  - Guardrail: always include campaign and display or mailer context in evaluation.
- Data size risk:
  - Guardrail: chunk reads, filtered joins, and Parquet outputs.
- Over-targeting risk:
  - Guardrail: exclude sure-things and low-probability lost causes from discount spend.

## Immediate Next Steps

1. Implement production versions of formulas and transforms in src modules.
2. Fill notebook cells with module-specific runnable pipelines.
3. Add robust tests for financial integrity and segmentation stability.
4. Build Power BI semantic model using processed datasets.
