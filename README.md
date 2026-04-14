# Project Chimera: A Calibrated Utility-Based Recommendation Engine for Incremental Grocery Margin

## Project Charter and Core Principle

### Selected Problem
Recommendation System.

### Philosophical Stance
Recommendation is not prediction. Recommendation is controlled intervention under constraints.

- Prediction: "User will buy Milk."
- Intervention: "User will buy Artisan Bread because we showed it to them instead of Milk."

Our system answers: "Given limited screen real estate, which item maximizes the expected incremental contribution to retailer margin?"

### The Unified Scoring Principle
We approximate a utility function $U(i,u)$ for item $i$ and user $u$:

$$
U(i,u)=w_1\cdot Relevance(i,u)+w_2\cdot Uplift(i,u)+w_3\cdot Margin(i)+w_4\cdot Context(i,u)
$$

All components are normalized to the range $[0,1]$. This ensures mathematical consistency and allows transparent weight tuning.

### Success Metrics
- Primary: Incremental Precision@5 (items not purchased in training period).
- Secondary: Projected Incremental Margin Lift vs. Popularity Baseline.
- Validation method: Ablation Study showing marginal contribution of each utility component.

## Module 1: The Ground Truth Data Model and Baseline Metrics (Days 1-2)

### Member 1: Data Architect and KPI Strategist
Purpose: Create a clean, financially accurate, and semantically consistent data foundation. All downstream components depend on this.

### Focus Files
- transaction_data.csv
- product.csv
- hh_demographic.csv

### Step 1.1: Resolve the Sales_Value Trap
- Problem: SALES_VALUE is retailer revenue after manufacturer reimbursement, not customer price paid.
- Required output columns:
  - Revenue_Retailer: Net revenue per unit to retailer. Used for margin estimation.
  - Price_Paid_Customer: Out-of-pocket cost to customer. Used for price sensitivity context.
  - Is_Promoted_Item: Boolean (True if any discount column is non-zero).
- Validation: Manually verify 10 rows against User Guide page 3.

### Step 1.2: Product Hierarchy Roll-Up
- Decision: Aggregate to COMMODITY_DESC (approximately 400 categories).
- Rationale: 92k SKUs create a matrix that is 99.99% sparse. ALS cannot converge meaningfully. COMMODITY_DESC represents a distinct need state and is the standard level for grocery personalization.
- Output: master_transactions.parquet

### Step 1.3: Define the Organic Basket
- Definition: A BASKET_ID where every item has Is_Promoted_Item == False.
- Purpose: This subset isolates true behavioral affinity. In Module 2, Market Basket Analysis runs only on Organic Baskets. This prevents learning spurious associations like Chips + Salsa that only occur during BOGO promotions.
- Output:
  - master_transactions_all.parquet (for ALS training)
  - master_transactions_organic_only.parquet (for MBA)

### Step 1.4: Baseline Justification Metric
- Action: Calculate the Median Basket Diversity (distinct COMMODITY_DESC per basket).
- Purpose: This single number justifies the project existence. A median of 2 to 3 categories indicates a massive, untapped cross-sell surface area.
- Integration: This metric feeds into the Uplift component in Module 3. A user with low basket diversity receives a higher potential uplift score for any new category.

## Module 2: The Recall Layer - Candidate Generation (Days 3-4)

### Member 2: Machine Learning Engineer
Purpose: Generate a diverse set of 100 to 200 candidate items for each user. This stage prioritizes coverage and relevance.

### Focus Files
- master_transactions_organic_only.parquet
- master_transactions_all.parquet

### Step 2.1: Model A - Organic Market Basket Analysis (Familiarity Signal)
- Algorithm: FP-Growth.
- Input: Organic Baskets only.
- Output: Association rules with Lift score.
- Normalization for utility function:
  - If Lift >= 3.0, Relevance_MBA = 1.0
  - If Lift <= 1.0, Relevance_MBA = 0.0
  - Else: linear scaling between 1.0 and 3.0
- Interpretation: Captures "If they have X in their basket, they are very likely to appreciate Y."

### Step 2.2: Model B - Implicit Collaborative Filtering (Discovery Signal)
- Algorithm: Alternating Least Squares (ALS) via implicit library.
- Matrix values: Weighted confidence = 1.0 + log10(Revenue_Retailer + 1).
- Output: Predicted score per user-item pair.
- Normalization for utility function: ALS scores are unbounded, so apply Min-Max scaling per user to convert raw ALS score into [0,1] Relevance_ALS within that user's candidate set.
- Why per-user: Ensures that for a user with generally low predicted scores, their top relative item still gets a score near 1.0.
- Interpretation: Captures "Users with similar latent taste profiles enjoyed this item."

### Step 2.3: Candidate Union and Initial Filtering
- Logic: Combine top 50 ALS items and top 50 MBA consequents (based on last 3 basket items).
- De-duplication: If an item appears in both lists, keep the maximum of the two relevance scores.
- Crucial filter: Remove any item purchased by the user in the last 4 weeks. This is the first stage of uplift enforcement.

## Module 3: The Utility Function and Ranking Engine (Days 5-7)

### Member 3: Data Scientist and Member 4: Quantitative Marketer (Pair Programming)
Purpose: Re-rank the candidate set using a calibrated utility function. This is the core algorithmic contribution of the project.

### Focus Files
- Candidate Set (Module 2 output)
- campaign_table.csv
- commodity_margin.csv (created using business logic)

### Step 3.1: Component 1 - Relevance Score (Normalized)
- Definition: Relevance(i,u) = max(Relevance_ALS(i,u), Relevance_MBA(i,u))
- Rationale: Trust the stronger of the two signals. Both are already normalized to [0,1].

### Step 3.2: Component 2 - Uplift Score (The Sure Thing Penalty)
- Definition: Uplift(i,u) = 1.0 - Habit_Strength(i,u)
- Where Habit_Strength(i,u) = (Number of baskets containing item i) / (Total baskets for user u)
- Normalization: Naturally in [0,1].
- Interpretation: If a user buys Milk in 90% of baskets, Uplift = 0.10. The item is severely penalized. Recommending it adds zero incremental value.

### Step 3.3: Component 3 - Margin Score (Business Alignment)
- Data: commodity_margin.csv with column Normalized_Margin.
- Construction:
  - Estimate raw margin using business proxies (for example Private Label = 40%, National Brand = 20%).
  - Apply Min-Max scaling across all commodities to convert raw margin percent to [0,1].
  - Highest margin category gets 1.0 and lowest gets 0.0.
- Interpretation: Ensures the recommender favors items that contribute more to the retailer bottom line.

### Step 3.4: Component 4 - Context Score (Temporal Intelligence)
- Definition: Context(i,u) captures the interaction between the user deal sensitivity and the current promotional environment.
- Calculation:
  - Deal_Sensitivity(u) = (Number of baskets with any promoted item) / (Total baskets), naturally in [0,1]
  - Determine Is_Active_Campaign_Period from campaign_desc
  - Determine Item_Is_Promoted(i) from current promotional data

#### Context Score Logic Table

| Deal Sensitivity | Active Campaign? | Item Promoted? | Context Score | Rationale |
|---|---|---|---|---|
| High (>0.6) | Yes | Yes | 1.0 | Perfect alignment: deal junkie gets a deal |
| High (>0.6) | Yes | No | 0.5 | Misalignment: deal junkie sees full price |
| Low (<0.3) | Any | Yes | 0.2 | Margin protection: loyalist does not need coupon |
| Any | No | Yes | 0.5 | Promoted item but no campaign urgency |
| Default | Any | Any | 0.7 | Neutral context |

Interpretation: This component ensures the recommender adapts to the user current shopping mode and the retailer current marketing calendar.

### Step 3.5: The Unified Utility Function
- Formula:

$$
U(i,u)=w_1\cdot Relevance(i,u)+w_2\cdot Uplift(i,u)+w_3\cdot Margin(i)+w_4\cdot Context(i,u)
$$

- Initial weights (baseline):
  - $w_1=0.40$
  - $w_2=0.25$
  - $w_3=0.20$
  - $w_4=0.15$
- Rationale: Relevance is the primary driver. Uplift is a strong modifier. Margin and Context provide fine-tuning.
- Output: Sorted list of top 5 items per user.

## Module 4: Offline Validation and Ablation Study (Day 8)

### Member 4: Quantitative Marketer (Evaluation)
Purpose: Rigorously prove that each component of the utility function adds measurable value.

### Focus Files
- Temporal holdout data (Weeks 81-102)
- hh_demographic.csv (for cold start)

### Step 4.1: Temporal Holdout Setup
- Training: Day 1 to Day 600
- Testing: Day 601 to Day 711
- Metric: Incremental Precision@5 (item counts only if not purchased in training period)

### Step 4.2: The Ablation Study (Crucial for Academic Rigor)
- Methodology: Run the ranking engine four times, each time removing one component of the utility function.

| Model Variant | Utility Components Active | Purpose of Test |
|---|---|---|
| Variant 0 (Baseline) | Relevance only ($w_1=1.0$) | Isolate pure collaborative and associative power |
| Variant 1 | Relevance + Uplift | Measure impact of removing sure things |
| Variant 2 | Relevance + Uplift + Margin | Measure impact of profit alignment |
| Variant 3 (Full Chimera) | All four components | Measure impact of temporal context |

Expected narrative for report:
- Variant 0: High raw precision, but low incremental precision (recommends Milk).
- Variant 1: Incremental precision jumps +15% (stopped wasting slots).
- Variant 2: Average recommended margin increases +22% (shifted mix to Private Label).
- Variant 3: Conversion rate for promo-junkies increases during campaign windows +8%.

### Step 4.3: Cold Start Protocol
- Problem: New users have no history for ALS or Habit Strength.
- Solution: Demographic prior.
  - Use hh_demographic to map new users to a pre-computed segment baseline.
  - For these users, relevance is based on segment-level popularity.
  - Uplift is set to a neutral 0.5 (unknown habit).
  - The utility function still operates with the remaining components.

## Module 5: The Executive Narrative and Simulation Dashboard (Days 9-10)

### Member 5: BI Architect and Creative Lead
Purpose: Translate the utility function into a compelling, defensible business story.

Focus: Power BI Dashboard and Final Report PDF.

### Step 5.1: The Deconfounded Case Study (The John Smith Walkthrough)
- Action: Select a specific household_key.
- Visual narrative:
  - User history: Show purchase timeline overlaid with campaign windows.
  - Variant 0 output: Model recommends Milk and Soda.
  - Chimera diagnosis: A table showing utility score breakdown for Milk vs Artisan Bread.
    - Milk: Relevance=0.95, Uplift=0.10, Margin=0.20, Context=0.70, Total=0.48
    - Artisan Bread: Relevance=0.70, Uplift=0.90, Margin=0.85, Context=0.70, Total=0.76
  - Conclusion: Artisan Bread wins because it represents a high-margin, non-habitual opportunity.

### Step 5.2: Dashboard Specifications (Power BI)
- Screen 1: Utility Decomposition (The Why Screen)
  - Visual: Stacked bar chart for top 5 recommendations of a selected user.
  - Bars show contribution of Relevance, Uplift, Margin, and Context to final score.
  - Purpose: Makes the black box transparent by showing exactly why an item was chosen.

- Screen 2: The Ablation Impact (The Proof Screen)
  - Visual: Slope chart or small-multiples bar chart showing lift in Incremental Precision@5 and Average Recommended Margin from Variant 0 to Variant 3.
  - Purpose: Quantifies ROI of each engineering layer.

- Screen 3: The Recommendation Simulator (Operational Tool)
  - Input: household_key search bar.
  - Output cards: Persona, Top Pick, and utility score formula rendered numerically (for example, 0.70 + 0.90 + 0.85 = 2.45).

Final report title:
Project Chimera: A Utility-Based Framework for Incremental, Margin-Aware Grocery Recommendations.

## Repository Structure and File Manifest

Repo name: chimera-utility-recsys

```text
chimera-utility-recsys/
├── README.md
├── requirements.txt
│
├── data/
│   ├── 01_raw/                        # Original CSVs (ignored by Git)
│   └── 02_processed/
│       ├── master_transactions_all.parquet
│       ├── master_transactions_organic.parquet
│       ├── association_rules.csv
│       ├── user_factors.npz
│       ├── item_factors.npz
│       ├── commodity_margin.csv       # Lookup table with Normalized_Margin
│       └── user_context_features.parquet
│
├── notebooks/
│   ├── 01_ground_truth_data_model.ipynb
│   ├── 02_candidate_generation_mba_als.ipynb
│   ├── 03_utility_function_ranking.ipynb
│   ├── 04_ablation_study_validation.ipynb
│   └── 05_cold_start_rules.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── financial_utils.py
│   ├── recall_engine.py               # MBA + ALS candidate generation
│   ├── utility_scorer.py              # The U(i,u) function implementation
│   └── cold_start.py
│
├── reports/
│   ├── figures/
│   │   ├── ablation_lift_chart.png
│   │   ├── utility_decomposition.png
│   │   └── john_smith_case_study.png
│   ├── chimera_final_report.pdf
│   └── chimera_dashboard.pbix
│
└── tests/
    ├── test_normalization.py
    └── test_utility_function.py
```
