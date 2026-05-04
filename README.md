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

### Implementation Rule
Treat `src/` as the source of truth for reusable logic.

- Put long transformation, modeling, and scoring code in `src/`.
- Keep notebooks focused on orchestration, visuals, validation, and insights.
- Import notebook-ready functions from `src/` instead of duplicating pipeline code inline.

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

Extended Project Plan: Project Chimera - From Proof-of-Concept to Production-Ready Personalization

All new modules reuse the data artifacts from Modules 1-5 (master transactions, candidate set, top-5 recommendations, ablation summary) and add specific validation, interpretability, and deployment layers.

## Module 6: Recommendation Impact on Basket Behaviour (Day 11)

### Member 6: Behavior Analyst and Evaluation Lead
Purpose: Quantify how the recommender changes real purchase patterns, not just offline metrics.

### Focus Files
- Temporal holdout from Module 4
- Module 5 recommendation outputs
- commodity_margin.csv

### Step 6.1: Pre-Post Analysis of Basket Composition
- Use the temporal holdout (Module 4 split).
- For each household, compare the test-period basket (what they actually bought) with the recommended items.
- Calculate Category Expansion Rate = percentage of households that purchased at least one new commodity category that appeared in their Top-5 recommendations.
- Compare this rate between the Full Chimera and the Popularity baseline.

### Step 6.2: Margin Shift Index
- Compute the average Normalized_Margin of items purchased in the test period vs. the training period.
- Show whether households that received Chimera recommendations moved toward higher-margin categories.

### Step 6.3: Basket Size Uplift
- Compare the average number of distinct commodities per basket in the test period between households that had Chimera recommendations (simulated) and those that would have received Popularity recommendations.
- Visual: histogram of basket diversity, split by treatment type.

### Step 6.4: Hit-Rate vs. Discovery Trade-Off
- For each household, measure the overlap between recommended items and actual purchases (hit-rate) against the number of new categories purchased.
- Show that the utility function achieves a better trade-off (higher discovery without sacrificing hit-rate) than a pure relevance model.

### Deliverables
- Notebook: 06_basket_behavior_impact.ipynb
- Source: src/basket_impact.py (functions for pre-post comparison, category expansion)
- Figures: basket_diversity_comparison.html, margin_shift.html, tradeoff_scatter.html

## Module 7: Interpretability and Trust - Why We Recommended This (Day 12)

### Member 7: Explainability Analyst and Stakeholder Translator
Purpose: Make the black-box utility function transparent, building trust with business stakeholders.

### Focus Files
- Module 6 outputs
- Candidate set and top-5 recommendations
- user_context_features.parquet, if available

### Step 7.1: Global Component Importance
- Using the already computed permutation importance from Module 6 (if available) or a dedicated analysis, fit a random forest classifier to predict whether a recommended item was purchased, using Relevance, Uplift, Margin, Context as features.
- Extract permutation importances.
- Show that Uplift and Margin are not just nice-to-have but statistically significant contributors to purchase likelihood.

### Step 7.2: Per-Recommendation Explanation (LIME-style)
- For a sample of households, generate local explanations.
- Since the utility function is a linear combination, the explanation is just the weighted component values.
- Create a Why this recommendation? card for the top item:
  - Relevance: high because you often buy Pasta and 78% of similar users bought this Cheese.
  - Uplift: you only buy this 5% of trips -> strong discovery opportunity.
  - Margin: this item has a 45% margin vs. 20% for alternatives.
  - Context: current campaign makes this extra attractive for deal-sensitive shoppers.

### Step 7.3: Counterfactual Explanations
- For a mis-recommended item (recommended but not purchased), show how small changes in habits (e.g., if the user had a bit more basket diversity) would have changed the ranking.
- This demonstrates the model's sensitivity to behavior and can inform marketing nudges.

### Step 7.4: Sensitivity to Weights
- Visualize how the Top-5 list for a fixed household changes when you tweak w2 (Uplift weight) or w3 (Margin weight).
- This reassures stakeholders that the model does not flip wildly; it is robust.

### Deliverables
- Notebook: 07_interpretability_explanation.ipynb
- Source: src/recommendation_explainer.py
- Figures: global_importance_bar.html, example_explanation_cards.png, weight_sensitivity_slope.html

## Module 8: Customer-Centric Personalization - The Archetype Lens (Day 13)

### Member 8: Segmentation Analyst and Personalization Lead
Purpose: Segment customers into behavioral archetypes and show how the recommender adapts its logic per archetype.

### Focus Files
- Module 6 outputs
- user_context_features.parquet
- household-level history features

### Step 8.1: Archetype Definition
- Based on features already computed: deal_sensitivity, basket_diversity, and maybe recency from the context layer.
- Use a simple rule-based personification (no complex clustering) so it is easily interpretable:
  - Routine Replenisher - low diversity, low deal sensitivity (just get the usual stuff).
  - Deal-Driven Explorer - high deal sensitivity, moderate diversity (needs coupons to try new things).
  - Premium Discoverer - low deal sensitivity, high diversity (wants new, high-quality items).
  - Frugal Loyalist - high deal sensitivity, low diversity (loyal but price-conscious).

### Step 8.2: Per-Archetype Recommendation Strategy
- For each archetype, compute the average utility composition of the Top-5 recommendations (relevance, uplift, margin, context) and the source mix (ALS vs MBA).
- Show that the recommender predominantly uses MBA for Routine Replenishers (familiar complements) and ALS for Premium Discoverers (adjacent categories).

### Step 8.3: Archetype-Specific Performance
- Calculate Incremental Precision@5 and Average Recommended Margin broken down by archetype.
- Highlight which group benefits most from the utility function (likely Premium Discoverers and Deal-Driven Explorers).

### Step 8.4: Case Study per Archetype
- Present a detailed walk-through for one household from each archetype, similar to the John Smith case but now showing how the recommender's logic is tailored.

### Deliverables
- Notebook: 08_customer_archetypes.ipynb
- Source: src/archetypes.py
- Figures: archetype_radar.html, archetype_performance_bar.html, archetype_case_profiles.pdf

## Module 9: Policy Evaluation - Chimera vs. Popularity Baseline (Day 14)

### Member 9: Experimentation Analyst and Decision Scientist
Purpose: Evaluate two real recommendation policies on observed recommendation composition, then translate the result into a budget targeting rule for production planning.

### Focus Files
- Module 4 holdout data
- Module 6 margin-shift and recommendation outputs
- Module 8 archetype assignments for balance checks and stratification

### Step 9.1: Experimental Design
- Unit of randomization: household_key.
- Control: Popularity Baseline built from the training window.
- Treatment: Chimera top-5 recommendations.
- Primary metric: average normalized margin of the recommended slate per household.
- Guardrails: preserve a clean 50/50 assignment and report balance across archetypes.

### Step 9.2: Statistical Evaluation
- Compare control and treatment using Welch's t-test.
- Estimate lift with a bootstrapped 95% confidence interval.
- Report effect size with Cohen's d and keep the interpretation explicit.

### Step 9.3: Power Analysis
- Estimate the minimum detectable effect from the observed variance.
- Report what effect sizes are detectable at 80% power with the current sample.
- Use the result as a practical guide for interpreting the policy comparison.

### Step 9.4: Budget Allocation Optimization
- Rank households by incremental margin delta and target the top 20%.
- Compare targeted vs. random allocation to show where the policy creates the most value.
- Save the ranked list for downstream campaign planning.

### Step 9.5: Decision Summary
- Present the final recommendation in business terms, with a clear caveat that this measures recommendation composition rather than actual purchases.

### Deliverables
- Notebook: 09_simulated_ab_test_budget.ipynb
- Source: src/ab_test_simulation.py, src/budget_allocation.py
- Data: module9_ab_test_results.csv, module9_optimal_targeting_top20pct.csv, ab_assignment_mapping.csv
- Figures: policy_eval_lift_bar.html, policy_eval_cumulative_gain.html, policy_eval_archetype_impact.html

## Module 10: Production Deployment Roadmap and Executive Dashboard Wireframe (Day 15)

### Member 10: MLOps Lead and Executive Reporting Designer
Purpose: Outline a concrete production implementation and design the monitoring dashboard to show live recommendation performance.

### Focus Files
- Module 6-9 outputs
- Existing recommendation artifacts
- Deployment configuration inputs

### Step 10.1: System Architecture Blueprint
- Diagram of the recommendation pipeline in production:
  - Batch layer (weekly retraining of ALS, MBA rules, utility components).
  - Serving layer (API that takes user’s recent basket and returns Top-5).
  - Monitoring layer (tracking recommendation uptake, margin, data drift).
- Discuss technology choices (e.g., FastAPI, Redis cache, Airflow for scheduling).

### Step 10.2: Monitoring Dashboard Wireframe (Power BI layout)
- Screen 1: Business Health - Global metrics: total incremental margin vs. baseline, average basket diversity lift, active users.
- Screen 2: Micro-View - Select a household, see their archetype, recent basket, recommended items, and clickthrough rate.
- Screen 3: Explainability - For a chosen recommendation, display the utility decomposition (stacked bar) and the Why this item? text.
- Screen 4: Model Health - Over time: Precision@5, average recommended margin, data drift indicators (distribution of deal_sensitivity, Relevance scores).

### Step 10.3: Re-training and Rollback Strategy
- Define criteria for re-training (e.g., when the average deal_sensitivity in the population shifts by > 5% from baseline).
- Specify how an A/B test would be used to validate any model update before full roll-out (or rollback).

### Step 10.4: Final Uniqueness Summary
- A one-page comparison table contrasting standard recsys vs. Chimera (from the production perspective).
- This cements the project's unique value proposition.

### Deliverables
- Notebook: 10_production_roadmap.ipynb (largely markdown with architecture diagrams)
- Source: src/deployment_plan.py (possibly a config file or JSON schema for the API)
- Figures: system_architecture.html, dashboard_wireframe.pdf, uniqueness_table.html

## Revised Repository Structure

The extended project keeps the original core layout and adds the new analysis modules below.

```text
chimera-utility-recsys/
├── README.md
├── requirements.txt
│
├── data/
│   ├── 01_raw/
│   └── 02_processed/
│       ├── association_rules.csv
│       ├── candidate_set_module2.csv
│       ├── candidate_set_module3_scored.csv
│       ├── commodity_margin.csv
│       ├── filtered_items_log.csv
│       ├── item_factors.npz
│       ├── master_transactions_all.parquet
│       ├── master_transactions_organic.parquet
│       ├── module4_ablation_summary.csv
│       ├── module5_case_study_comparison.csv
│       ├── module5_recommendation_simulator.csv
│       ├── top5_recommendations_module3.csv
│       ├── user_context_features.parquet
│       ├── user_factors.npz
│       └── user_item_matrix.npz
│
├── notebooks/
│   ├── 01_ground_truth_data_model.ipynb
│   ├── 02_candidate_generation_mba_als.ipynb
│   ├── 03_utility_function_ranking.ipynb
│   ├── 04_ablation_study_validation.ipynb
│   ├── 05_cold_start_rules.ipynb
│   ├── 06_basket_behavior_impact.ipynb
│   ├── 07_interpretability_explanation.ipynb
│   ├── 08_customer_archetypes.ipynb
│   ├── 09_simulated_ab_test_budget.ipynb
│   └── 10_production_roadmap.ipynb
│
├── reports/
│   └── figures/
│       ├── ab_cumulative_profit.html
│       ├── ablation_lift_chart.html
│       ├── ablation_precision_distribution.html
│       ├── ablation_proof_screen.html
│       ├── ablation_relative_lift.html
│       ├── archetype_case_profiles.pdf
│       ├── archetype_performance_bar.html
│       ├── archetype_radar.html
│       ├── basket_diversity_comparison.html
│       ├── confidence_interval_lift.html
│       ├── dashboard_wireframe.pdf
│       ├── example_explanation_cards.png
│       ├── global_importance_bar.html
│       ├── john_smith_case_study.html
│       ├── margin_shift.html
│       ├── network_graph_mba.html
│       ├── policy_eval_archetype_impact.html
│       ├── policy_eval_cumulative_gain.html
│       ├── policy_eval_lift_bar.html
│       ├── system_architecture.html
│       ├── tradeoff_scatter.html
│       ├── uniqueness_table.html
│       ├── utility_decomposition.html
│       └── weight_sensitivity_slope.html
│
├── src/
│   ├── __init__.py
│   ├── ab_test_simulation.py
│   ├── archetypes.py
│   ├── basket_impact.py
│   ├── budget_allocation.py
│   ├── cold_start.py
│   ├── data_loader.py
│   ├── deployment_plan.py
│   ├── financial_utils.py
│   ├── recall_engine.py
│   ├── recommendation_explainer.py
│   └── utility_scorer.py
│
└── tests/
    ├── test_normalization.py
    ├── test_utility_function.py
    └── test_recommendation_explainer.py
```

This extension keeps every analysis directly relevant to the recommendation system, while adding validation, transparency, personalization, and business readiness. The final report now contains a complete narrative from design to production, with rigorous evidence at each step.



