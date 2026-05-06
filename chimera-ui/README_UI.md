# 🌌 Project Chimera: AI-Driven Decision Support System

Welcome to the **Project Chimera UI**, a high-fidelity Streamlit dashboard designed for executive oversight and granular marketing strategy optimization. This system bridges the gap between complex recommendation models and actionable business decisions.

---

## 🚀 Quick Start Guide

### 1. Prerequisites
Ensure you have **Python 3.9+** installed on your system.

### 2. Installation
Open your terminal (PowerShell, CMD, or Bash) and follow these steps:

```powershell
# Navigate to the project root
cd "path/to/project_ddm_complete_journey"

# Create a virtual environment (recommended)
python -m venv .venv

# Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On Mac/Linux:
source .venv/bin/activate

# Install the required dependencies
pip install -r chimera-ui/requirements_ui.txt
```

### 3. Data Setup
The UI relies on pre-processed analytical files. Ensure the following files exist in `data/02_processed/`:
*   `top5_recommendations_module3.csv` (Core recommendations)
*   `module8_archetype_assignments.csv` (Customer segments)
*   `candidate_set_module3_scored.csv` (Full scoring landscape)
*   `module4_ablation_summary.csv` (Model performance metrics)

### 4. Running the Dashboard
Execute the following command from the project root:

```powershell
streamlit run chimera-ui/app.py
```
The application will automatically open in your default browser at `http://localhost:8501`.

---

## 📖 Comprehensive Usage Guide

### 🏟 01. Executive Dashboard
*   **Purpose**: High-level overview of campaign health and ROI.
*   **Key Action**: Monitor **Cumulative Gain** to see what percentage of total margin is captured by your top recommendations.
*   **Insight**: Check the **Candidate Source Mix** to ensure a healthy balance between model-driven and heuristic recommendations.

### 🔍 02. Household Explorer
*   **Purpose**: Deep-dive into specific shopper profiles.
*   **Usage**: Use the search box to find a **Household ID**.
*   **Explainability**: Expand the "Why?" section on any card to see exactly how Relevance, Uplift, and Margin contributed to that specific recommendation.
*   **Action**: Use the **Dynamic Coupon Generator** to simulate how a discount shifts an item's utility. Click 📌 to stage items for your final campaign.

### 🎯 03. Archetype Lens
*   **Purpose**: Segment-level strategy analysis.
*   **Usage**: Select an archetype (e.g., *Frugal Loyalist*) to see their top-performing items and their observed margin shift over time.
*   **Insight**: Identify which segments are most sensitive to margin-heavy vs. relevance-heavy strategies.

### ⚖ 04. Weight Simulator
*   **Purpose**: Real-time strategy tuning.
*   **Usage**: Adjust the four component sliders (Relevance, Uplift, Margin, Context).
*   **Reactivity**: The **Bump Chart** will instantly show you how your favorite items "shift" in rank as you change weights.
*   **Stability**: Monitor the **Stability Score** to ensure your strategy changes aren't causing erratic ranking shifts.

### 📉 05. Counterfactual Explorer
*   **Purpose**: "What-if" scenario testing.
*   **Usage**: Select an item that *didn't* make the top-5.
*   **Analysis**: See the **Target Threshold** needed for that item to enter the top-5.
*   **Nudge**: Use the discount slider to see if a coupon is enough to push that item into the target rank.

### 🧪 06. Policy Evaluation
*   **Purpose**: A/B test results and uplift analysis.
*   **Usage**: Compare treatment vs. control groups across different household archetypes.
*   **Insight**: Validate if the "High Margin" policy actually delivered higher incremental value compared to the baseline.

### 🏥 07. Model Health
*   **Purpose**: Monitoring and Technical Oversight.
*   **Usage**: Check **System Pulse** for data freshness and **Data Lineage** to verify all required CSV/Parquet files are present.
*   **Metric**: Review **Ablation Results** to see how adding each component (Uplift, Context) improved precision.

---

## 🎨 Design Principles
Project Chimera uses a **Dark-UI High-Contrast** aesthetic:
*   **White (#FFFFFF)**: Primary text and critical titles.
*   **Light Grey (#D1D5DB)**: Labels, captions, and secondary information.
*   **Blue/Green/Purple**: Strategy components (Relevance, Margin, Context).

---

## 🛠 Troubleshooting
*   **Blank Charts?**: Ensure `Plotly` is installed and Javascript is enabled in your browser.
*   **Data Missing?**: Check the **Model Health** page; it will show a red ❌ next to any missing files.
*   **Layout Issues?**: This UI is optimized for Chrome and Edge. Ensure your browser window is at least 1280px wide.

---
*Developed for Project Chimera — Bridging AI and Action.*
