"""Module 10 helpers for the production roadmap and executive dashboard wireframe."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import pandas as pd


DEFAULT_DEPLOYMENT_CONFIG: Dict[str, object] = {
    "batch_retrain_frequency": "weekly",
    "training_trigger": "deal_sensitivity shift > 5% or Precision@5 drop > 3%",
    "serving_api": "FastAPI",
    "cache_layer": "Redis",
    "scheduler": "Airflow",
    "model_registry": "MLflow",
    "dashboard_stack": "Power BI",
    "data_quality_stack": "Great Expectations",
    "drift_monitoring_stack": "Prometheus + pandas profiling",
    "rollback_policy": "rollback to the previous approved artifact when health gates fail",
    "owner": "MLOps lead",
}


@dataclass
class DeploymentRoadmapArtifacts:
    """Container for module 10 planning tables and export-ready assets."""

    config: Dict[str, object]
    architecture_table: pd.DataFrame
    dashboard_wireframe_table: pd.DataFrame
    retraining_policy_table: pd.DataFrame
    uniqueness_table: pd.DataFrame


def _active_config(config: Optional[Mapping[str, object]] = None) -> Dict[str, object]:
    active = dict(DEFAULT_DEPLOYMENT_CONFIG)
    if config:
        active.update(config)
    return active


def build_deployment_config(config: Optional[Mapping[str, object]] = None) -> Dict[str, object]:
    return _active_config(config)


def build_system_architecture_table(config: Optional[Mapping[str, object]] = None) -> pd.DataFrame:
    active = _active_config(config)
    rows = [
        {
            "Layer": "Batch learning",
            "Responsibility": "Refresh ALS factors, MBA rules, utility scores, and candidate snapshots.",
            "Inputs": "Weekly transaction extracts, campaign tables, commodity margins, holdout metrics.",
            "Outputs": "Versioned recommendation artifacts and validation summaries.",
            "Suggested stack": f"{active['scheduler']} + pandas + implicit + mlxtend",
            "Monitoring signal": "Training freshness, data completeness, offline Precision@5",
        },
        {
            "Layer": "Serving API",
            "Responsibility": "Return Top-5 recommendations for the latest household basket in milliseconds.",
            "Inputs": "Household key, recent basket items, cached feature lookups, current campaign context.",
            "Outputs": "Ranked recommendation payload with utility decomposition.",
            "Suggested stack": f"{active['serving_api']} + {active['cache_layer']} + JSON schema",
            "Monitoring signal": "Latency, cache hit rate, clickthrough rate",
        },
        {
            "Layer": "Monitoring",
            "Responsibility": "Track uptake, margin lift, drift, and rollback gates.",
            "Inputs": "Recommendation logs, clickstream events, realized margin, feature distributions.",
            "Outputs": "Health dashboards and alert thresholds.",
            "Suggested stack": f"{active['dashboard_stack']} + Prometheus + {active['data_quality_stack']}",
            "Monitoring signal": "Precision@5, margin lift, deal_sensitivity drift",
        },
        {
            "Layer": "Governance",
            "Responsibility": "Version, approve, and roll back releases before full rollout.",
            "Inputs": "Model registry versions, validation reports, A/B test results.",
            "Outputs": "Approved release notes and rollback decisions.",
            "Suggested stack": f"{active['model_registry']} + approval checklist",
            "Monitoring signal": "Release audit trail, approval status",
        },
    ]
    return pd.DataFrame(rows)


def build_dashboard_wireframe_table() -> pd.DataFrame:
    rows = [
        {
            "Screen": "Business Health",
            "Audience": "Executives",
            "Core visuals": "Trend cards, baseline comparison, active-user KPI strip.",
            "Primary metrics": "Total incremental margin, basket diversity lift, active users.",
            "Business question": "Is Chimera generating more money than the baseline?",
        },
        {
            "Screen": "Micro-View",
            "Audience": "Category managers",
            "Core visuals": "Household profile, recent basket, recommendation cards.",
            "Primary metrics": "Archetype, clickthrough rate, top pick, utility score.",
            "Business question": "What should we show this household right now?",
        },
        {
            "Screen": "Explainability",
            "Audience": "Merchandising and analytics",
            "Core visuals": "Stacked utility bar and natural-language why-this-item card.",
            "Primary metrics": "Relevance, uplift, margin, context contributions.",
            "Business question": "Why did the model choose this item?",
        },
        {
            "Screen": "Model Health",
            "Audience": "MLOps and analytics",
            "Core visuals": "Precision@5 line, average margin line, drift indicators.",
            "Primary metrics": "Precision@5, average recommended margin, deal_sensitivity drift.",
            "Business question": "When do we retrain or roll back?",
        },
    ]
    return pd.DataFrame(rows)


def build_retraining_policy_table(config: Optional[Mapping[str, object]] = None) -> pd.DataFrame:
    active = _active_config(config)
    rows = [
        {
            "Trigger": "Population behavior shift",
            "Threshold": "deal_sensitivity changes by more than 5% from baseline",
            "Action": f"Schedule a fresh batch run through {active['scheduler']}.",
            "Owner": active["owner"],
        },
        {
            "Trigger": "Offline quality regression",
            "Threshold": "Precision@5 falls by more than 3% on the validation window",
            "Action": "Hold release and review candidate generation, scoring, and feature inputs.",
            "Owner": "Analytics lead",
        },
        {
            "Trigger": "Campaign or promotion drift",
            "Threshold": "current promotion mix differs materially from the prior training window",
            "Action": "Refresh campaign flags and promoted commodity lookups before serving.",
            "Owner": "Data engineering",
        },
        {
            "Trigger": "Health gate failure",
            "Threshold": "latency, clickthrough, or margin lift violates agreed guardrails",
            "Action": str(active["rollback_policy"]),
            "Owner": active["owner"],
        },
    ]
    return pd.DataFrame(rows)


def build_uniqueness_table() -> pd.DataFrame:
    rows = [
        {
            "Dimension": "Objective",
            "Standard recsys": "Maximize proxy relevance or predicted click probability.",
            "Chimera": "Maximize incremental retailer margin under utility constraints.",
        },
        {
            "Dimension": "Scoring logic",
            "Standard recsys": "Single ranking score, often opaque to stakeholders.",
            "Chimera": "Weighted sum of relevance, uplift, margin, and context.",
        },
        {
            "Dimension": "Business control",
            "Standard recsys": "Recommendation quality is discussed after deployment.",
            "Chimera": "Margin, promotion context, and habit strength are explicit levers.",
        },
        {
            "Dimension": "Validation",
            "Standard recsys": "Offline accuracy is the main proof point.",
            "Chimera": "Incremental Precision@5, ablation, and simulated A/B lift.",
        },
        {
            "Dimension": "Operations",
            "Standard recsys": "Model retraining is sometimes ad hoc.",
            "Chimera": "Weekly retraining, drift thresholds, and rollback gates are defined.",
        },
    ]
    return pd.DataFrame(rows)


def build_deployment_roadmap(config: Optional[Mapping[str, object]] = None) -> DeploymentRoadmapArtifacts:
    active = _active_config(config)
    return DeploymentRoadmapArtifacts(
        config=active,
        architecture_table=build_system_architecture_table(active),
        dashboard_wireframe_table=build_dashboard_wireframe_table(),
        retraining_policy_table=build_retraining_policy_table(active),
        uniqueness_table=build_uniqueness_table(),
    )


def _dataframe_to_html_section(title: str, description: str, frame: pd.DataFrame) -> str:
    return "\n".join(
        [
            "<section class='panel'>",
            f"<h2>{escape(title)}</h2>",
            f"<p>{escape(description)}</p>",
            frame.to_html(index=False, classes="table", border=0, escape=False),
            "</section>",
        ]
    )


def render_html_report(title: str, subtitle: str, sections: Iterable[str]) -> str:
    section_html = "\n".join(sections)
    return f"""<!DOCTYPE html>
<html lang='en'>
<head>
  <meta charset='utf-8' />
  <meta name='viewport' content='width=device-width, initial-scale=1' />
  <title>{escape(title)}</title>
  <style>
    :root {{
      --bg: #f5f1e8;
      --panel: #ffffff;
      --ink: #1f2937;
      --accent: #1f6f78;
      --accent-soft: #d6ebe8;
      --line: #d9d4c7;
    }}
    body {{
      margin: 0;
      padding: 32px;
      font-family: Arial, Helvetica, sans-serif;
      background: linear-gradient(180deg, #fbf8f1 0%, var(--bg) 100%);
      color: var(--ink);
    }}
    .page {{ max-width: 1180px; margin: 0 auto; }}
    header {{
      background: linear-gradient(135deg, #163945 0%, #1f6f78 60%, #4b8f8c 100%);
      color: white;
      padding: 28px 32px;
      border-radius: 24px;
      box-shadow: 0 14px 36px rgba(31, 111, 120, 0.18);
    }}
    header h1 {{ margin: 0 0 10px; font-size: 2.2rem; }}
    header p {{ margin: 0; max-width: 900px; line-height: 1.55; }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 20px;
      margin-top: 24px;
      padding: 22px 24px;
      box-shadow: 0 10px 26px rgba(31, 41, 55, 0.06);
    }}
    .panel h2 {{ margin: 0 0 8px; color: var(--accent); }}
    .panel p {{ margin: 0 0 16px; line-height: 1.6; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 12px 10px;
      text-align: left;
      vertical-align: top;
      font-size: 0.95rem;
    }}
    th {{ background: var(--accent-soft); color: #12363b; }}
    tr:last-child td {{ border-bottom: none; }}
  </style>
</head>
<body>
  <div class='page'>
    <header>
      <h1>{escape(title)}</h1>
      <p>{escape(subtitle)}</p>
    </header>
    {section_html}
  </div>
</body>
</html>"""


def save_html_report(path: Path, title: str, subtitle: str, sections: Iterable[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html_report(title=title, subtitle=subtitle, sections=sections), encoding="utf-8")
    return path


def save_dashboard_wireframe_pdf(path: Path, dashboard_wireframe_table: pd.DataFrame) -> Path:
    from matplotlib import pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(15, 10), facecolor="#f5f1e8")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    fig.text(0.05, 0.94, "Module 10 Dashboard Wireframe", fontsize=24, weight="bold", color="#163945")
    fig.text(
        0.05,
        0.905,
        "Four screens designed for executive, operational, and MLOps audiences.",
        fontsize=11,
        color="#4b5563",
    )

    positions = [
        (0.05, 0.53),
        (0.53, 0.53),
        (0.05, 0.08),
        (0.53, 0.08),
    ]
    card_width = 0.42
    card_height = 0.34
    rows = dashboard_wireframe_table.to_dict("records")

    for index, row in enumerate(rows):
        x, y = positions[index]
        card = FancyBboxPatch(
            (x, y),
            card_width,
            card_height,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.5,
            edgecolor="#7aa6a1",
            facecolor="#ffffff",
        )
        ax.add_patch(card)
        fig.text(x + 0.02, y + card_height - 0.05, row["Screen"], fontsize=16, weight="bold", color="#1f6f78")
        fig.text(x + 0.02, y + card_height - 0.09, f"Audience: {row['Audience']}", fontsize=10.5, color="#374151")
        fig.text(x + 0.02, y + card_height - 0.14, "Core visuals", fontsize=11, weight="bold", color="#163945")
        fig.text(x + 0.02, y + card_height - 0.18, row["Core visuals"], fontsize=10, color="#111827")
        fig.text(x + 0.02, y + card_height - 0.24, "Primary metrics", fontsize=11, weight="bold", color="#163945")
        fig.text(x + 0.02, y + card_height - 0.28, row["Primary metrics"], fontsize=10, color="#111827")
        fig.text(x + 0.02, y + 0.03, row["Business question"], fontsize=10, style="italic", color="#6b7280")

    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return path


def export_deployment_roadmap(artifacts: DeploymentRoadmapArtifacts, output_dir: Path) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    architecture_path = save_html_report(
        output_dir / "system_architecture.html",
        "System Architecture Blueprint",
        "Batch, serving, monitoring, and governance work together to keep Chimera accurate and accountable.",
        [
            _dataframe_to_html_section(
                "Production pipeline",
                "A concise view of the end-to-end production flow.",
                artifacts.architecture_table,
            ),
        ],
    )
    wireframe_path = save_dashboard_wireframe_pdf(output_dir / "dashboard_wireframe.pdf", artifacts.dashboard_wireframe_table)
    uniqueness_path = save_html_report(
        output_dir / "uniqueness_table.html",
        "Chimera vs. Standard Recommendation Systems",
        "A one-page comparison that frames the project as a production decision system rather than a generic recommender.",
        [
            _dataframe_to_html_section(
                "Uniqueness summary",
                "How Chimera changes the objective, validation, and operating model.",
                artifacts.uniqueness_table,
            ),
        ],
    )
    return {
        "system_architecture": architecture_path,
        "dashboard_wireframe": wireframe_path,
        "uniqueness_table": uniqueness_path,
    }


__all__ = [
    "DEFAULT_DEPLOYMENT_CONFIG",
    "DeploymentRoadmapArtifacts",
    "build_dashboard_wireframe_table",
    "build_deployment_config",
    "build_deployment_roadmap",
    "build_retraining_policy_table",
    "build_system_architecture_table",
    "build_uniqueness_table",
    "export_deployment_roadmap",
    "render_html_report",
    "save_dashboard_wireframe_pdf",
    "save_html_report",
]
