from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from src.deployment_plan import build_deployment_roadmap, export_deployment_roadmap  # noqa: E402


def test_deployment_roadmap_tables_have_expected_module_10_screens():
	artifacts = build_deployment_roadmap()
	assert list(artifacts.dashboard_wireframe_table["Screen"]) == [
		"Business Health",
		"Micro-View",
		"Explainability",
		"Model Health",
	]
	assert "FastAPI" in artifacts.architecture_table.iloc[1]["Suggested stack"]
	assert artifacts.retraining_policy_table.shape[0] >= 4
	assert artifacts.uniqueness_table.shape[0] >= 4


def test_export_deployment_roadmap_writes_expected_files(tmp_path):
	artifacts = build_deployment_roadmap()
	outputs = export_deployment_roadmap(artifacts, tmp_path)
	assert outputs["system_architecture"].exists()
	assert outputs["dashboard_wireframe"].exists()
	assert outputs["uniqueness_table"].exists()