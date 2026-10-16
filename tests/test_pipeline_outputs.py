"""Testes de integração para o pipeline de exportação de métricas e figuras em 300 DPI."""

import json
from pathlib import Path

from PIL import Image

from youtube_viral.pipeline import DEFAULT_REPORTS_DIR


def test_pipeline_generated_all_metrics_and_300dpi_figures() -> None:
    """Confirma geração de summary_results.json, model_comparison.csv e fig01..fig05 em 300 DPI."""
    summary_path = DEFAULT_REPORTS_DIR / "metrics" / "summary_results.json"
    csv_path = DEFAULT_REPORTS_DIR / "metrics" / "model_comparison.csv"

    assert summary_path.exists(), "summary_results.json não foi gerado."
    assert csv_path.exists(), "model_comparison.csv não foi gerado."

    with open(summary_path, encoding="utf-8") as fp:
        payload = json.load(fp)

    expected_figs = [
        "fig01_distributions_and_threshold.png",
        "fig02_simpson_paradox_categories.png",
        "fig03_creator_levers_and_timing.png",
        "fig04_dual_modeling_performance.png",
        "fig05_scenario_a_vs_b_importance.png",
    ]
    for fname in expected_figs:
        fpath: Path = DEFAULT_REPORTS_DIR / "figures" / fname
        assert fpath.exists(), f"Figura ausente: {fpath}"
        assert fpath.stat().st_size > 25000
        with Image.open(fpath) as img:
            width, height = img.size
            assert width == 3600, f"Largura esperada 3600px (12in * 300dpi), obtida {width}px"
            assert height == 1560, f"Altura esperada 1560px (5.2in * 300dpi), obtida {height}px"

    assert "H1_channel_dominance" in payload["hypotheses"]
    assert "LightGBM_DualRegressor" in payload["classification"]["A"]
