"""Orquestrador principal do pipeline analítico: executa contratos, modelos, métricas e figuras."""

import json
from pathlib import Path
from typing import Any

import pandas as pd

from youtube_viral.contracts import validate_raw_dataset
from youtube_viral.evaluation import run_hypothesis_tests
from youtube_viral.modeling import run_modeling_suite
from youtube_viral.plotting import (
    plot_fig01_distributions_and_threshold,
    plot_fig02_simpson_paradox_categories,
    plot_fig03_creator_levers_and_timing,
    plot_fig04_dual_modeling_performance,
    plot_fig05_scenario_a_vs_b_importance,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CSV = PROJECT_ROOT / "viral_videos.csv"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"


def execute_full_pipeline(
    csv_path: Path = DEFAULT_CSV,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
    n_splits: int = 5,
) -> dict[str, Any]:
    """Executa o pipeline completo e persiste artefatos em reports/metrics e reports/figures.

    Args:
        csv_path: Caminho para viral_videos.csv.
        reports_dir: Diretório raiz para salvar metrics/ e figures/.
        n_splits: Número de dobras na validação cruzada estratificada.

    Returns:
        Dicionário serializável com todos os resultados auditados.
    """
    raw_df = pd.read_csv(csv_path)
    df = validate_raw_dataset(raw_df)

    hypotheses = run_hypothesis_tests(df)
    suite = run_modeling_suite(df, n_splits=n_splits)

    figures_dir = reports_dir / "figures"
    metrics_dir = reports_dir / "metrics"
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    fig_paths = {
        "fig01": str(
            plot_fig01_distributions_and_threshold(
                df, hypotheses, figures_dir / "fig01_distributions_and_threshold.png"
            )
        ),
        "fig02": str(
            plot_fig02_simpson_paradox_categories(
                df, hypotheses, figures_dir / "fig02_simpson_paradox_categories.png"
            )
        ),
        "fig03": str(
            plot_fig03_creator_levers_and_timing(
                df, hypotheses, figures_dir / "fig03_creator_levers_and_timing.png"
            )
        ),
        "fig04": str(
            plot_fig04_dual_modeling_performance(
                suite, figures_dir / "fig04_dual_modeling_performance.png"
            )
        ),
        "fig05": str(
            plot_fig05_scenario_a_vs_b_importance(
                suite, figures_dir / "fig05_scenario_a_vs_b_importance.png"
            )
        ),
    }

    # Exporta tabela comparativa multi-métrica em CSV
    rows_csv: list[dict[str, Any]] = []
    for regime in ("A", "B"):
        for model_name, res in suite["classification"][regime].items():
            rows_csv.append(
                {
                    "task": "classification",
                    "regime": regime,
                    "model": model_name,
                    "cv_primary_mean": res["cv"]["pr_auc_mean"],
                    "cv_primary_std": res["cv"]["pr_auc_std"],
                    "test_pr_auc": res["test"]["pr_auc"],
                    "test_roc_auc": res["test"]["roc_auc"],
                    "test_f1_p80": res["test"]["f1_at_p80"],
                    "test_f1_050": res["test"]["f1_at_050"],
                    "test_brier": res["test"]["brier_score"],
                }
            )
        for model_name, res in suite["regression"][regime].items():
            rows_csv.append(
                {
                    "task": "regression",
                    "regime": regime,
                    "model": model_name,
                    "cv_primary_mean": res["cv"]["rmsle_mean"],
                    "cv_primary_std": res["cv"]["rmsle_std"],
                    "test_rmsle": res["test"]["rmsle"],
                    "test_mae_ln": res["test"]["mae_ln"],
                    "test_r2_ln": res["test"]["r2_ln"],
                    "test_mdape_pct": res["test"]["mdape_pct"],
                }
            )

    pd.DataFrame(rows_csv).to_csv(metrics_dir / "model_comparison.csv", index=False)

    summary_payload = {
        "hypotheses": hypotheses,
        "classification": suite["classification"],
        "regression": suite["regression"],
        "deconfounded_multiplier": suite["deconfounded_multiplier"],
        "H5_regime_comparison": suite["H5_regime_comparison"],
        "feature_importance": suite["feature_importance"],
        "figures": fig_paths,
    }

    with open(metrics_dir / "summary_results.json", "w", encoding="utf-8") as fp:
        json.dump(summary_payload, fp, indent=2, ensure_ascii=False)

    return summary_payload


if __name__ == "__main__":
    execute_full_pipeline()
