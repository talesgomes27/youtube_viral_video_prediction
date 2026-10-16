"""Testes unitários para métricas (RMSLE, PR-AUC, Duan) e hipóteses estatísticas H1 a H4."""

import numpy as np
import pandas as pd
import pytest

from youtube_viral.evaluation import (
    compute_classification_metrics,
    compute_regression_metrics,
    run_hypothesis_tests,
)


def test_regression_metrics_accuracy_and_duan_smearing() -> None:
    """Predição perfeita em log1p(views) deve produzir RMSLE=0, MAE_ln=0, R2_ln=1 e MdAPE=0."""
    y_true = np.array([10.0, 12.0, 14.0, 16.0])
    m_perfect = compute_regression_metrics(y_true, y_true)
    assert m_perfect["rmsle"] == pytest.approx(0.0, abs=1e-9)
    assert m_perfect["mae_ln"] == pytest.approx(0.0, abs=1e-9)
    assert m_perfect["r2_ln"] == pytest.approx(1.0, abs=1e-9)
    assert m_perfect["mdape_pct"] == pytest.approx(0.0, abs=1e-6)
    assert m_perfect["duan_smearing"] == pytest.approx(1.0, abs=1e-9)


def test_classification_metrics_perfect_and_p80_threshold() -> None:
    """Escores separados no top 20% devem atingir PR-AUC=1.0, ROC-AUC=1.0 e F1@P80=1.0."""
    y_true = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
    y_score = np.array([0.05, 0.10, 0.12, 0.15, 0.91, 0.08, 0.11, 0.14, 0.18, 0.95])
    m = compute_classification_metrics(y_true, y_score)
    assert m["pr_auc"] == pytest.approx(1.0, abs=1e-6)
    assert m["roc_auc"] == pytest.approx(1.0, abs=1e-6)
    assert m["f1_at_p80"] == pytest.approx(1.0, abs=1e-6)
    assert 0.0 <= m["brier_score"] < 0.05


def test_statistical_hypotheses_h1_to_h4_on_real_dataset(raw_df: pd.DataFrame) -> None:
    """Valida empiricamente as hipóteses H1, H2 (Simpson), H3 (Alavancas) e H4 (Timing)."""
    results = run_hypothesis_tests(raw_df)

    # H1: Dominância de log_subscriber_count sobre log1p_views (r > 0.80)
    assert results["H1_channel_dominance"]["r_log_subs_views"] > 0.80
    assert results["H1_channel_dominance"]["r2_log_subs_alone"] > 0.65

    # H2: Paradoxo de Simpson na duração (marginal r < -0.18, mas desconfundido |r| < 0.02)
    h2 = results["H2_simpson_duration"]
    assert h2["r_marginal_log_views"] < -0.18
    assert h2["p_marginal_log_views"] < 1e-10
    assert abs(h2["r_deconfounded_views_per_sub"]) < 0.02
    assert h2["p_deconfounded_views_per_sub"] > 0.05

    # H3: Rosto na thumbnail e tags têm efeito positivo significativo (p < 0.001)
    h3 = results["H3_creator_levers"]
    assert h3["face_viral_rate_1"] > h3["face_viral_rate_0"]
    assert h3["face_mannwhitney_p"] < 1e-10
    assert h3["tags_r_deconfounded"] > 0.04
    assert h3["tags_p_deconfounded"] < 1e-5

    # H4: Neutralidade de upload_hour e upload_day no alcance relativo (p > 0.05)
    h4 = results["H4_timing_neutrality"]
    assert h4["hour_kruskal_p"] > 0.05
    assert h4["day_kruskal_p"] > 0.05
    assert h4["day_chi2_viral_p"] > 0.05
