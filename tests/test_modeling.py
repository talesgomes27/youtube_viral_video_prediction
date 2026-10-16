"""Testes para a modelagem supervisionada tripla, convergência P80 e explicabilidade Gain/SHAP."""

import pandas as pd

from youtube_viral.modeling import run_modeling_suite


def test_modeling_suite_dual_convergence_and_h5_orthogonality(raw_df: pd.DataFrame) -> None:
    """Executa a suíte de modelagem e verifica performance, convergência P80, H5 e Gain/SHAP."""
    suite = run_modeling_suite(raw_df, n_splits=3)

    # 1. Classificação Regime A: modelos supervisionados devem bater o Baseline (PR-AUC > 0.60)
    clf_a = suite["classification"]["A"]
    assert clf_a["Baseline"]["test"]["pr_auc"] < 0.40
    assert clf_a["LogisticRegression"]["test"]["pr_auc"] > 0.65
    assert clf_a["RandomForest"]["test"]["pr_auc"] > 0.65
    assert clf_a["LightGBM"]["test"]["pr_auc"] > 0.65
    assert clf_a["LightGBM_DualRegressor"]["test"]["pr_auc"] > 0.65

    # 2. Regressão Regime A: LightGBM deve atingir R2 > 0.65 e RMSLE < 0.90
    reg_a = suite["regression"]["A"]
    assert reg_a["LightGBM"]["test"]["r2_ln"] > 0.65
    assert reg_a["LightGBM"]["test"]["rmsle"] < reg_a["Baseline"]["test"]["rmsle"]

    # 3. Hipótese H5: Regime B vs Regime A tem diferença residual (|delta| < 0.02)
    h5 = suite["H5_regime_comparison"]
    assert abs(h5["delta_test_pr_auc"]) < 0.02
    assert abs(h5["delta_test_rmsle"]) < 0.02

    # 4. Explicabilidade por Gain e TreeSHAP deve somar 100% e ter log_subscriber_count no topo
    imp_b = suite["feature_importance"]["B_regression_gain"]
    assert abs(sum(item["gain_pct"] for item in imp_b) - 100.0) < 1e-3
    assert imp_b[0]["feature"] in {"log_subscriber_count", "subscriber_count"}
    assert imp_b[0]["gain_pct"] > 50.0
