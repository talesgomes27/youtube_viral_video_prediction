"""Testes unitários para o particionamento 80/20 e engenharia dos Regimes A e B."""

import numpy as np
import pandas as pd
import pytest

from youtube_viral.contracts import assert_no_leakage
from youtube_viral.features import (
    REGIME_A_COLS,
    REGIME_B_COLS,
    build_feature_matrix,
    build_targets,
    stratified_train_test_split,
)


def test_stratified_split_80_20_preserves_joint_distribution(raw_df: pd.DataFrame) -> None:
    """Partição 80/20 deve ter 9600/2400 linhas, zero sobreposição de video_id e 20% viral."""
    train_df, test_df = stratified_train_test_split(raw_df)
    assert len(train_df) == 9600
    assert len(test_df) == 2400
    assert set(train_df["video_id"]).isdisjoint(set(test_df["video_id"]))
    assert train_df["viral"].mean() == pytest.approx(0.20, abs=1e-3)
    assert test_df["viral"].mean() == pytest.approx(0.20, abs=1e-3)

    # Verifica que todas as 15 categorias mantêm proporção próxima de 80/20
    train_cat_counts = train_df["category"].value_counts()
    raw_cat_counts = raw_df["category"].value_counts()
    for cat in raw_cat_counts.index:
        ratio = train_cat_counts[cat] / raw_cat_counts[cat]
        assert ratio == pytest.approx(0.80, abs=0.01)


def test_build_regime_a_features_satisfies_anti_leakage_and_math(raw_df: pd.DataFrame) -> None:
    """Matriz do Regime A deve conter apenas variáveis pré-upload sem nulos ou infinitos."""
    X_a = build_feature_matrix(raw_df, regime="A")
    assert list(X_a.columns) == list(REGIME_A_COLS)
    assert int(X_a.isna().sum().sum()) == 0
    assert_no_leakage(X_a.columns.tolist(), regime="A")

    # Identidade trigonométrica sin^2 + cos^2 == 1
    trig_norm = X_a["hour_sin"] ** 2 + X_a["hour_cos"] ** 2
    np.testing.assert_allclose(trig_norm.to_numpy(), 1.0, atol=1e-7)
    assert set(X_a["is_weekend"].unique()) == {0, 1}


def test_build_regime_b_features_adds_relative_engagement_signals(raw_df: pd.DataFrame) -> None:
    """Matriz do Regime B deve estender o Regime A com taxas relativas válidas."""
    X_b = build_feature_matrix(raw_df, regime="B")
    assert list(X_b.columns) == list(REGIME_B_COLS)
    assert int(X_b.isna().sum().sum()) == 0
    assert_no_leakage(X_b.columns.tolist(), regime="B")

    # Taxas relativas devem estar entre 0 e 1
    for col in ("like_rate", "comment_rate", "share_rate", "engagement_rate_total"):
        assert (X_b[col] >= 0.0).all()
        assert (X_b[col] <= 1.0).all()


def test_build_targets_triple_architecture_accuracy(raw_df: pd.DataFrame) -> None:
    """Verifica cálculo exato dos 3 alvos: viral, log1p_views e log_views_per_sub."""
    targets = build_targets(raw_df)
    assert list(targets.columns) == ["viral", "log1p_views", "log_views_per_sub", "strata"]
    np.testing.assert_allclose(
        targets["log1p_views"].to_numpy(),
        np.log1p(raw_df["views"].to_numpy(dtype=float)),
        rtol=1e-7,
    )
    expected_mult = np.log(
        raw_df["views"].to_numpy(dtype=float) / raw_df["subscriber_count"].to_numpy(dtype=float)
    )
    np.testing.assert_allclose(targets["log_views_per_sub"].to_numpy(), expected_mult, rtol=1e-7)
