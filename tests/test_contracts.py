"""Testes unitários e contratuais para DC-01 a DC-07 e bloqueio de leakage."""

import pandas as pd
import pytest

from youtube_viral.contracts import (
    EXPECTED_CATEGORIES,
    EXPECTED_UPLOAD_DAYS,
    VIRAL_VIEWS_CUTOFF,
    assert_no_leakage,
    validate_raw_dataset,
)


def test_real_csv_satisfies_all_data_contracts(raw_df: pd.DataFrame) -> None:
    """O arquivo viral_videos.csv deve cumprir integralmente os contratos DC-01 a DC-06."""
    validated = validate_raw_dataset(raw_df)
    assert validated.shape == (12000, 15)
    assert len(set(validated["category"])) == len(EXPECTED_CATEGORIES)
    assert set(validated["upload_day"]) == EXPECTED_UPLOAD_DAYS
    assert validated["viral"].mean() == pytest.approx(0.20, abs=1e-6)
    max_non_viral = int(validated.loc[validated["viral"] == 0, "views"].max())
    min_viral = int(validated.loc[validated["viral"] == 1, "views"].min())
    assert max_non_viral < VIRAL_VIEWS_CUTOFF <= min_viral


@pytest.mark.parametrize(
    ("mutator", "expected_msg"),
    [
        (lambda df: df.iloc[:-1], "DC-01"),
        (lambda df: df.assign(video_id="vid_000001"), "DC-01"),
        (lambda df: df.assign(title_length=None), "DC-02"),
        (lambda df: df.assign(category="InvalidCategory"), "DC-03"),
        (lambda df: df.assign(upload_hour=25), "DC-04"),
        (lambda df: df.assign(likes=df["views"] + 10), "DC-05"),
        (lambda df: df.assign(viral=1), "DC-06"),
    ],
)
def test_corrupted_dataframe_violates_specific_contract(
    raw_df: pd.DataFrame,
    mutator,
    expected_msg: str,
) -> None:
    """Violações em qualquer regra DC-01 a DC-06 devem levantar ValueError explícito."""
    corrupted = mutator(raw_df.copy())
    with pytest.raises(ValueError, match=expected_msg):
        validate_raw_dataset(corrupted)


@pytest.mark.parametrize(
    "forbidden_col",
    [
        "video_id",
        "views",
        "viral",
        "likes",
        "comments",
        "shares",
        "log1p_views",
        "views_per_sub",
        "log_views_per_sub",
    ],
)
def test_global_leakage_columns_blocked_in_both_regimes(forbidden_col: str) -> None:
    """Colunas de target ou proxy leakage bruto devem falhar em ambos os Regimes A e B (DC-07)."""
    features = ["duration_sec", "log_subscriber_count", forbidden_col]
    for regime in ("A", "B"):
        with pytest.raises(ValueError, match="DC-07"):
            assert_no_leakage(features, regime=regime)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "early_signal_col",
    [
        "like_rate",
        "comment_rate",
        "share_rate",
        "engagement_rate_total",
        "comments_per_like",
        "shares_per_like",
    ],
)
def test_early_engagement_rates_blocked_in_regime_a_but_allowed_in_regime_b(
    early_signal_col: str,
) -> None:
    """Taxas relativas pós-upload devem ser bloqueadas no Regime A (Cold Start) e aceitas no B."""
    features = ["duration_sec", "log_subscriber_count", early_signal_col]
    with pytest.raises(ValueError, match="DC-07"):
        assert_no_leakage(features, regime="A")
    assert_no_leakage(features, regime="B")
