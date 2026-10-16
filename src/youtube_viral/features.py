"""Engenharia de atributos, particionamento estratificado e construção dos Regimes A e B."""

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from youtube_viral import SEED
from youtube_viral.contracts import assert_no_leakage, validate_raw_dataset

REGIME_A_COLS: tuple[str, ...] = (
    "category",
    "upload_day",
    "duration_sec",
    "log_duration_sec",
    "title_length",
    "thumbnail_face",
    "upload_hour",
    "hour_sin",
    "hour_cos",
    "is_weekend",
    "tags_count",
    "has_tags",
    "tags_per_title_char",
    "description_length",
    "has_description",
    "desc_to_title_ratio",
    "subscriber_count",
    "log_subscriber_count",
)

REGIME_B_EXTRA_COLS: tuple[str, ...] = (
    "like_rate",
    "comment_rate",
    "share_rate",
    "engagement_rate_total",
    "comments_per_like",
    "shares_per_like",
)

REGIME_B_COLS: tuple[str, ...] = REGIME_A_COLS + REGIME_B_EXTRA_COLS

DECONFOUND_COLS: tuple[str, ...] = tuple(
    c for c in REGIME_A_COLS if c not in {"subscriber_count", "log_subscriber_count"}
)


def stratified_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide o dataset validado em 80% treino e 20% teste estratificando por category x viral.

    Args:
        df: DataFrame bruto de viral_videos.csv.
        test_size: Fração do conjunto de teste (padrão 0.20 = 2.400 linhas).
        random_state: Semente determinística global.

    Returns:
        Tupla (train_df, test_df) com índices redefinidos.
    """
    validated = validate_raw_dataset(df)
    strata = validated["category"].astype(str) + "_" + validated["viral"].astype(str)
    train_df, test_df = train_test_split(
        validated,
        test_size=test_size,
        stratify=strata,
        random_state=random_state,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def build_feature_matrix(
    df: pd.DataFrame,
    regime: Literal["A", "B"] = "A",
    deconfound: bool = False,
) -> pd.DataFrame:
    """Constrói a matriz de atributos X para o Regime A (Cold Start) ou B (Early Signals).

    Args:
        df: DataFrame de entrada contendo as colunas brutas.
        regime: 'A' (apenas metadados pré-upload) ou 'B' (adiciona taxas relativas).
        deconfound: Se True, remove subscriber_count e log_subscriber_count para a Frente 3.

    Returns:
        DataFrame X validado por assert_no_leakage.
    """
    out = pd.DataFrame(index=df.index)
    out["category"] = df["category"].astype(str)
    out["upload_day"] = df["upload_day"].astype(str)

    dur = df["duration_sec"].astype(float)
    out["duration_sec"] = dur
    out["log_duration_sec"] = np.log1p(dur)

    title_len = df["title_length"].astype(float)
    out["title_length"] = title_len
    out["thumbnail_face"] = df["thumbnail_face"].astype(int)

    hour = df["upload_hour"].astype(float)
    out["upload_hour"] = hour
    angle = 2.0 * np.pi * hour / 24.0
    out["hour_sin"] = np.sin(angle)
    out["hour_cos"] = np.cos(angle)
    out["is_weekend"] = df["upload_day"].isin(["Sat", "Sun"]).astype(int)

    tags = df["tags_count"].astype(float)
    out["tags_count"] = tags
    out["has_tags"] = (tags > 0).astype(int)
    out["tags_per_title_char"] = tags / np.maximum(1.0, title_len)

    desc_len = df["description_length"].astype(float)
    out["description_length"] = desc_len
    out["has_description"] = (desc_len > 0).astype(int)
    out["desc_to_title_ratio"] = desc_len / np.maximum(1.0, title_len)

    if not deconfound:
        subs = df["subscriber_count"].astype(float)
        out["subscriber_count"] = subs
        out["log_subscriber_count"] = np.log1p(subs)

    if regime == "B":
        views = np.maximum(1.0, df["views"].astype(float))
        likes = df["likes"].astype(float)
        comments = df["comments"].astype(float)
        shares = df["shares"].astype(float)
        safe_likes = np.maximum(1.0, likes)

        out["like_rate"] = likes / views
        out["comment_rate"] = comments / views
        out["share_rate"] = shares / views
        out["engagement_rate_total"] = (likes + comments + shares) / views
        out["comments_per_like"] = comments / safe_likes
        out["shares_per_like"] = shares / safe_likes

    target_cols = (
        DECONFOUND_COLS
        if deconfound and regime == "A"
        else (REGIME_A_COLS if regime == "A" else REGIME_B_COLS)
    )
    out = out[list(target_cols)]
    assert_no_leakage(out.columns.tolist(), regime=regime, forbid_subscribers=deconfound)
    return out


def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Extrai os três alvos analíticos e o vetor de estratificação.

    Args:
        df: DataFrame contendo views, subscriber_count, viral e category.

    Returns:
        DataFrame com colunas ['viral', 'log1p_views', 'log_views_per_sub', 'strata'].
    """
    views = df["views"].astype(float)
    subs = df["subscriber_count"].astype(float)
    targets = pd.DataFrame(index=df.index)
    targets["viral"] = df["viral"].astype(int)
    targets["log1p_views"] = np.log1p(views)
    targets["log_views_per_sub"] = np.log(views / subs)
    targets["strata"] = df["category"].astype(str) + "_" + df["viral"].astype(str)
    return targets
