"""Contratos de dados e barreiras anti-leakage em tempo de execução (DC-01 a DC-07)."""

from collections.abc import Sequence
from typing import Literal

import pandas as pd

EXPECTED_ROWS: int = 12000
EXPECTED_COLS: int = 15
VIRAL_VIEWS_CUTOFF: int = 734379

EXPECTED_COLUMNS: tuple[str, ...] = (
    "video_id",
    "category",
    "duration_sec",
    "title_length",
    "thumbnail_face",
    "upload_hour",
    "upload_day",
    "tags_count",
    "description_length",
    "subscriber_count",
    "views",
    "likes",
    "comments",
    "shares",
    "viral",
)

EXPECTED_CATEGORIES: frozenset[str] = frozenset(
    {
        "Beauty",
        "Comedy",
        "Cooking",
        "DIY",
        "Education",
        "Finance",
        "Fitness",
        "Gaming",
        "Music",
        "News",
        "Science",
        "Sports",
        "Tech",
        "Travel",
        "Vlog",
    }
)

EXPECTED_UPLOAD_DAYS: frozenset[str] = frozenset({"Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"})

GLOBAL_FORBIDDEN_FEATURES: frozenset[str] = frozenset(
    {
        "video_id",
        "views",
        "viral",
        "likes",
        "comments",
        "shares",
        "log_views",
        "log1p_views",
        "views_per_sub",
        "log_views_per_sub",
    }
)

REGIME_B_ONLY_FEATURES: frozenset[str] = frozenset(
    {
        "like_rate",
        "comment_rate",
        "share_rate",
        "engagement_rate_total",
        "comments_per_like",
        "shares_per_like",
    }
)


def validate_raw_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Valida as regras contratuais DC-01 a DC-06 sobre o DataFrame de entrada.

    Args:
        df: DataFrame lido a partir de viral_videos.csv.

    Returns:
        O próprio DataFrame validado caso nenhuma invariante seja violada.

    Raises:
        ValueError: Se qualquer regra contratual (DC-01 a DC-06) falhar.
    """
    if df.shape != (EXPECTED_ROWS, EXPECTED_COLS) or tuple(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            f"[DC-01] Dimensões ou colunas inválidas: esperado {(EXPECTED_ROWS, EXPECTED_COLS)}, "
            f"recebido {df.shape}."
        )

    if int(df.isna().sum().sum()) > 0:
        raise ValueError("[DC-02] Foram encontrados valores nulos/NaN no dataset.")

    valid_pattern = df["video_id"].astype(str).str.match(r"^vid_\d{6}$").all()
    if not df["video_id"].is_unique or not valid_pattern:
        raise ValueError(
            "[DC-01] Coluna video_id contém duplicatas ou formato fora de ^vid_\\d{6}$."
        )

    if set(df["category"]) != EXPECTED_CATEGORIES:
        diff_cat = set(df["category"]) ^ EXPECTED_CATEGORIES
        raise ValueError(f"[DC-03] Domínio de category inválido: {diff_cat}")

    if set(df["upload_day"]) != EXPECTED_UPLOAD_DAYS:
        diff_day = set(df["upload_day"]) ^ EXPECTED_UPLOAD_DAYS
        raise ValueError(f"[DC-03] Domínio de upload_day inválido: {diff_day}")

    if not (
        df["duration_sec"].between(30, 3600).all()
        and df["title_length"].between(1, 200).all()
        and df["thumbnail_face"].isin([0, 1]).all()
        and df["upload_hour"].between(0, 23).all()
        and (df["tags_count"] >= 0).all()
        and (df["description_length"] >= 0).all()
    ):
        raise ValueError("[DC-04] Violação de limites numéricos nos metadados pré-upload.")

    if not (
        (df["subscriber_count"] > 0).all()
        and (df["views"] > 0).all()
        and (df["likes"] >= 0).all()
        and (df["likes"] <= df["views"]).all()
        and (df["comments"] >= 0).all()
        and (df["comments"] <= df["views"]).all()
        and (df["shares"] >= 0).all()
        and (df["shares"] <= df["views"]).all()
    ):
        raise ValueError("[DC-05] Violação de invariantes de porte ou engajamento > views.")

    if not df["viral"].isin([0, 1]).all():
        raise ValueError("[DC-06] Coluna viral deve ser estritamente binária {0, 1}.")

    viral_rate = float(df["viral"].mean())
    if abs(viral_rate - 0.20) > 1e-3:
        raise ValueError(f"[DC-06] Proporção viral fora de 20,0%: {viral_rate:.4f}.")

    max_views_0 = int(df.loc[df["viral"] == 0, "views"].max())
    min_views_1 = int(df.loc[df["viral"] == 1, "views"].min())
    if max_views_0 >= min_views_1 or min_views_1 < VIRAL_VIEWS_CUTOFF:
        raise ValueError(
            f"[DC-06] Fronteira determinística P80 violada: max(views|0)={max_views_0}, "
            f"min(views|1)={min_views_1}."
        )

    return df


def assert_no_leakage(
    feature_names: Sequence[str],
    regime: Literal["A", "B"] = "A",
    forbid_subscribers: bool = False,
) -> None:
    """Verifica a ausência de colunas de target ou proxy leakage na matriz X (DC-07).

    Args:
        feature_names: Nomes das colunas preditoras da matriz X.
        regime: 'A' para Pré-Upload (Cold Start) ou 'B' para Pós-Upload Relativo.
        forbid_subscribers: Se True, também bloqueia subscriber_count (usado na Frente 3
            de modelagem do multiplicador relativo views/subscriber_count).

    Raises:
        ValueError: Se alguma coluna proibida estiver presente em feature_names.
    """
    col_set = set(feature_names)
    leaked_global = col_set & GLOBAL_FORBIDDEN_FEATURES
    if leaked_global:
        raise ValueError(
            f"[DC-07] Target ou Proxy Leakage detectado na matriz X: {sorted(leaked_global)}"
        )

    if regime == "A":
        leaked_regime_b = col_set & REGIME_B_ONLY_FEATURES
        if leaked_regime_b:
            raise ValueError(
                f"[DC-07] Sinais pós-upload proibidos no Regime A (Cold Start): "
                f"{sorted(leaked_regime_b)}"
            )

    if forbid_subscribers:
        sub_cols = col_set & {"subscriber_count", "log_subscriber_count"}
        if sub_cols:
            raise ValueError(
                f"[DC-07] Colunas de inscritos proibidas no modelo de desconfundimento: "
                f"{sorted(sub_cols)}"
            )
