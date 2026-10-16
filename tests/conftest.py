"""Fixtures compartilhadas para a suíte de testes TDD do projeto youtube_viral."""

from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "viral_videos.csv"


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    """Carrega o dataset bruto viral_videos.csv uma única vez por sessão de testes."""
    return pd.read_csv(CSV_PATH)
