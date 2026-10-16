"""Métricas de avaliação (RMSLE, PR-AUC, Duan) e bateria de testes de hipóteses (H1 a H5)."""

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def compute_regression_metrics(
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    is_log1p: bool = True,
) -> dict[str, float]:
    """Calcula RMSLE, MAE_ln, R2_ln e MdAPE com fator de correção de viés de Duan.

    Args:
        y_true_log: Vetor real transformado (ln(1+views) ou ln(views/subs)).
        y_pred_log: Vetor predito na mesma escala logarítmica.
        is_log1p: Se True, aplica exp(y)*smearing - 1 para calcular MdAPE em views.

    Returns:
        Dicionário com métricas arredondadas para precisão de ponto flutuante padrão.
    """
    y_t = np.asarray(y_true_log, dtype=float)
    y_p = np.asarray(y_pred_log, dtype=float)

    rmsle = float(np.sqrt(mean_squared_error(y_t, y_p)))
    mae_ln = float(mean_absolute_error(y_t, y_p))
    r2_ln = float(r2_score(y_t, y_p))

    residuals = y_t - y_p
    smearing = float(np.mean(np.exp(np.clip(residuals, -10.0, 10.0))))

    if is_log1p:
        orig_true = np.expm1(y_t)
        orig_pred = np.maximum(1.0, np.exp(y_p) * smearing - 1.0)
    else:
        orig_true = np.exp(y_t)
        orig_pred = np.exp(y_p) * smearing

    ape = np.abs(orig_true - orig_pred) / np.maximum(1e-9, orig_true) * 100.0
    mdape_pct = float(np.median(ape))

    return {
        "rmsle": rmsle,
        "mae_ln": mae_ln,
        "r2_ln": r2_ln,
        "mdape_pct": mdape_pct,
        "duan_smearing": smearing,
    }


def compute_classification_metrics(
    y_true_bin: np.ndarray,
    y_score: np.ndarray,
    is_probability: bool = True,
) -> dict[str, float]:
    """Calcula PR-AUC, ROC-AUC, F1@0.50, F1@P80, Precision@P80, Recall@P80 e Brier Score.

    Args:
        y_true_bin: Rótulo binário verdadeiro {0, 1}.
        y_score: Probabilidades preditas [0, 1] ou escores contínuos monotônicos (ex: log1p_views).
        is_probability: Se False (ex: escore contínuo de regressão), normaliza por ranking
            ou sigmoide em torno de P80 para cálculo do Brier Score equivalente.

    Returns:
        Dicionário com métricas de classificação.
    """
    y_t = np.asarray(y_true_bin, dtype=int)
    scores = np.asarray(y_score, dtype=float)

    pr_auc = float(average_precision_score(y_t, scores))
    roc_auc = float(roc_auc_score(y_t, scores))

    # Threshold P80 (top 20% preditos como viral == 1)
    tau_p80 = float(np.quantile(scores, 0.80))
    pred_p80 = (scores >= tau_p80).astype(int)
    f1_p80 = float(f1_score(y_t, pred_p80, zero_division=0))
    prec_p80 = float(precision_score(y_t, pred_p80, zero_division=0))
    rec_p80 = float(recall_score(y_t, pred_p80, zero_division=0))

    if is_probability:
        pred_050 = (scores >= 0.50).astype(int)
        f1_050 = float(f1_score(y_t, pred_050, zero_division=0))
        probs = np.clip(scores, 0.0, 1.0)
        brier = float(brier_score_loss(y_t, probs))
    else:
        # Para o regressor dual, o limiar natural em log1p(views) é ln(1 + 734379)
        tau_natural = float(np.log1p(734379.0))
        pred_nat = (scores >= tau_natural).astype(int)
        f1_050 = float(f1_score(y_t, pred_nat, zero_division=0))
        std_s = max(1e-6, float(np.std(scores)))
        probs = 1.0 / (1.0 + np.exp(-(scores - tau_natural) / (0.35 * std_s)))
        brier = float(brier_score_loss(y_t, probs))

    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "f1_at_p80": f1_p80,
        "f1_at_050": f1_050,
        "precision_at_p80": prec_p80,
        "recall_at_p80": rec_p80,
        "brier_score": brier,
        "threshold_p80": tau_p80,
    }


def run_hypothesis_tests(df: pd.DataFrame) -> dict[str, Any]:
    """Executa os testes estatísticos formais para as hipóteses H1, H2, H3 e H4.

    Args:
        df: DataFrame completo ou de treino de viral_videos.csv.

    Returns:
        Dicionário estruturado com estatísticas de teste e p-valores.
    """
    log_views = np.log1p(df["views"].to_numpy(dtype=float))
    log_subs = np.log1p(df["subscriber_count"].to_numpy(dtype=float))
    log_mult = np.log(
        df["views"].to_numpy(dtype=float) / df["subscriber_count"].to_numpy(dtype=float)
    )
    dur = df["duration_sec"].to_numpy(dtype=float)
    tags = df["tags_count"].to_numpy(dtype=float)
    viral = df["viral"].to_numpy(dtype=int)

    # H1: Dominância de subscriber_count
    r_subs, p_subs = stats.pearsonr(log_subs, log_views)
    r_subs_viral, _ = stats.pearsonr(log_subs, viral)

    # H2: Paradoxo de Simpson em duration_sec
    r_dur_marg, p_dur_marg = stats.pearsonr(dur, log_views)
    r_dur_deconf, p_dur_deconf = stats.pearsonr(dur, log_mult)

    # Correlação parcial de duration_sec com log_views controlada por log_subs
    slope_v, intercept_v, _, _, _ = stats.linregress(log_subs, log_views)
    res_views = log_views - (intercept_v + slope_v * log_subs)
    slope_d, intercept_d, _, _, _ = stats.linregress(log_subs, dur)
    res_dur = dur - (intercept_d + slope_d * log_subs)
    r_dur_partial_subs, p_dur_partial_subs = stats.pearsonr(res_dur, res_views)

    # H3: Alavancas do criador (thumbnail_face e tags_count)
    mask_face_1 = df["thumbnail_face"].to_numpy(dtype=int) == 1
    mult_face_1 = log_mult[mask_face_1]
    mult_face_0 = log_mult[~mask_face_1]
    u_stat, p_mw_face = stats.mannwhitneyu(mult_face_1, mult_face_0, alternative="two-sided")
    t_stat, p_t_face = stats.ttest_ind(mult_face_1, mult_face_0, equal_var=False)

    viral_rate_face_1 = float(np.mean(viral[mask_face_1]))
    viral_rate_face_0 = float(np.mean(viral[~mask_face_1]))

    r_tags_deconf, p_tags_deconf = stats.pearsonr(tags, log_mult)
    r_tags_marg, p_tags_marg = stats.pearsonr(tags, log_views)

    # H4: Neutralidade de upload_hour e upload_day
    hour_groups = [
        grp.to_numpy(dtype=float)
        for _, grp in pd.Series(log_mult).groupby(df["upload_hour"].to_numpy())
    ]
    kw_hour_stat, kw_hour_p = stats.kruskal(*hour_groups)

    day_groups = [
        grp.to_numpy(dtype=float)
        for _, grp in pd.Series(log_mult).groupby(df["upload_day"].to_numpy())
    ]
    kw_day_stat, kw_day_p = stats.kruskal(*day_groups)

    contingency_day = pd.crosstab(df["upload_day"], df["viral"])
    chi2_day, p_chi2_day, _, _ = stats.chi2_contingency(contingency_day)

    contingency_hour = pd.crosstab(df["upload_hour"], df["viral"])
    chi2_hour, p_chi2_hour, _, _ = stats.chi2_contingency(contingency_hour)

    return {
        "H1_channel_dominance": {
            "r_log_subs_views": float(r_subs),
            "r2_log_subs_alone": float(r_subs**2),
            "p_log_subs_views": float(p_subs),
            "r_log_subs_viral": float(r_subs_viral),
        },
        "H2_simpson_duration": {
            "r_marginal_log_views": float(r_dur_marg),
            "p_marginal_log_views": float(p_dur_marg),
            "r_deconfounded_views_per_sub": float(r_dur_deconf),
            "p_deconfounded_views_per_sub": float(p_dur_deconf),
            "r_partial_controlling_log_subs": float(r_dur_partial_subs),
            "p_partial_controlling_log_subs": float(p_dur_partial_subs),
        },
        "H3_creator_levers": {
            "face_viral_rate_1": viral_rate_face_1,
            "face_viral_rate_0": viral_rate_face_0,
            "face_relative_lift_pct": float((viral_rate_face_1 / viral_rate_face_0 - 1.0) * 100.0),
            "face_mult_mean_1": float(np.mean(mult_face_1)),
            "face_mult_mean_0": float(np.mean(mult_face_0)),
            "face_mannwhitney_u": float(u_stat),
            "face_mannwhitney_p": float(p_mw_face),
            "face_welch_t": float(t_stat),
            "face_welch_p": float(p_t_face),
            "tags_r_marginal": float(r_tags_marg),
            "tags_p_marginal": float(p_tags_marg),
            "tags_r_deconfounded": float(r_tags_deconf),
            "tags_p_deconfounded": float(p_tags_deconf),
        },
        "H4_timing_neutrality": {
            "hour_kruskal_stat": float(kw_hour_stat),
            "hour_kruskal_p": float(kw_hour_p),
            "hour_chi2_viral_stat": float(chi2_hour),
            "hour_chi2_viral_p": float(p_chi2_hour),
            "day_kruskal_stat": float(kw_day_stat),
            "day_kruskal_p": float(kw_day_p),
            "day_chi2_viral_stat": float(chi2_day),
            "day_chi2_viral_p": float(p_chi2_day),
        },
    }
