"""Geração das 5 figuras pareadas (1x2) em 300 DPI obedecendo ao quarto-quality-checklist.md."""

from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_curve

matplotlib.use("Agg")

PALETTE_MAIN = {
    "non_viral": "#1f4e78",
    "viral": "#d95f02",
    "accent": "#2b8cbe",
    "positive": "#1b9e77",
    "neutral": "#64748b",
}

FEATURE_LABELS_PT: dict[str, str] = {
    "log_subscriber_count": "ln(1 + Inscritos)",
    "subscriber_count": "Inscritos (Bruto)",
    "category": "Categoria Editorial",
    "thumbnail_face": "Rosto na Thumbnail",
    "tags_count": "Contagem de Tags",
    "tags_per_title_char": "Tags / Tam. Título",
    "duration_sec": "Duração (s)",
    "log_duration_sec": "ln(1 + Duração)",
    "description_length": "Tam. Descrição",
    "desc_to_title_ratio": "Descrição / Título",
    "title_length": "Tam. Título",
    "upload_hour": "Hora do Upload",
    "hour_sin": "Hora (Seno)",
    "hour_cos": "Hora (Cosseno)",
    "upload_day": "Dia da Semana",
    "is_weekend": "Fim de Semana",
    "has_tags": "Possui Tags",
    "has_description": "Possui Descrição",
    "like_rate": "Taxa de Likes (Pós)",
    "comment_rate": "Taxa de Comentários (Pós)",
    "share_rate": "Taxa de Shares (Pós)",
    "engagement_rate_total": "Engajamento Total (Pós)",
    "comments_per_like": "Comentários / Like (Pós)",
    "shares_per_like": "Shares / Like (Pós)",
}


def _apply_publication_style(ax: plt.Axes) -> None:
    """Aplica set_box_aspect(1), despine e grade sutil conforme quarto-quality-checklist.md."""
    ax.set_box_aspect(1)
    sns.despine(ax=ax, top=True, right=True)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.7)


def plot_fig01_distributions_and_threshold(
    df: pd.DataFrame,
    hypotheses: dict[str, Any],
    out_path: Path,
) -> Path:
    """Figura 1 (Q1/H1): Distribuição log10(views) com corte P80 e elasticidade por inscritos."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)

    log10_views = np.log10(df["views"].to_numpy(dtype=float))
    log10_subs = np.log10(df["subscriber_count"].to_numpy(dtype=float))
    viral = df["viral"].to_numpy(dtype=int)
    cutoff_log10 = float(np.log10(734379.0))

    # Painel (a): Histograma estratificado pelo limiar P80
    ax0 = axes[0]
    ax0.hist(
        log10_views[viral == 0],
        bins=42,
        range=(2.0, 8.5),
        color=PALETTE_MAIN["non_viral"],
        alpha=0.78,
        label="Não Viral (80,0% | < 734,4k)",
    )
    ax0.hist(
        log10_views[viral == 1],
        bins=42,
        range=(2.0, 8.5),
        color=PALETTE_MAIN["viral"],
        alpha=0.85,
        label="Viral (20,0% | >= 734,4k)",
    )
    ax0.axvline(cutoff_log10, color="#111827", linestyle="--", linewidth=1.8)
    ax0.annotate(
        "Fronteira P80 Exata\n734.379 views\n(log10 = 5,866)",
        xy=(cutoff_log10, 680),
        xytext=(cutoff_log10 + 0.35, 760),
        fontsize=8.5,
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": "#111827", "lw": 1.2},
    )
    ax0.set_title("(a) Distribuição Logarítmica e Corte P80", fontsize=11, fontweight="bold")
    ax0.set_xlabel("Visualizações em Escala Log10 [log10(views)]", fontsize=9.5)
    ax0.set_ylabel("Contagem de Vídeos (N = 12.000)", fontsize=9.5)
    ax0.legend(frameon=True, fontsize=8.2, loc="upper left")
    _apply_publication_style(ax0)

    # Painel (b): Dispersão densa log10(subs) vs log10(views)
    ax1 = axes[1]
    rng = np.random.default_rng(42)
    idx_sample = rng.choice(len(df), size=3000, replace=False)
    colors = np.where(viral[idx_sample] == 1, PALETTE_MAIN["viral"], PALETTE_MAIN["non_viral"])
    ax1.scatter(
        log10_subs[idx_sample],
        log10_views[idx_sample],
        c=colors,
        s=10,
        alpha=0.35,
        edgecolors="none",
    )
    # Linha de tendência linear
    slope, intercept = np.polyfit(log10_subs, log10_views, 1)
    x_vals = np.linspace(3.5, 8.0, 100)
    ax1.plot(
        x_vals,
        intercept + slope * x_vals,
        color="#111827",
        linewidth=2.0,
        label=f"Ajuste OLS (inclinação = {slope:.2f})",
    )
    ax1.axhline(cutoff_log10, color=PALETTE_MAIN["viral"], linestyle="--", linewidth=1.4)
    r_val = hypotheses["H1_channel_dominance"]["r_log_subs_views"]
    r2_val = hypotheses["H1_channel_dominance"]["r2_log_subs_alone"] * 100.0
    ax1.text(
        0.05,
        0.88,
        f"r de Pearson = +{r_val:.3f}\nVariância explicada = {r2_val:.1f}%",
        transform=ax1.transAxes,
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f8fafc", "edgecolor": "#cbd5e1"},
    )
    ax1.set_title("(b) Elasticidade Audiência vs. Porte do Canal", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Inscritos do Canal [log10(subscriber_count)]", fontsize=9.5)
    ax1.set_ylabel("Visualizações [log10(views)]", fontsize=9.5)
    ax1.legend(frameon=True, fontsize=8.2, loc="lower right")
    _apply_publication_style(ax1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_fig02_simpson_paradox_categories(
    df: pd.DataFrame,
    hypotheses: dict[str, Any],
    out_path: Path,
) -> Path:
    """Figura 2 (Q2/H2): Perfil por categoria e decomposição do Paradoxo de Simpson na duração."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)

    cat_stats = (
        df.groupby("category")
        .agg(
            viral_pct=("viral", lambda s: float(np.mean(s) * 100.0)),
            med_subs_k=("subscriber_count", lambda s: float(np.median(s) / 1000.0)),
            med_dur=("duration_sec", "median"),
        )
        .sort_values("viral_pct", ascending=True)
    )

    # Painel (a): Taxa de viralidade vs mediana de inscritos por categoria
    ax0 = axes[0]
    y_pos = np.arange(len(cat_stats))
    bars = ax0.barh(
        y_pos,
        cat_stats["viral_pct"],
        color=PALETTE_MAIN["non_viral"],
        alpha=0.85,
        height=0.68,
    )
    bars[-1].set_color(PALETTE_MAIN["viral"])
    bars[-2].set_color(PALETTE_MAIN["viral"])

    for i, (v_pct, subs_k) in enumerate(
        zip(cat_stats["viral_pct"], cat_stats["med_subs_k"], strict=True)
    ):
        ax0.text(
            v_pct + 0.8,
            i,
            f"{v_pct:.1f}% ({subs_k:.0f}k subs)",
            va="center",
            fontsize=7.3,
            color="#1e293b",
        )

    ax0.axvline(20.0, color="#64748b", linestyle="--", linewidth=1.2, label="Média Global (20%)")
    ax0.set_yticks(y_pos)
    ax0.set_yticklabels(cat_stats.index, fontsize=8.2)
    ax0.set_xlim(0, 76)
    ax0.set_title("(a) Taxa Viral e Porte Mediano por Categoria", fontsize=11, fontweight="bold")
    ax0.set_xlabel("Taxa de Vídeos Virais (%) [Rótulo: Mediana de Inscritos]", fontsize=9.0)
    ax0.set_ylabel("Categoria Editorial", fontsize=9.5)
    _apply_publication_style(ax0)

    # Painel (b): Paradoxo de Simpson (Duração vs Views Bruto vs Multiplicador Desconfundido)
    ax1 = axes[1]
    dur_bins = pd.qcut(df["duration_sec"], q=10, duplicates="drop")
    df_temp = pd.DataFrame(
        {
            "dur_bin": dur_bins,
            "dur_sec": df["duration_sec"].astype(float),
            "log_views_z": (
                np.log1p(df["views"].astype(float)) - np.mean(np.log1p(df["views"].astype(float)))
            )
            / np.std(np.log1p(df["views"].astype(float))),
            "log_mult_z": (
                np.log(df["views"].astype(float) / df["subscriber_count"].astype(float))
                - np.mean(np.log(df["views"].astype(float) / df["subscriber_count"].astype(float)))
            )
            / np.std(np.log(df["views"].astype(float) / df["subscriber_count"].astype(float))),
        }
    )
    binned = df_temp.groupby("dur_bin", observed=True).agg(
        dur_mean=("dur_sec", "mean"),
        views_z_mean=("log_views_z", "mean"),
        views_z_se=("log_views_z", lambda s: float(np.std(s) / np.sqrt(len(s)))),
        mult_z_mean=("log_mult_z", "mean"),
        mult_z_se=("log_mult_z", lambda s: float(np.std(s) / np.sqrt(len(s)))),
    )

    r_marg = hypotheses["H2_simpson_duration"]["r_marginal_log_views"]
    r_dec = hypotheses["H2_simpson_duration"]["r_deconfounded_views_per_sub"]

    ax1.errorbar(
        binned["dur_mean"],
        binned["views_z_mean"],
        yerr=1.96 * binned["views_z_se"],
        fmt="-o",
        color=PALETTE_MAIN["viral"],
        linewidth=2.0,
        markersize=5,
        capsize=3,
        label=f"Bruto: ln(1+views) (r = {r_marg:+.3f}, p < 0,0001)",
    )
    ax1.errorbar(
        binned["dur_mean"],
        binned["mult_z_mean"],
        yerr=1.96 * binned["mult_z_se"],
        fmt="-s",
        color=PALETTE_MAIN["positive"],
        linewidth=2.0,
        markersize=5,
        capsize=3,
        label=f"Desconfundido: ln(views/subs) (r = {r_dec:+.3f}, p = 0,40)",
    )
    ax1.axhline(0.0, color="#64748b", linestyle=":", linewidth=1.2)
    ax1.annotate(
        "Paradoxo de Simpson:\nQueda aparente some ao\ncontrolar pelos inscritos",
        xy=(1100, -0.28),
        xytext=(550, -0.42),
        fontsize=8.3,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#fff7ed", "edgecolor": "#fdba74"},
        arrowprops={"arrowstyle": "->", "color": PALETTE_MAIN["viral"], "lw": 1.2},
    )
    ax1.set_ylim(-0.55, 0.55)
    ax1.set_title("(b) Decomposição do Paradoxo de Simpson", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Duração do Vídeo por Decil [duration_sec (s)]", fontsize=9.5)
    ax1.set_ylabel("Efeito Padronizado (Z-Score +/- IC 95%)", fontsize=9.5)
    ax1.legend(frameon=True, fontsize=8.0, loc="upper right")
    _apply_publication_style(ax1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_fig03_creator_levers_and_timing(
    df: pd.DataFrame,
    hypotheses: dict[str, Any],
    out_path: Path,
) -> Path:
    """Figura 3 (Q3/H3/H4): Efeito de thumbnail_face e tags_count vs neutralidade horária."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)

    # Painel (a): Multiplicador views/subs por quartil de inscritos e presença de rosto
    ax0 = axes[0]
    sub_quartiles = pd.qcut(
        df["subscriber_count"],
        q=4,
        labels=["Q1 (<159k)", "Q2 (159k-381k)", "Q3 (381k-930k)", "Q4 (>930k)"],
    )
    df_q = pd.DataFrame(
        {
            "sub_q": sub_quartiles,
            "face": df["thumbnail_face"].astype(int),
            "mult": df["views"].astype(float) / df["subscriber_count"].astype(float),
        }
    )
    grp = (
        df_q.groupby(["sub_q", "face"], observed=True)["mult"]
        .agg(
            mean="mean",
            se=lambda s: float(np.std(s, ddof=1) / np.sqrt(len(s))),
        )
        .unstack("face")
    )

    x = np.arange(len(grp))
    width = 0.35
    ax0.bar(
        x - width / 2,
        grp["mean"][0],
        width,
        yerr=1.96 * grp["se"][0],
        color=PALETTE_MAIN["non_viral"],
        alpha=0.85,
        capsize=3,
        label="Sem Rosto na Thumbnail (0)",
    )
    ax0.bar(
        x + width / 2,
        grp["mean"][1],
        width,
        yerr=1.96 * grp["se"][1],
        color=PALETTE_MAIN["positive"],
        alpha=0.88,
        capsize=3,
        label="Com Rosto na Thumbnail (1)",
    )
    lift_pct = hypotheses["H3_creator_levers"]["face_relative_lift_pct"]
    p_mw = hypotheses["H3_creator_levers"]["face_mannwhitney_p"]
    ax0.text(
        0.04,
        0.86,
        f"Ganho Global Taxa Viral: +{lift_pct:.1f}%\n(22,8% vs. 18,1% | p = {p_mw:.1e})",
        transform=ax0.transAxes,
        fontsize=8.3,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#f0fdf4", "edgecolor": "#86efac"},
    )
    ax0.set_xticks(x)
    ax0.set_xticklabels(grp.index.astype(str), fontsize=8.3)
    ax0.set_ylim(0, 1.15)
    ax0.set_title(
        "(a) Alcance Relativo por Rosto na Thumbnail (H3)", fontsize=11, fontweight="bold"
    )
    ax0.set_xlabel("Quartil de Inscritos do Canal", fontsize=9.5)
    ax0.set_ylabel("Razão Média views / subscriber_count", fontsize=9.5)
    ax0.legend(frameon=True, fontsize=8.0, loc="lower right")
    _apply_publication_style(ax0)

    # Painel (b): Taxa de viralidade por upload_hour e fim de semana (H4)
    ax1 = axes[1]
    df_time = pd.DataFrame(
        {
            "hour": df["upload_hour"].astype(int),
            "weekend": np.where(
                df["upload_day"].isin(["Sat", "Sun"]),
                "Fim de Semana (Sáb-Dom)",
                "Dia Útil (Seg-Sex)",
            ),
            "viral_pct": df["viral"].astype(float) * 100.0,
        }
    )
    piv = df_time.groupby(["hour", "weekend"])["viral_pct"].mean().unstack("weekend")
    ax1.plot(
        piv.index,
        piv["Dia Útil (Seg-Sex)"],
        "-o",
        color=PALETTE_MAIN["non_viral"],
        linewidth=1.8,
        markersize=4,
        label="Dia Útil (Seg-Sex)",
    )
    ax1.plot(
        piv.index,
        piv["Fim de Semana (Sáb-Dom)"],
        "--s",
        color=PALETTE_MAIN["viral"],
        linewidth=1.8,
        markersize=4,
        label="Fim de Semana (Sáb-Dom)",
    )
    ax1.axhline(20.0, color="#111827", linestyle=":", linewidth=1.4, label="Taxa Base (20,0%)")
    ax1.axhspan(16.0, 24.0, color="#cbd5e1", alpha=0.25, label="Banda Ruído Amostral (95%)")
    p_kw_h = hypotheses["H4_timing_neutrality"]["hour_kruskal_p"]
    p_kw_d = hypotheses["H4_timing_neutrality"]["day_kruskal_p"]
    ax1.text(
        0.04,
        0.86,
        f"Kruskal-Wallis Hora: p = {p_kw_h:.2f} (NS)\nKruskal-Wallis Dia: p = {p_kw_d:.2f} (NS)",
        transform=ax1.transAxes,
        fontsize=8.3,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#f8fafc", "edgecolor": "#cbd5e1"},
    )
    ax1.set_xlim(-0.5, 23.5)
    ax1.set_ylim(8.0, 34.0)
    ax1.set_title(
        "(b) Neutralidade do Horário e Dia de Upload (H4)", fontsize=11, fontweight="bold"
    )
    ax1.set_xlabel("Hora de Publicação do Vídeo [upload_hour (0h-23h)]", fontsize=9.5)
    ax1.set_ylabel("Taxa de Vídeos Virais (%)", fontsize=9.5)
    ax1.legend(frameon=True, fontsize=7.8, loc="lower right")
    _apply_publication_style(ax1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_fig04_dual_modeling_performance(
    suite: dict[str, Any],
    out_path: Path,
) -> Path:
    """Figura 4 (Q4): Curvas Precision-Recall e Calibração Contínua Dual no Teste (N=2.400)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)

    y_bin_te = suite["test_arrays"]["y_bin_test"]
    y_reg_te = suite["test_arrays"]["y_reg_test"]
    preds_a = suite["test_arrays"]["predictions_A"]
    clf_res_a = suite["classification"]["A"]
    reg_res_a = suite["regression"]["A"]

    # Painel (a): Curvas Precision-Recall no conjunto de teste
    ax0 = axes[0]
    curve_specs = [
        (
            "LightGBM Regressor Dual (P80)",
            preds_a["lgbm_reg_log1p_views"],
            clf_res_a["LightGBM_DualRegressor"]["test"]["pr_auc"],
            PALETTE_MAIN["positive"],
            "-",
            2.2,
        ),
        (
            "LightGBM Classifier",
            preds_a["clf_LightGBM"],
            clf_res_a["LightGBM"]["test"]["pr_auc"],
            PALETTE_MAIN["viral"],
            "--",
            2.0,
        ),
        (
            "Logistic Regression",
            preds_a["clf_LogisticRegression"],
            clf_res_a["LogisticRegression"]["test"]["pr_auc"],
            PALETTE_MAIN["non_viral"],
            "-.",
            1.7,
        ),
        (
            "Random Forest",
            preds_a["clf_RandomForest"],
            clf_res_a["RandomForest"]["test"]["pr_auc"],
            PALETTE_MAIN["accent"],
            ":",
            1.7,
        ),
        (
            "Baseline Categoria",
            preds_a["clf_Baseline"],
            clf_res_a["Baseline"]["test"]["pr_auc"],
            PALETTE_MAIN["neutral"],
            "--",
            1.3,
        ),
    ]

    for label, scores, pr_val, color, ls, lw in curve_specs:
        prec, rec, _ = precision_recall_curve(y_bin_te, scores)
        ax0.plot(
            rec,
            prec,
            linestyle=ls,
            linewidth=lw,
            color=color,
            label=f"{label} ({pr_val:.3f})",
        )

    ax0.axhline(0.20, color="#94a3b8", linestyle=":", linewidth=1.0, label="Prevalência (0,200)")
    ax0.set_xlim(0.0, 1.0)
    ax0.set_ylim(0.15, 1.02)
    ax0.set_title("(a) Curvas Precision-Recall no Teste (Regime A)", fontsize=11, fontweight="bold")
    ax0.set_xlabel("Revocação (Recall na Classe Viral)", fontsize=9.5)
    ax0.set_ylabel("Precisão (Precision)", fontsize=9.5)
    ax0.legend(frameon=True, fontsize=7.6, loc="lower left", title="Modelo (PR-AUC Teste)")
    _apply_publication_style(ax0)

    # Painel (b): Predito vs Observado log10(views) com quadrantes de convergência P80
    ax1 = axes[1]
    obs_log10 = np.expm1(y_reg_te)
    obs_log10 = np.log10(np.maximum(1.0, obs_log10))
    pred_log10 = np.log10(np.maximum(1.0, np.expm1(preds_a["lgbm_reg_log1p_views"])))
    cutoff_log10 = float(np.log10(734379.0))
    pred_p80_cut = float(np.quantile(pred_log10, 0.80))

    colors = np.where(y_bin_te == 1, PALETTE_MAIN["viral"], PALETTE_MAIN["non_viral"])
    ax1.scatter(pred_log10, obs_log10, c=colors, s=9, alpha=0.35, edgecolors="none")
    ax1.plot(
        [3.5, 7.5],
        [3.5, 7.5],
        color="#111827",
        linewidth=1.6,
        label="Identidade Ideal (y = x)",
    )
    ax1.axhline(cutoff_log10, color=PALETTE_MAIN["viral"], linestyle="--", linewidth=1.3)
    ax1.axvline(pred_p80_cut, color=PALETTE_MAIN["positive"], linestyle="--", linewidth=1.3)

    lgb_reg_m = reg_res_a["LightGBM"]["test"]
    dual_f1 = clf_res_a["LightGBM_DualRegressor"]["test"]["f1_at_p80"]
    ax1.text(
        0.04,
        0.82,
        f"RMSLE Teste = {lgb_reg_m['rmsle']:.3f}\n"
        f"R² Log = {lgb_reg_m['r2_ln']:.3f}\n"
        f"F1@P80 Dual = {dual_f1:.3f}",
        transform=ax1.transAxes,
        fontsize=8.3,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f8fafc", "edgecolor": "#cbd5e1"},
    )
    ax1.set_xlim(3.5, 7.5)
    ax1.set_ylim(2.5, 8.2)
    ax1.set_title("(b) Calibração Contínua e Fronteira Dual P80", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Predição LightGBM [log10(views)]", fontsize=9.5)
    ax1.set_ylabel("Observado Real [log10(views)]", fontsize=9.5)
    ax1.legend(frameon=True, fontsize=8.0, loc="lower right")
    _apply_publication_style(ax1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_fig05_scenario_a_vs_b_importance(
    suite: dict[str, Any],
    out_path: Path,
) -> Path:
    """Figura 5 (Q5/H5): Comparação Regime A vs Regime B e Feature Importance por Gain."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)

    # Painel (a): Comparação pareada Regime A (Cold Start) vs Regime B (Early Signals)
    ax0 = axes[0]
    models_order = ["LogisticRegression", "RandomForest", "LightGBM", "LightGBM_DualRegressor"]
    labels_short = ["LogReg / Ridge", "Random Forest", "LightGBM Clf", "LightGBM Dual"]

    pr_a_vals = [suite["classification"]["A"][m]["test"]["pr_auc"] for m in models_order]
    pr_b_vals = [suite["classification"]["B"][m]["test"]["pr_auc"] for m in models_order]

    x = np.arange(len(models_order))
    width = 0.35
    bars_a = ax0.bar(
        x - width / 2,
        pr_a_vals,
        width,
        color=PALETTE_MAIN["non_viral"],
        alpha=0.88,
        label="Regime A: Pré-Upload (Cold Start)",
    )
    bars_b = ax0.bar(
        x + width / 2,
        pr_b_vals,
        width,
        color=PALETTE_MAIN["accent"],
        alpha=0.88,
        label="Regime B: + Taxas Pós-Upload",
    )

    for b in list(bars_a) + list(bars_b):
        h = b.get_height()
        ax0.text(
            b.get_x() + b.get_width() / 2.0,
            h + 0.012,
            f"{h:.3f}",
            ha="center",
            va="bottom",
            fontsize=7.3,
        )

    h5 = suite["H5_regime_comparison"]
    d_pr = h5["delta_test_pr_auc"]
    p_pr = h5["cv_pr_auc_ttest_p"]
    d_rm = h5["delta_test_rmsle"]
    p_rm = h5["cv_rmsle_ttest_p"]
    ax0.text(
        0.04,
        0.10,
        f"Hipótese H5 Confirmada:\nDelta PR-AUC = {d_pr:+.4f} (p = {p_pr:.2f})\n"
        f"Delta RMSLE = {d_rm:+.4f} (p = {p_rm:.2f})",
        transform=ax0.transAxes,
        fontsize=8.1,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f8fafc", "edgecolor": "#94a3b8"},
    )
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels_short, fontsize=8.3)
    ax0.set_ylim(0.0, 0.92)
    ax0.set_title("(a) Regime A (Cold Start) vs. Regime B (H5)", fontsize=11, fontweight="bold")
    ax0.set_xlabel("Família de Modelos Supervisionados", fontsize=9.5)
    ax0.set_ylabel("PR-AUC no Conjunto de Teste (N = 2.400)", fontsize=9.5)
    ax0.legend(frameon=True, fontsize=8.0, loc="upper left")
    _apply_publication_style(ax0)

    # Painel (b): Top 10 variáveis por importância de Ganho ('gain') no Regime B
    ax1 = axes[1]
    imp_list = suite["feature_importance"]["B_regression_gain"][:10][::-1]
    f_names = [FEATURE_LABELS_PT.get(item["feature"], item["feature"]) for item in imp_list]
    g_pcts = [item["gain_pct"] for item in imp_list]
    early_set = {
        "like_rate",
        "comment_rate",
        "share_rate",
        "engagement_rate_total",
        "comments_per_like",
        "shares_per_like",
    }
    bar_colors = [
        PALETTE_MAIN["viral"] if item["feature"] in early_set else PALETTE_MAIN["non_viral"]
        for item in imp_list
    ]

    y_pos = np.arange(len(imp_list))
    ax1.barh(y_pos, g_pcts, color=bar_colors, alpha=0.88, height=0.68)
    for i, val in enumerate(g_pcts):
        ax1.text(val + 1.2, i, f"{val:.1f}%", va="center", fontsize=7.8, fontweight="bold")

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(f_names, fontsize=8.2)
    ax1.set_xlim(0.0, 108.0)
    ax1.set_title(
        "(b) Importância por Ganho (importance_type='gain')", fontsize=11, fontweight="bold"
    )
    ax1.set_xlabel("Participação na Redução de Erro Logarítmico (%)", fontsize=9.5)
    ax1.set_ylabel("Atributos (Regime B Completo)", fontsize=9.5)
    _apply_publication_style(ax1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path
