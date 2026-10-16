"""Modelagem supervisionada tripla, convergência dual no quantil P80 e explicabilidade Gain/SHAP."""

from typing import Any, Literal

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from youtube_viral import SEED
from youtube_viral.evaluation import (
    compute_classification_metrics,
    compute_regression_metrics,
)
from youtube_viral.features import (
    build_feature_matrix,
    build_targets,
    stratified_train_test_split,
)

CATEGORICAL_COLS: list[str] = ["category", "upload_day"]


def _make_preprocessor(df_X: pd.DataFrame, scale_numeric: bool = True) -> ColumnTransformer:
    """Cria o ColumnTransformer codificando categóricas e escalonando numéricas."""
    num_cols = [c for c in df_X.columns if c not in CATEGORICAL_COLS]
    num_step = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLS,
            ),
            ("num", num_step, num_cols),
        ],
        remainder="drop",
    )


def _prepare_lgbm_frame(df_X: pd.DataFrame) -> pd.DataFrame:
    """Converte category e upload_day para dtype 'category' nativo do LightGBM."""
    out = df_X.copy()
    for col in CATEGORICAL_COLS:
        out[col] = out[col].astype("category")
    return out


def _extract_lgbm_gain_and_shap(
    estimator: LGBMRegressor | LGBMClassifier,
    X_eval: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Extrai importância normalizada por 'gain' e média absoluta TreeSHAP do LightGBM."""
    X_cat = _prepare_lgbm_frame(X_eval)
    booster = estimator.booster_
    gains = np.asarray(booster.feature_importance(importance_type="gain"), dtype=float)
    splits = np.asarray(booster.feature_importance(importance_type="split"), dtype=float)
    total_gain = max(1e-12, float(np.sum(gains)))
    gain_pcts = (gains / total_gain) * 100.0

    # TreeSHAP nativo via pred_contrib=True (última coluna é o intercepto/expected_value)
    contribs = np.asarray(estimator.predict(X_cat, pred_contrib=True), dtype=float)
    mean_abs_shap = np.mean(np.abs(contribs[:, :-1]), axis=0)

    items = []
    for fname, g_val, g_pct, sp_val, sh_val in zip(
        X_cat.columns, gains, gain_pcts, splits, mean_abs_shap, strict=True
    ):
        items.append(
            {
                "feature": str(fname),
                "gain_raw": float(g_val),
                "gain_pct": float(g_pct),
                "split_count": int(sp_val),
                "mean_abs_shap": float(sh_val),
            }
        )
    items.sort(key=lambda d: d["gain_pct"], reverse=True)
    return items


def _aggregate_cv_folds(fold_metrics: list[dict[str, float]]) -> dict[str, float]:
    """Calcula média e desvio-padrão de cada métrica ao longo das dobras de CV."""
    keys = list(fold_metrics[0].keys())
    summary: dict[str, float] = {}
    for k in keys:
        vals = [fm[k] for fm in fold_metrics]
        summary[f"{k}_mean"] = float(np.mean(vals))
        summary[f"{k}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return summary


def run_modeling_suite(
    df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = SEED,
) -> dict[str, Any]:
    """Executa a suíte de modelagem (Classificação, Regressão, Dual P80 e Desconfundimento).

    Args:
        df: DataFrame completo de viral_videos.csv.
        n_splits: Número de dobras na validação cruzada estratificada (padrão 5).
        random_state: Semente determinística global.

    Returns:
        Dicionário contendo métricas de CV e teste, predições de teste, H5 e importâncias.
    """
    train_df, test_df = stratified_train_test_split(df, random_state=random_state)
    y_train_df = build_targets(train_df)
    y_test_df = build_targets(test_df)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = list(skf.split(train_df, y_train_df["strata"]))

    classification_results: dict[str, dict[str, Any]] = {"A": {}, "B": {}}
    regression_results: dict[str, dict[str, Any]] = {"A": {}, "B": {}}
    test_predictions: dict[str, dict[str, np.ndarray]] = {"A": {}, "B": {}}
    fold_wise_lgbm: dict[str, dict[str, list[float]]] = {
        "A": {"pr_auc": [], "rmsle": []},
        "B": {"pr_auc": [], "rmsle": []},
    }
    feature_importance: dict[str, list[dict[str, Any]]] = {}

    for regime in ("A", "B"):
        reg_lit: Literal["A", "B"] = "A" if regime == "A" else "B"
        X_train = build_feature_matrix(train_df, regime=reg_lit)
        X_test = build_feature_matrix(test_df, regime=reg_lit)
        X_train_lgbm = _prepare_lgbm_frame(X_train)
        X_test_lgbm = _prepare_lgbm_frame(X_test)

        y_bin_tr = y_train_df["viral"].to_numpy(dtype=int)
        y_bin_te = y_test_df["viral"].to_numpy(dtype=int)
        y_reg_tr = y_train_df["log1p_views"].to_numpy(dtype=float)
        y_reg_te = y_test_df["log1p_views"].to_numpy(dtype=float)

        # --- 1. MODELOS DE REGRESSÃO (log1p_views) ---
        reg_specs: dict[str, Any] = {
            "Baseline": "category_mean",
            "Ridge": Pipeline(
                [
                    ("prep", _make_preprocessor(X_train, scale_numeric=True)),
                    ("model", Ridge(alpha=1.0, random_state=random_state)),
                ]
            ),
            "RandomForest": Pipeline(
                [
                    ("prep", _make_preprocessor(X_train, scale_numeric=False)),
                    (
                        "model",
                        RandomForestRegressor(
                            n_estimators=200,
                            max_depth=10,
                            min_samples_leaf=5,
                            random_state=random_state,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            "LightGBM": LGBMRegressor(
                n_estimators=250,
                learning_rate=0.04,
                num_leaves=15,
                min_child_samples=40,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
                n_jobs=-1,
                verbose=-1,
            ),
        }

        lgbm_reg_oof = np.zeros(len(train_df), dtype=float)
        lgbm_reg_fold_clfs: list[dict[str, float]] = []

        for m_name, spec in reg_specs.items():
            fold_ms: list[dict[str, float]] = []
            for tr_idx, val_idx in folds:
                if spec == "category_mean":
                    cat_means = (
                        pd.Series(y_reg_tr[tr_idx])
                        .groupby(X_train.iloc[tr_idx]["category"].to_numpy())
                        .mean()
                    )
                    val_preds = (
                        X_train.iloc[val_idx]["category"]
                        .map(cat_means)
                        .fillna(float(np.mean(y_reg_tr[tr_idx])))
                        .to_numpy(dtype=float)
                    )
                elif m_name == "LightGBM":
                    model = LGBMRegressor(**spec.get_params())
                    model.fit(X_train_lgbm.iloc[tr_idx], y_reg_tr[tr_idx])
                    val_preds = np.asarray(model.predict(X_train_lgbm.iloc[val_idx]), dtype=float)
                    lgbm_reg_oof[val_idx] = val_preds
                    lgbm_reg_fold_clfs.append(
                        compute_classification_metrics(
                            y_bin_tr[val_idx], val_preds, is_probability=False
                        )
                    )
                else:
                    spec.fit(X_train.iloc[tr_idx], y_reg_tr[tr_idx])
                    val_preds = np.asarray(spec.predict(X_train.iloc[val_idx]), dtype=float)

                f_m = compute_regression_metrics(y_reg_tr[val_idx], val_preds)
                fold_ms.append(f_m)
                if m_name == "LightGBM":
                    fold_wise_lgbm[regime]["rmsle"].append(f_m["rmsle"])

            # Ajuste no treino completo e avaliação no cofre de teste
            if spec == "category_mean":
                cat_means_full = pd.Series(y_reg_tr).groupby(X_train["category"].to_numpy()).mean()
                te_preds = (
                    X_test["category"]
                    .map(cat_means_full)
                    .fillna(float(np.mean(y_reg_tr)))
                    .to_numpy(dtype=float)
                )
            elif m_name == "LightGBM":
                spec.fit(X_train_lgbm, y_reg_tr)
                te_preds = np.asarray(spec.predict(X_test_lgbm), dtype=float)
                test_predictions[regime]["lgbm_reg_log1p_views"] = te_preds
                feature_importance[f"{regime}_regression_gain"] = _extract_lgbm_gain_and_shap(
                    spec, X_test
                )
            else:
                spec.fit(X_train, y_reg_tr)
                te_preds = np.asarray(spec.predict(X_test), dtype=float)
                if m_name == "Ridge":
                    test_predictions[regime]["ridge_reg_log1p_views"] = te_preds

            regression_results[regime][m_name] = {
                "cv": _aggregate_cv_folds(fold_ms),
                "test": compute_regression_metrics(y_reg_te, te_preds),
            }

        # --- 2. MODELOS DE CLASSIFICAÇÃO (viral) ---
        clf_specs: dict[str, Any] = {
            "Baseline": "category_prior",
            "LogisticRegression": Pipeline(
                [
                    ("prep", _make_preprocessor(X_train, scale_numeric=True)),
                    ("model", LogisticRegression(C=1.0, max_iter=1000, random_state=random_state)),
                ]
            ),
            "RandomForest": Pipeline(
                [
                    ("prep", _make_preprocessor(X_train, scale_numeric=False)),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=200,
                            max_depth=10,
                            min_samples_leaf=5,
                            random_state=random_state,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            "LightGBM": LGBMClassifier(
                n_estimators=250,
                learning_rate=0.04,
                num_leaves=15,
                min_child_samples=40,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
                n_jobs=-1,
                verbose=-1,
            ),
        }

        for m_name, spec in clf_specs.items():
            fold_ms_c: list[dict[str, float]] = []
            for tr_idx, val_idx in folds:
                if spec == "category_prior":
                    cat_priors = (
                        pd.Series(y_bin_tr[tr_idx])
                        .groupby(X_train.iloc[tr_idx]["category"].to_numpy())
                        .mean()
                    )
                    val_probs = (
                        X_train.iloc[val_idx]["category"]
                        .map(cat_priors)
                        .fillna(float(np.mean(y_bin_tr[tr_idx])))
                        .to_numpy(dtype=float)
                    )
                elif m_name == "LightGBM":
                    model = LGBMClassifier(**spec.get_params())
                    model.fit(X_train_lgbm.iloc[tr_idx], y_bin_tr[tr_idx])
                    val_probs = np.asarray(
                        model.predict_proba(X_train_lgbm.iloc[val_idx])[:, 1], dtype=float
                    )
                else:
                    spec.fit(X_train.iloc[tr_idx], y_bin_tr[tr_idx])
                    val_probs = np.asarray(
                        spec.predict_proba(X_train.iloc[val_idx])[:, 1], dtype=float
                    )

                f_mc = compute_classification_metrics(
                    y_bin_tr[val_idx], val_probs, is_probability=True
                )
                fold_ms_c.append(f_mc)
                if m_name == "LightGBM":
                    fold_wise_lgbm[regime]["pr_auc"].append(f_mc["pr_auc"])

            if spec == "category_prior":
                cat_priors_full = pd.Series(y_bin_tr).groupby(X_train["category"].to_numpy()).mean()
                te_probs = (
                    X_test["category"]
                    .map(cat_priors_full)
                    .fillna(float(np.mean(y_bin_tr)))
                    .to_numpy(dtype=float)
                )
            elif m_name == "LightGBM":
                spec.fit(X_train_lgbm, y_bin_tr)
                te_probs = np.asarray(spec.predict_proba(X_test_lgbm)[:, 1], dtype=float)
                feature_importance[f"{regime}_classification_gain"] = _extract_lgbm_gain_and_shap(
                    spec, X_test
                )
            else:
                spec.fit(X_train, y_bin_tr)
                te_probs = np.asarray(spec.predict_proba(X_test)[:, 1], dtype=float)

            test_predictions[regime][f"clf_{m_name}"] = te_probs
            classification_results[regime][m_name] = {
                "cv": _aggregate_cv_folds(fold_ms_c),
                "test": compute_classification_metrics(y_bin_te, te_probs, is_probability=True),
            }

        # Adiciona a Convergência Dual (LGBMRegressor avaliado como classificador em P80)
        te_reg_scores = test_predictions[regime]["lgbm_reg_log1p_views"]
        classification_results[regime]["LightGBM_DualRegressor"] = {
            "cv": _aggregate_cv_folds(lgbm_reg_fold_clfs),
            "test": compute_classification_metrics(y_bin_te, te_reg_scores, is_probability=False),
        }

    # --- 3. MODELO DE DESCONFUNDIMENTO (Frente 3: log(views / subscriber_count)) ---
    X_tr_dec = build_feature_matrix(train_df, regime="A", deconfound=True)
    X_te_dec = build_feature_matrix(test_df, regime="A", deconfound=True)
    y_mult_tr = y_train_df["log_views_per_sub"].to_numpy(dtype=float)
    y_mult_te = y_test_df["log_views_per_sub"].to_numpy(dtype=float)

    lgbm_dec = LGBMRegressor(
        n_estimators=150,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=50,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )
    lgbm_dec.fit(_prepare_lgbm_frame(X_tr_dec), y_mult_tr)
    dec_preds = np.asarray(lgbm_dec.predict(_prepare_lgbm_frame(X_te_dec)), dtype=float)
    deconfounded_metrics = compute_regression_metrics(y_mult_te, dec_preds, is_log1p=False)
    feature_importance["A_deconfounded_multiplier_gain"] = _extract_lgbm_gain_and_shap(
        lgbm_dec, X_te_dec
    )

    # --- 4. TESTE DE HIPÓTESE H5 (Regime B vs. Regime A) ---
    pr_a = classification_results["A"]["LightGBM"]["test"]["pr_auc"]
    pr_b = classification_results["B"]["LightGBM"]["test"]["pr_auc"]
    rmsle_a = regression_results["A"]["LightGBM"]["test"]["rmsle"]
    rmsle_b = regression_results["B"]["LightGBM"]["test"]["rmsle"]

    diffs_pr = np.array(fold_wise_lgbm["B"]["pr_auc"]) - np.array(fold_wise_lgbm["A"]["pr_auc"])
    diffs_rmsle = np.array(fold_wise_lgbm["B"]["rmsle"]) - np.array(fold_wise_lgbm["A"]["rmsle"])
    _, p_ttest_pr = stats.ttest_1samp(diffs_pr, 0.0)
    _, p_ttest_rmsle = stats.ttest_1samp(diffs_rmsle, 0.0)

    # Soma da importância percentual das 6 variáveis pós-upload no Regime B
    early_cols = {
        "like_rate",
        "comment_rate",
        "share_rate",
        "engagement_rate_total",
        "comments_per_like",
        "shares_per_like",
    }
    early_reg_gain_pct = sum(
        item["gain_pct"]
        for item in feature_importance["B_regression_gain"]
        if item["feature"] in early_cols
    )
    early_clf_gain_pct = sum(
        item["gain_pct"]
        for item in feature_importance["B_classification_gain"]
        if item["feature"] in early_cols
    )

    return {
        "classification": classification_results,
        "regression": regression_results,
        "deconfounded_multiplier": deconfounded_metrics,
        "feature_importance": feature_importance,
        "H5_regime_comparison": {
            "test_pr_auc_A": float(pr_a),
            "test_pr_auc_B": float(pr_b),
            "delta_test_pr_auc": float(pr_b - pr_a),
            "cv_pr_auc_diff_mean": float(np.mean(diffs_pr)),
            "cv_pr_auc_ttest_p": float(p_ttest_pr),
            "test_rmsle_A": float(rmsle_a),
            "test_rmsle_B": float(rmsle_b),
            "delta_test_rmsle": float(rmsle_b - rmsle_a),
            "cv_rmsle_diff_mean": float(np.mean(diffs_rmsle)),
            "cv_rmsle_ttest_p": float(p_ttest_rmsle),
            "early_signals_total_gain_pct_regression": float(early_reg_gain_pct),
            "early_signals_total_gain_pct_classification": float(early_clf_gain_pct),
        },
        "test_arrays": {
            "y_bin_test": y_bin_te,
            "y_reg_test": y_reg_te,
            "predictions_A": test_predictions["A"],
            "predictions_B": test_predictions["B"],
        },
    }
