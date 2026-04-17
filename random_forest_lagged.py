"""
random_forest_lagged.py

Random forest with temporal lagging, matched to bayes_model.py methodology.

Key design decisions (aligned with the Bayesian model):
  - Target: Points (total season points), not points-per-game.
  - Features: previous-season values (lagged by one season).
  - Split: hold out 2024/25 entirely. Use 2023/24 as Optuna validation
    to avoid the tuning-on-test leakage present in bayes_model.py.
  - Preprocessing: log1p on financial features, missing-value flags,
    group-wise median imputation keyed on previous league.
  - Model: one pooled RF with league one-hot. One-hot is the RF analog
    of hierarchical league effects.
  - Prediction intervals: per-tree predictions plus Gaussian noise from
    out-of-bag residuals; quantiles give a 94% predictive interval,
    matching the Bayesian 94% HDI.
  - Feature importance: permutation importance with 100 repeats gives
    mean, std and 94% CI — analog of the coefficient HDI summary.
  - SHAP values with TreeExplainer — per-prediction attribution.
  - Outputs mirror the bayes_model.py directory structure 1-to-1.
"""

from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
N_TRIALS = 40
N_PERM_REPEATS = 50
INTERVAL_PROB = 0.94
DATA_PATH = Path("dataset.ods")
OUTPUT_DIR = Path("outputs_rf_lagged")
HOLDOUT_SEASON = "2024/25"
VAL_SEASON = "2023/24"

RAW_NUMERIC_COLUMNS = ["Avg. Squad Age", "Points", "Goals For", "Goals Against"]
RAW_FINANCIAL_COLUMNS = ["Expenditure", "Income", "Squad Value"]
RAW_OPTIONAL_COLUMNS = ["Expenditure", "Income"]
RAW_REQUIRED_COLUMNS = [
    "Season", "Club", "Points", "Squad Value",
    "Avg. Squad Age", "Goals For", "Goals Against",
]
SOURCE_FEATURE_COLUMNS = [
    "Expenditure", "Income", "Squad Value", "Avg. Squad Age",
    "Goals For", "Goals Against", "Points",
]
PREV_FEATURE_COLUMNS = [f"Prev {c}" for c in SOURCE_FEATURE_COLUMNS]
PREV_FINANCIAL_COLUMNS = [f"Prev {c}" for c in RAW_FINANCIAL_COLUMNS]
PREV_OPTIONAL_COLUMNS = [f"Prev {c}" for c in RAW_OPTIONAL_COLUMNS]

KNOWN_COLUMN_ALIASES = {
    "season result": "Points",
    "points per game": "Points per game",
    "goals for": "Goals For",
    "goals against": "Goals Against",
    "avg. squad age": "Avg. Squad Age",
    "avg squad age": "Avg. Squad Age",
    "average squad age": "Avg. Squad Age",
    "squad value": "Squad Value",
    "expenditure": "Expenditure",
    "income": "Income",
    "season": "Season",
    "club": "Club",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def parse_money(value):
    """Parse strings like '€35.46m', '1.16bn', '450k', '-', '' to float."""
    if pd.isna(value):
        return np.nan
    s = str(value).strip().lower()
    if s in {"", "-", "nan", "none"}:
        return np.nan
    s = (s.replace("€", "").replace("eur", "")
         .replace(",", "").replace("bn", "b"))
    s = re.sub(r"[^0-9kmb\.\-]", "", s)
    if s == "":
        return np.nan
    mult = 1.0
    if s.endswith("k"):
        mult, s = 1e3, s[:-1]
    elif s.endswith("m"):
        mult, s = 1e6, s[:-1]
    elif s.endswith("b"):
        mult, s = 1e9, s[:-1]
    try:
        return float(s) * mult
    except ValueError:
        return np.nan


def normalize_col(name):
    norm = re.sub(r"\s+", " ", str(name).strip()).lower()
    return KNOWN_COLUMN_ALIASES.get(norm, re.sub(r"\s+", " ", str(name).strip()))


def parse_season_start_year(season):
    s = str(season).strip()
    m = re.match(r"^\s*(\d{4})\s*[/\-]", s)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d{4})", s)
    if m:
        return int(m.group(1))
    raise ValueError(f"Can't parse year from: {season}")


def normalize_club_key(name):
    return re.sub(r"\s+", " ", str(name).strip()).lower()


def slugify(text):
    slug = re.sub(r"[^a-z0-9]+", "_", str(text).strip().lower()).strip("_")
    return slug or "league"


def load_and_clean_dataset(path):
    sheets = pd.read_excel(path, sheet_name=None, engine="odf")
    dfs = []
    for sheet_name, df in sheets.items():
        df = df.copy()
        df.columns = [normalize_col(c) for c in df.columns]
        for col in RAW_OPTIONAL_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan
        missing = [c for c in RAW_REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Sheet {sheet_name} missing: {missing}")
        df["League"] = str(sheet_name).replace("_", " ").strip()
        for col in RAW_FINANCIAL_COLUMNS:
            df[col] = df[col].apply(parse_money)
        for col in RAW_NUMERIC_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        dfs.append(df)
    full = pd.concat(dfs, ignore_index=True)
    full = full.dropna(
        subset=["Points", "Squad Value", "Avg. Squad Age",
                "Goals For", "Goals Against"]
    ).reset_index(drop=True)
    return full


def build_lagged_dataset(df):
    panel = df.copy()
    panel["ClubKey"] = panel["Club"].map(normalize_club_key)
    panel["year"] = panel["Season"].map(parse_season_start_year)
    panel = panel.sort_values(["ClubKey", "year"])
    grp = panel.groupby("ClubKey", sort=False)
    panel["Prev Season"] = grp["Season"].shift(1)
    panel["Prev League"] = grp["League"].shift(1)
    panel["prev_year"] = grp["year"].shift(1)
    for col in SOURCE_FEATURE_COLUMNS:
        panel[f"Prev {col}"] = grp[col].shift(1)
    panel["gap"] = panel["year"] - panel["prev_year"]
    lagged = panel[panel["gap"] == 1].copy()
    lagged = lagged.drop(columns=["ClubKey", "year", "prev_year", "gap"])
    if lagged.empty:
        raise ValueError("No lagged rows after shifting.")
    return lagged


def split_train_val_test(df):
    train = df[(df["Season"] != HOLDOUT_SEASON) & (df["Season"] != VAL_SEASON)].copy()
    val = df[df["Season"] == VAL_SEASON].copy()
    test = df[df["Season"] == HOLDOUT_SEASON].copy()
    for name, frame in [("train", train), ("val", val), ("test", test)]:
        if frame.empty:
            raise ValueError(f"{name} split is empty. "
                             f"Check HOLDOUT_SEASON={HOLDOUT_SEASON}, VAL_SEASON={VAL_SEASON}.")
    return train, val, test


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
@dataclass
class PreprocArtifacts:
    feature_cols: list
    continuous_cols: list
    group_medians: pd.DataFrame
    global_medians: pd.Series
    imputation_group_col: str


def fit_preprocessor(train_df):
    train = train_df.copy()
    for col in PREV_OPTIONAL_COLUMNS:
        train[f"{col}_missing"] = train[col].isna().astype(int)
    for col in PREV_FINANCIAL_COLUMNS:
        train[col] = np.log1p(train[col])
    grp = train["Prev League"].fillna(train["League"])
    train["_grp"] = grp
    group_medians = train.groupby("_grp")[PREV_FEATURE_COLUMNS].median()
    global_medians = train[PREV_FEATURE_COLUMNS].median()
    for g in train["_grp"].dropna().unique():
        mask = train["_grp"] == g
        medians = group_medians.loc[g] if g in group_medians.index else global_medians
        train.loc[mask, PREV_FEATURE_COLUMNS] = (
            train.loc[mask, PREV_FEATURE_COLUMNS].fillna(medians)
        )
    train[PREV_FEATURE_COLUMNS] = train[PREV_FEATURE_COLUMNS].fillna(global_medians)
    train = train.drop(columns="_grp")
    missing_flag_cols = [f"{c}_missing" for c in PREV_OPTIONAL_COLUMNS]
    arts = PreprocArtifacts(
        feature_cols=PREV_FEATURE_COLUMNS + missing_flag_cols,
        continuous_cols=PREV_FEATURE_COLUMNS,
        group_medians=group_medians,
        global_medians=global_medians,
        imputation_group_col="Prev League",
    )
    return train, arts


def apply_preprocessor(df, arts):
    out = df.copy()
    for col in PREV_OPTIONAL_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
        out[f"{col}_missing"] = out[col].isna().astype(int)
    for col in PREV_FINANCIAL_COLUMNS:
        out[col] = np.log1p(out[col])
    grp = out["Prev League"].fillna(out["League"])
    out["_grp"] = grp
    for g in out["_grp"].dropna().unique():
        mask = out["_grp"] == g
        medians = (arts.group_medians.loc[g] if g in arts.group_medians.index
                   else arts.global_medians)
        out.loc[mask, arts.continuous_cols] = (
            out.loc[mask, arts.continuous_cols].fillna(medians)
        )
    out[arts.continuous_cols] = out[arts.continuous_cols].fillna(arts.global_medians)
    return out.drop(columns=["_grp"], errors="ignore")


# ---------------------------------------------------------------------------
# Feature matrix + tuning
# ---------------------------------------------------------------------------
def build_feature_matrix(df, arts, train_onehot_cols=None):
    oh = pd.get_dummies(df["League"], prefix="lg", dtype=float)
    if train_onehot_cols is not None:
        for c in train_onehot_cols:
            if c not in oh.columns:
                oh[c] = 0.0
        oh = oh[train_onehot_cols]
    X_base = df[arts.feature_cols].values.astype(float)
    X = np.hstack([X_base, oh.values])
    feat_names = list(arts.feature_cols) + list(oh.columns)
    return X, feat_names, oh.columns.tolist()


def tune_rf(X_train, y_train, X_val, y_val, n_trials=N_TRIALS):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_float("max_features", 0.3, 1.0),
            "max_samples": trial.suggest_float("max_samples", 0.5, 1.0),
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        }
        rf = RandomForestRegressor(**params)
        rf.fit(X_train, y_train)
        return float(np.sqrt(mean_squared_error(y_val, rf.predict(X_val))))

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study


# ---------------------------------------------------------------------------
# Prediction intervals
# ---------------------------------------------------------------------------
def predict_with_intervals(rf, X, residual_std, interval_prob=INTERVAL_PROB,
                           n_noise_samples=200):
    """
    Per-tree predictions + Gaussian noise from training residuals.
    Quantiles of the resulting distribution give a predictive interval
    that covers both epistemic (between trees) and aleatoric (noise) uncertainty.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    tree_preds = np.array([t.predict(X) for t in rf.estimators_])  # (n_trees, n_rows)
    n_trees, n_rows = tree_preds.shape
    noise = rng.normal(0.0, residual_std, size=(n_trees, n_noise_samples, n_rows))
    samples = tree_preds[:, None, :] + noise  # (n_trees, n_noise, n_rows)
    samples = samples.reshape(-1, n_rows)

    alpha_tail = (1.0 - interval_prob) / 2.0
    return {
        "mean": tree_preds.mean(axis=0),
        "sd": samples.std(axis=0, ddof=1),
        "hdi_lower": np.quantile(samples, alpha_tail, axis=0),
        "hdi_upper": np.quantile(samples, 1.0 - alpha_tail, axis=0),
        "interval_prob": interval_prob,
    }


# ---------------------------------------------------------------------------
# Permutation importance with 94% CI
# ---------------------------------------------------------------------------
def perm_importance_with_ci(model, X, y, feat_names,
                            n_repeats=N_PERM_REPEATS, interval_prob=INTERVAL_PROB):
    if len(y) < 2 or np.var(y) <= 0:
        return pd.DataFrame(columns=["feature", "importance_mean", "importance_std",
                                     "ci_lower", "ci_upper", "robust_positive"])
    perm = permutation_importance(
        model, X, y, n_repeats=n_repeats,
        random_state=RANDOM_STATE, scoring="r2", n_jobs=-1,
    )
    alpha_tail = (1.0 - interval_prob) / 2.0
    rows = []
    for i, name in enumerate(feat_names):
        imps = perm.importances[i]
        ci_lo = float(np.quantile(imps, alpha_tail))
        ci_hi = float(np.quantile(imps, 1.0 - alpha_tail))
        rows.append({
            "feature": name,
            "importance_mean": float(imps.mean()),
            "importance_std": float(imps.std(ddof=1)),
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "robust_positive": bool(ci_lo > 0),
        })
    return (pd.DataFrame(rows)
            .sort_values("importance_mean", ascending=False)
            .reset_index(drop=True))


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def save_importance_plot(df, title, path, max_display=15):
    if df.empty:
        return
    top = df.head(max_display).iloc[::-1]  # reverse so biggest at top
    fig, ax = plt.subplots(figsize=(9, max(3, 0.35 * len(top))))
    colors = ["tab:blue" if v >= 0 else "tab:red" for v in top["importance_mean"]]
    ax.barh(top["feature"], top["importance_mean"], color=colors)
    if "ci_lower" in top.columns and "ci_upper" in top.columns:
        err_lo = top["importance_mean"] - top["ci_lower"]
        err_hi = top["ci_upper"] - top["importance_mean"]
        ax.errorbar(top["importance_mean"], top["feature"],
                    xerr=[err_lo, err_hi], fmt="none", color="black", capsize=3)
    ax.axvline(0, ls="--", color="grey")
    ax.set_xlabel("Permutation importance (drop in R²)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_actual_vs_predicted(pred_df, title, path):
    if pred_df.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    err_lo = pred_df["Predicted Points"] - pred_df["Prediction HDI Lower"]
    err_hi = pred_df["Prediction HDI Upper"] - pred_df["Predicted Points"]
    ax.errorbar(pred_df["Points"], pred_df["Predicted Points"],
                yerr=[err_lo, err_hi], fmt="o", alpha=0.6, capsize=3)
    mn = min(pred_df["Points"].min(), pred_df["Predicted Points"].min())
    mx = max(pred_df["Points"].max(), pred_df["Predicted Points"].max())
    pad = 1.0 if np.isclose(mx, mn) else 0.05 * (mx - mn)
    ax.plot([mn - pad, mx + pad], [mn - pad, mx + pad], ls="--", color="black")
    ax.set_xlabel("Actual Points")
    ax.set_ylabel("Predicted Points")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_prediction_df(df, pred_summary):
    out = df[["Season", "Prev Season", "League", "Prev League", "Club", "Points"]].copy()
    out["Predicted Points"] = pred_summary["mean"]
    out["Prediction SD"] = pred_summary["sd"]
    out["Prediction HDI Lower"] = pred_summary["hdi_lower"]
    out["Prediction HDI Upper"] = pred_summary["hdi_upper"]
    out["Residual"] = out["Points"] - out["Predicted Points"]
    out["Within Prediction Interval"] = (
        (out["Points"] >= out["Prediction HDI Lower"])
        & (out["Points"] <= out["Prediction HDI Upper"])
    )
    return out


def save_shap_plots(rf, X_test, feat_names, out_dir):
    explainer = shap.TreeExplainer(rf, feature_perturbation="tree_path_dependent")
    shap_values = explainer.shap_values(X_test)

    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=feat_names,
                      show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=feat_names,
                      plot_type="bar", show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Dependence plot: top non-league feature
    mean_abs = np.abs(shap_values).mean(axis=0)
    non_lg = [i for i, f in enumerate(feat_names) if not f.startswith("lg_")]
    if non_lg:
        top = int(non_lg[np.argmax(mean_abs[non_lg])])
        plt.figure()
        shap.dependence_plot(top, shap_values, X_test,
                             feature_names=feat_names, show=False)
        plt.tight_layout()
        plt.savefig(out_dir / "shap_dependence.png", dpi=150, bbox_inches="tight")
        plt.close()

    # Save mean |SHAP| per feature as CSV for tabular view
    pd.DataFrame({
        "feature": feat_names,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).to_csv(
        out_dir / "shap_mean_abs.csv", index=False
    )


# ---------------------------------------------------------------------------
# Per-league output directory
# ---------------------------------------------------------------------------
def save_per_league_outputs(per_league_dir, rf, X_test, y_test, feat_names,
                            test_df, pred_summary):
    per_league_dir.mkdir(parents=True, exist_ok=True)

    global_imp = perm_importance_with_ci(rf, X_test, y_test, feat_names)
    global_imp.to_csv(per_league_dir / "importance_global.csv", index=False)
    save_importance_plot(global_imp,
                         "Global permutation importance (holdout)",
                         per_league_dir / "importance_global.png")

    pred_df = build_prediction_df(test_df, pred_summary)
    pred_df.to_csv(per_league_dir / "all_holdout_predictions.csv", index=False)

    metrics_rows = []
    cross_rows = []
    for league in sorted(test_df["League"].unique()):
        league_dir = per_league_dir / slugify(league)
        league_dir.mkdir(parents=True, exist_ok=True)

        mask = (test_df["League"] == league).values
        league_pred = pred_df[pred_df["League"] == league].copy()
        league_pred.to_csv(league_dir / "holdout_predictions.csv", index=False)

        y_true = league_pred["Points"].values
        y_pred = league_pred["Predicted Points"].values

        metrics = {
            "League": league,
            "holdout_rows": int(len(league_pred)),
            "rmse": (float(np.sqrt(mean_squared_error(y_true, y_pred)))
                     if len(y_true) > 0 else None),
            "mae": (float(mean_absolute_error(y_true, y_pred))
                    if len(y_true) > 0 else None),
            "r2": (float(r2_score(y_true, y_pred))
                   if len(y_true) > 1 and np.var(y_true) > 0 else None),
            "prediction_interval_coverage": (
                float(league_pred["Within Prediction Interval"].mean())
                if len(league_pred) > 0 else None
            ),
        }
        metrics_rows.append(metrics)
        with open(league_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        if len(league_pred) >= 2:
            save_actual_vs_predicted(
                league_pred, f"{league}: actual vs predicted",
                league_dir / "actual_vs_predicted.png",
            )

        if mask.sum() >= 2:
            league_imp = perm_importance_with_ci(
                rf, X_test[mask], y_test[mask], feat_names
            )
            league_imp.to_csv(league_dir / "importance.csv", index=False)
            save_importance_plot(
                league_imp,
                f"{league}: permutation importance (holdout)",
                league_dir / "importance.png",
            )
            for _, row in league_imp.iterrows():
                cross_rows.append({"League": league, **row.to_dict()})

    pd.DataFrame(metrics_rows).to_csv(
        per_league_dir / "per_league_metrics.csv", index=False
    )

    if cross_rows:
        cross = pd.DataFrame(cross_rows)
        summary = (
            cross.groupby("feature")
            .agg(
                NumLeagues=("League", "nunique"),
                NumRobustPositive=("robust_positive", "sum"),
                MeanImportance=("importance_mean", "mean"),
                MaxImportance=("importance_mean", "max"),
            )
            .reset_index()
            .sort_values(
                ["NumRobustPositive", "MeanImportance"],
                ascending=[False, False],
            )
        )
        summary.to_csv(
            per_league_dir / "cross_league_feature_summary.csv", index=False
        )


# ---------------------------------------------------------------------------
# Top-level save
# ---------------------------------------------------------------------------
def save_outputs(rf, best_params, study, prep_arts, train_df, val_df, test_df,
                 X_test, y_test, feat_names, pred_summary, oh_cols, residual_std):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    y_true = test_df["Points"].values
    y_pred = pred_summary["mean"]
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    pred_df = build_prediction_df(test_df, pred_summary)
    coverage = float(pred_df["Within Prediction Interval"].mean())
    pred_df.to_csv(OUTPUT_DIR / "holdout_predictions.csv", index=False)
    save_actual_vs_predicted(
        pred_df, "All leagues: actual vs predicted",
        OUTPUT_DIR / "actual_vs_predicted.png",
    )

    diagnostics = {
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "holdout_rows": int(len(test_df)),
        "holdout_season": HOLDOUT_SEASON,
        "val_season": VAL_SEASON,
        "predictor_timing": "previous_season",
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "prediction_interval_coverage": coverage,
        "prediction_interval_probability": INTERVAL_PROB,
        "oob_residual_std": residual_std,
    }
    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(diagnostics, f, indent=2)

    preprocessing_summary = {
        "holdout_season": HOLDOUT_SEASON,
        "val_season": VAL_SEASON,
        "predictor_timing": "previous_season",
        "feature_cols": prep_arts.feature_cols,
        "continuous_cols": prep_arts.continuous_cols,
        "imputation_group_col": prep_arts.imputation_group_col,
        "league_dummies": list(oh_cols),
    }
    with open(OUTPUT_DIR / "preprocessing_summary.json", "w") as f:
        json.dump(preprocessing_summary, f, indent=2)

    with open(OUTPUT_DIR / "optuna_best_params.json", "w") as f:
        json.dump({
            "best_value": study.best_value,
            "best_params": best_params,
            "best_trial": study.best_trial.number,
        }, f, indent=2)
    study.trials_dataframe().to_csv(OUTPUT_DIR / "optuna_trials.csv", index=False)

    save_shap_plots(rf, X_test, feat_names, OUTPUT_DIR)
    save_per_league_outputs(
        OUTPUT_DIR / "per_league",
        rf, X_test, y_test, feat_names, test_df, pred_summary,
    )

    report_lines = [
        "Random forest report (methodology aligned with bayes_model.py)",
        "Predictor timing: previous season",
        f"Holdout season: {HOLDOUT_SEASON}",
        f"Val season (used by Optuna): {VAL_SEASON}",
        f"Train rows: {len(train_df)}",
        f"Val rows: {len(val_df)}",
        f"Holdout rows: {len(test_df)}",
        "",
        "Overall holdout metrics",
        f"RMSE: {rmse:.4f}",
        f"MAE: {mae:.4f}",
        f"R²: {r2:.4f}",
        f"Prediction interval coverage ({INTERVAL_PROB:.0%}): {coverage:.4f}",
        "",
        "Best Optuna params:",
        json.dumps(best_params, indent=2),
        "",
        "Per-league outputs: outputs_rf_lagged/per_league/",
        "SHAP plots:        outputs_rf_lagged/shap_*.png",
    ]
    (OUTPUT_DIR / "report.txt").write_text("\n".join(report_lines) + "\n")

    return diagnostics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(RANDOM_STATE)

    print("Loading & cleaning data...")
    df = load_and_clean_dataset(DATA_PATH)
    print(f"  {len(df)} rows across {df['League'].nunique()} leagues")

    print("Building lagged club-season dataset...")
    lagged = build_lagged_dataset(df)
    print(f"  {len(lagged)} lagged rows")

    print(f"Splitting train / val ({VAL_SEASON}) / test ({HOLDOUT_SEASON})...")
    train_raw, val_raw, test_raw = split_train_val_test(lagged)
    print(f"  train={len(train_raw)} val={len(val_raw)} test={len(test_raw)}")

    print("Fitting preprocessor on train only...")
    train_pp, prep_arts = fit_preprocessor(train_raw)
    val_pp = apply_preprocessor(val_raw, prep_arts)
    test_pp = apply_preprocessor(test_raw, prep_arts)

    X_train, feat_names, oh_cols = build_feature_matrix(train_pp, prep_arts)
    X_val, _, _ = build_feature_matrix(val_pp, prep_arts, train_onehot_cols=oh_cols)
    X_test, _, _ = build_feature_matrix(test_pp, prep_arts, train_onehot_cols=oh_cols)
    y_train = train_pp["Points"].values
    y_val = val_pp["Points"].values
    y_test = test_pp["Points"].values

    print(f"Running Optuna ({N_TRIALS} trials), validating on {VAL_SEASON}...")
    study = tune_rf(X_train, y_train, X_val, y_val)
    best_params = study.best_params
    print(f"  Best val RMSE: {study.best_value:.4f}")
    print(f"  Best params:   {best_params}")

    print("Refitting best model on train + val...")
    X_fit = np.vstack([X_train, X_val])
    y_fit = np.concatenate([y_train, y_val])
    rf = RandomForestRegressor(
        **best_params, random_state=RANDOM_STATE, n_jobs=-1,
        oob_score=True, bootstrap=True,
    )
    rf.fit(X_fit, y_fit)

    oob_resid = y_fit - rf.oob_prediction_
    residual_std = float(np.std(oob_resid, ddof=1))
    print(f"  OOB residual SD: {residual_std:.3f}")

    print(f"Generating predictions with {INTERVAL_PROB:.0%} intervals...")
    pred_summary = predict_with_intervals(rf, X_test, residual_std=residual_std)

    print("Saving outputs (+ SHAP, permutation importance, per-league)...")
    diagnostics = save_outputs(
        rf, best_params, study, prep_arts, train_pp, val_pp, test_pp,
        X_test, y_test, feat_names, pred_summary, oh_cols, residual_std,
    )

    print("\n=== HOLDOUT METRICS ===")
    for k, v in diagnostics.items():
        print(f"  {k}: {v}")
    print(f"\nAll outputs saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
