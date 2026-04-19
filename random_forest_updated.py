"""
random_forest_lagged.py

Random forest with temporal lagging, matched to bayes_model.py methodology.

Key design decisions (aligned with the Bayesian model):
  - Target: Points (total season points), not points-per-game.
  - Features: previous-season values (lagged by one season).
  - Split: hold out 2023/24 and 2024/25 entirely as test (no row whose
    target Season is in the holdout set is ever seen in training or
    tuning). Walk-forward (expanding window) cross-validation by
    season inside Optuna — each trial's score is the mean validation
    RMSE across folds, where fold k trains on all seasons < S_k and
    validates on S_k. Uses every pre-holdout season for tuning instead
    of pinning the choice to one val year, while still avoiding the
    tuning-on-test leakage present in bayes_model.py.
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
  - Raw results: model-native outputs (MDI importance, OOB diagnostics,
    residual structure, prediction-bucket matrix) saved under
    outputs_rf_lagged/raw_results/. SHAP standardises across models;
    this directory preserves the detail SHAP averages away.
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
from scipy import stats as sps
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
# Test set: both of these seasons are held out entirely from training
# and tuning. Pooled into one test set for overall metrics, and also
# broken out per season in outputs_rf_lagged/per_season/.
HOLDOUT_SEASONS = ["2023/24", "2024/25"]
# Walk-forward CV: need at least this many seasons of training history
# before the first validation fold.
CV_MIN_TRAIN_SEASONS = 2

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


def split_dev_test(df):
    """
    Development set = everything pre-holdout (used for walk-forward CV
    hyperparameter tuning AND the final refit). Test set = all rows
    whose Season is in HOLDOUT_SEASONS. Never touched until evaluation.
    """
    dev = df[~df["Season"].isin(HOLDOUT_SEASONS)].copy()
    test = df[df["Season"].isin(HOLDOUT_SEASONS)].copy()
    if dev.empty:
        raise ValueError("Dev split is empty.")
    if test.empty:
        raise ValueError(f"Test split is empty. HOLDOUT_SEASONS={HOLDOUT_SEASONS}")
    missing = [s for s in HOLDOUT_SEASONS if not (test["Season"] == s).any()]
    if missing:
        raise ValueError(f"Holdout seasons missing from data: {missing}")
    return dev, test


def make_walk_forward_folds(dev_df, min_train_seasons=CV_MIN_TRAIN_SEASONS):
    """
    Build walk-forward (expanding-window) folds ordered by season.
    Fold k: train on all seasons strictly before S_k, validate on S_k.
    The first usable validation season is the one that has at least
    `min_train_seasons` preceding seasons in the dev set.

    Returns list of (train_mask, val_mask, val_season_label) tuples.
    """
    years = dev_df["Season"].map(parse_season_start_year).values
    seasons_by_year = (
        dev_df[["Season"]].assign(_year=years)
        .drop_duplicates("_year").sort_values("_year")
    )
    ordered_years = seasons_by_year["_year"].tolist()
    ordered_labels = seasons_by_year["Season"].tolist()

    folds = []
    for i, (val_year, val_label) in enumerate(zip(ordered_years, ordered_labels)):
        if i < min_train_seasons:
            continue
        train_mask = (years < val_year)
        val_mask = (years == val_year)
        if val_mask.sum() == 0 or train_mask.sum() == 0:
            continue
        folds.append((train_mask, val_mask, val_label))

    if not folds:
        raise ValueError(
            f"No walk-forward folds built. Need >= {min_train_seasons + 1} "
            f"seasons in dev set; got {len(ordered_years)} "
            f"({ordered_labels})."
        )
    return folds


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


def tune_rf_walk_forward(X_dev, y_dev, folds, n_trials=N_TRIALS):
    """
    Optuna tuning with walk-forward CV by season.

    Each trial's objective is the mean validation RMSE across all folds.
    With 3 folds and 40 trials this is ~120 RF fits (vs 40 in single-season
    tuning), but the hyperparameter choice is no longer hostage to one
    validation year.
    """
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
        fold_rmses = []
        for train_mask, val_mask, _ in folds:
            rf = RandomForestRegressor(**params)
            rf.fit(X_dev[train_mask], y_dev[train_mask])
            preds = rf.predict(X_dev[val_mask])
            fold_rmses.append(
                float(np.sqrt(mean_squared_error(y_dev[val_mask], preds)))
            )
        # Record per-fold RMSE on the trial for later inspection
        trial.set_user_attr("fold_rmses", fold_rmses)
        return float(np.mean(fold_rmses))

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
# Raw / native model results
# ---------------------------------------------------------------------------
# These outputs preserve the per-model detail that SHAP averages away.
# India: SHAP standardises across models; we still want each model's
# native output somewhere. Confusion matrix doesn't fit regression, so
# we use residual diagnostics + a points-quartile bucket matrix as the
# common "raw data representation" across models.
# ---------------------------------------------------------------------------
def save_native_importance_plot(mdi_df, path, max_display=15):
    if mdi_df.empty:
        return
    top = mdi_df.head(max_display).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(3, 0.35 * len(top))))
    ax.barh(top["feature"], top["mdi_importance"], color="tab:green")
    ax.set_xlabel("MDI importance (native RF)")
    ax.set_title("Native RF feature importance (MDI / Gini)")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_residual_diagnostics(y_true, y_pred, path):
    """Regression analog of a confusion matrix: residual structure."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    resid = y_true - y_pred
    if len(resid) < 2:
        return

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    axes[0, 0].hist(resid, bins=min(20, max(5, len(resid) // 3)),
                    color="steelblue", edgecolor="black")
    axes[0, 0].axvline(0, ls="--", color="red")
    axes[0, 0].set_title("Residual distribution")
    axes[0, 0].set_xlabel("Residual (actual - predicted)")
    axes[0, 0].set_ylabel("Frequency")

    axes[0, 1].scatter(y_pred, resid, alpha=0.6)
    axes[0, 1].axhline(0, ls="--", color="red")
    axes[0, 1].set_title("Residuals vs predicted")
    axes[0, 1].set_xlabel("Predicted points")
    axes[0, 1].set_ylabel("Residual")

    axes[1, 0].scatter(y_true, resid, alpha=0.6)
    axes[1, 0].axhline(0, ls="--", color="red")
    axes[1, 0].set_title("Residuals vs actual")
    axes[1, 0].set_xlabel("Actual points")
    axes[1, 0].set_ylabel("Residual")

    sps.probplot(resid, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title("Residual QQ plot")

    fig.suptitle("Residual diagnostics (regression analog of confusion matrix)")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_residuals_by_league(pred_df, path):
    if pred_df.empty:
        return
    leagues = sorted(pred_df["League"].unique())
    data = [pred_df[pred_df["League"] == lg]["Residual"].values for lg in leagues]
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(leagues)), 5))
    ax.boxplot(data, labels=leagues)
    ax.axhline(0, ls="--", color="red")
    ax.set_ylabel("Residual (actual - predicted)")
    ax.set_title("Residuals by league (holdout)")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_prediction_bucket_matrix(y_true, y_pred, edges, png_path, csv_path):
    """
    Cross-model equivalent of a confusion matrix for regression output.
    Bins continuous points into quartiles (edges computed on training
    data so it's identical across models) and counts actual-vs-predicted
    bucket hits. Same shape can be produced for any regression model.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        return
    n_buckets = len(edges) - 1
    labels = [f"Q{i + 1}" for i in range(n_buckets)]
    bins_true = np.clip(np.digitize(y_true, edges[1:-1]), 0, n_buckets - 1)
    bins_pred = np.clip(np.digitize(y_pred, edges[1:-1]), 0, n_buckets - 1)

    matrix = np.zeros((n_buckets, n_buckets), dtype=int)
    for bt, bp in zip(bins_true, bins_pred):
        matrix[bt, bp] += 1

    matrix_df = pd.DataFrame(
        matrix,
        index=[f"actual_{l}" for l in labels],
        columns=[f"pred_{l}" for l in labels],
    )
    matrix_df.to_csv(csv_path)

    edges_path = Path(str(csv_path).replace(".csv", "_edges.csv"))
    pd.DataFrame({
        "quartile": labels,
        "lower_edge": edges[:-1],
        "upper_edge": edges[1:],
    }).to_csv(edges_path, index=False)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(n_buckets))
    ax.set_yticks(range(n_buckets))
    ax.set_xticklabels([f"pred {l}" for l in labels])
    ax.set_yticklabels([f"actual {l}" for l in labels])
    ax.set_xlabel("Predicted points quartile")
    ax.set_ylabel("Actual points quartile")
    ax.set_title("Prediction bucket matrix (training-quartile bins)")
    vmax = matrix.max() if matrix.max() > 0 else 1
    for i in range(n_buckets):
        for j in range(n_buckets):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > vmax / 2 else "black")
    fig.colorbar(im, ax=ax, label="Count")
    fig.tight_layout()
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_raw_results(out_dir, rf, dev_df, test_df, pred_df,
                     feat_names, pred_summary):
    """
    Per-model raw outputs. Mirror this function's output shape across
    every model file so cross-model comparison is mechanical.
    """
    raw_dir = out_dir / "raw_results"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Native RF feature importance (MDI) ------------------------------
    mdi = pd.DataFrame({
        "feature": feat_names,
        "mdi_importance": rf.feature_importances_,
    }).sort_values("mdi_importance", ascending=False).reset_index(drop=True)
    mdi.to_csv(raw_dir / "native_feature_importance.csv", index=False)
    save_native_importance_plot(
        mdi, raw_dir / "native_feature_importance.png",
    )

    # --- 2. OOB diagnostics (training-side, native to RF) -------------------
    fit_df = dev_df.reset_index(drop=True)
    y_fit = fit_df["Points"].values.astype(float)
    oob_pred = np.asarray(rf.oob_prediction_, dtype=float)
    oob_resid = y_fit - oob_pred

    oob_df = fit_df[["Season", "Prev Season", "League", "Prev League",
                     "Club", "Points"]].copy()
    oob_df["OOB Predicted Points"] = oob_pred
    oob_df["OOB Residual"] = oob_resid
    oob_df.to_csv(raw_dir / "oob_predictions.csv", index=False)

    # R² on OOB predictions, guarded against zero-variance
    oob_r2 = (float(r2_score(y_fit, oob_pred))
              if np.var(y_fit) > 0 else None)

    oob_summary = {
        "n_fit_rows": int(len(y_fit)),
        "oob_score_r2": oob_r2,
        "oob_rmse": float(np.sqrt(np.mean(oob_resid ** 2))),
        "oob_mae": float(np.mean(np.abs(oob_resid))),
        "oob_residual_std": float(np.std(oob_resid, ddof=1)),
        "oob_residual_mean_bias": float(np.mean(oob_resid)),
    }
    with open(raw_dir / "oob_summary.json", "w") as f:
        json.dump(oob_summary, f, indent=2)

    # --- 3. Residual diagnostics (holdout) ----------------------------------
    y_true = test_df["Points"].values.astype(float)
    y_pred = np.asarray(pred_summary["mean"], dtype=float)
    save_residual_diagnostics(
        y_true, y_pred, raw_dir / "residual_diagnostics.png",
    )
    save_residuals_by_league(
        pred_df, raw_dir / "residuals_by_league.png",
    )

    # --- 4. Prediction bucket matrix (cross-model comparable) ---------------
    # Edges fixed on training data so the same buckets are used across
    # every model file — this is the "something else to represent the
    # raw data for all the models" from the Telegram chat.
    edges = np.quantile(y_fit, [0.0, 0.25, 0.5, 0.75, 1.0])
    # Guard against duplicate edges (e.g. many zero-point teams).
    edges = np.unique(edges)
    if len(edges) >= 3:
        save_prediction_bucket_matrix(
            y_true, y_pred, edges,
            raw_dir / "prediction_bucket_matrix.png",
            raw_dir / "prediction_bucket_matrix.csv",
        )

    # --- 5. Canonical raw-metrics JSON (identical schema per model) ---------
    resid_test = y_true - y_pred
    mask_nz = y_true != 0
    mape = (float(np.mean(np.abs(resid_test[mask_nz] / y_true[mask_nz])) * 100)
            if mask_nz.any() else None)
    coverage = float(
        ((y_true >= pred_summary["hdi_lower"])
         & (y_true <= pred_summary["hdi_upper"])).mean()
    )
    raw_metrics = {
        "model": "random_forest_lagged",
        "target": "Points",
        "predictor_timing": "previous_season",
        "n_fit": int(len(y_fit)),
        "n_test": int(len(y_true)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": (float(r2_score(y_true, y_pred))
               if np.var(y_true) > 0 else None),
        "mape_pct": mape,
        "bias_mean_error": float(resid_test.mean()),
        "residual_std": float(resid_test.std(ddof=1)),
        "interval_prob": INTERVAL_PROB,
        "interval_coverage": coverage,
        "oob_score_r2": oob_r2,
    }
    with open(raw_dir / "raw_metrics.json", "w") as f:
        json.dump(raw_metrics, f, indent=2)


# ---------------------------------------------------------------------------
# Per-test-season output directory
# ---------------------------------------------------------------------------
def save_per_season_outputs(per_season_dir, pred_df, holdout_seasons):
    """
    Breakdown of holdout performance by test season. With a multi-season
    holdout (e.g. both 2023/24 and 2024/25) this shows whether the model
    degrades on the more recent season and lets the group compare the
    two years directly.
    """
    per_season_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows = []
    for season in holdout_seasons:
        season_pred = pred_df[pred_df["Season"] == season].copy()
        season_dir = per_season_dir / slugify(season)
        season_dir.mkdir(parents=True, exist_ok=True)
        season_pred.to_csv(season_dir / "holdout_predictions.csv", index=False)

        y_true = season_pred["Points"].values
        y_pred = season_pred["Predicted Points"].values
        resid = y_true - y_pred
        mask_nz = y_true != 0
        mape = (float(np.mean(np.abs(resid[mask_nz] / y_true[mask_nz])) * 100)
                if mask_nz.any() else None)

        metrics = {
            "season": season,
            "holdout_rows": int(len(season_pred)),
            "rmse": (float(np.sqrt(mean_squared_error(y_true, y_pred)))
                     if len(y_true) > 0 else None),
            "mae": (float(mean_absolute_error(y_true, y_pred))
                    if len(y_true) > 0 else None),
            "r2": (float(r2_score(y_true, y_pred))
                   if len(y_true) > 1 and np.var(y_true) > 0 else None),
            "mape_pct": mape,
            "bias_mean_error": (float(resid.mean())
                                if len(resid) > 0 else None),
            "prediction_interval_coverage": (
                float(season_pred["Within Prediction Interval"].mean())
                if len(season_pred) > 0 else None
            ),
        }
        metrics_rows.append(metrics)
        with open(season_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        if len(season_pred) >= 2:
            save_actual_vs_predicted(
                season_pred, f"{season}: actual vs predicted",
                season_dir / "actual_vs_predicted.png",
            )

    pd.DataFrame(metrics_rows).to_csv(
        per_season_dir / "per_season_metrics.csv", index=False
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
def save_outputs(rf, best_params, study, prep_arts, dev_df, test_df, folds,
                 X_test, y_test, feat_names, pred_summary, oh_cols, residual_std):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    y_true = test_df["Points"].values
    y_pred = pred_summary["mean"]
    resid = y_true - y_pred
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mask_nz = y_true != 0
    mape = (float(np.mean(np.abs(resid[mask_nz] / y_true[mask_nz])) * 100)
            if mask_nz.any() else None)
    bias = float(resid.mean())

    pred_df = build_prediction_df(test_df, pred_summary)
    coverage = float(pred_df["Within Prediction Interval"].mean())
    pred_df.to_csv(OUTPUT_DIR / "holdout_predictions.csv", index=False)
    save_actual_vs_predicted(
        pred_df, "All leagues: actual vs predicted",
        OUTPUT_DIR / "actual_vs_predicted.png",
    )

    # Per-fold Optuna performance for the best trial (useful diagnostic)
    best_fold_rmses = study.best_trial.user_attrs.get("fold_rmses", [])
    cv_fold_val_seasons = [f[2] for f in folds]

    diagnostics = {
        "dev_rows": int(len(dev_df)),
        "holdout_rows": int(len(test_df)),
        "holdout_seasons": list(HOLDOUT_SEASONS),
        "cv_strategy": "walk_forward_expanding_window",
        "cv_fold_val_seasons": cv_fold_val_seasons,
        "cv_min_train_seasons": CV_MIN_TRAIN_SEASONS,
        "cv_best_fold_rmses": best_fold_rmses,
        "cv_best_mean_rmse": float(study.best_value),
        "predictor_timing": "previous_season",
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape_pct": mape,
        "bias_mean_error": bias,
        "prediction_interval_coverage": coverage,
        "prediction_interval_probability": INTERVAL_PROB,
        "oob_residual_std": residual_std,
    }
    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(diagnostics, f, indent=2)

    preprocessing_summary = {
        "holdout_seasons": list(HOLDOUT_SEASONS),
        "cv_strategy": "walk_forward_expanding_window",
        "cv_fold_val_seasons": cv_fold_val_seasons,
        "cv_min_train_seasons": CV_MIN_TRAIN_SEASONS,
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
            "cv_fold_val_seasons": cv_fold_val_seasons,
            "best_trial_fold_rmses": best_fold_rmses,
        }, f, indent=2)
    study.trials_dataframe().to_csv(OUTPUT_DIR / "optuna_trials.csv", index=False)

    save_shap_plots(rf, X_test, feat_names, OUTPUT_DIR)
    save_raw_results(
        OUTPUT_DIR, rf, dev_df, test_df, pred_df,
        feat_names, pred_summary,
    )
    save_per_season_outputs(
        OUTPUT_DIR / "per_season", pred_df, HOLDOUT_SEASONS,
    )
    save_per_league_outputs(
        OUTPUT_DIR / "per_league",
        rf, X_test, y_test, feat_names, test_df, pred_summary,
    )

    fold_lines = [
        f"  fold {i}: val_season={lbl}, rmse={r:.4f}"
        for i, (lbl, r) in enumerate(zip(cv_fold_val_seasons, best_fold_rmses), 1)
    ] if best_fold_rmses else ["  (no per-fold RMSE recorded)"]

    report_lines = [
        "Random forest report (methodology aligned with bayes_model.py)",
        "Predictor timing: previous season",
        f"Holdout seasons: {list(HOLDOUT_SEASONS)}",
        "CV strategy:   walk-forward (expanding window) by season",
        f"CV min train seasons: {CV_MIN_TRAIN_SEASONS}",
        f"CV fold val seasons: {cv_fold_val_seasons}",
        f"Dev rows: {len(dev_df)}",
        f"Holdout rows: {len(test_df)}",
        "",
        "Best-trial per-fold RMSE (walk-forward CV):",
        *fold_lines,
        f"Mean CV RMSE (Optuna objective): {study.best_value:.4f}",
        "",
        "Overall holdout metrics (pooled across all test seasons)",
        f"RMSE: {rmse:.4f}",
        f"MAE: {mae:.4f}",
        f"R²: {r2:.4f}",
        f"MAPE: {mape:.2f}%" if mape is not None else "MAPE: n/a",
        f"Mean bias (actual - pred): {bias:+.4f}",
        f"Prediction interval coverage ({INTERVAL_PROB:.0%}): {coverage:.4f}",
        "",
        "Best Optuna params:",
        json.dumps(best_params, indent=2),
        "",
        "Per-season outputs: outputs_rf_lagged/per_season/",
        "Per-league outputs: outputs_rf_lagged/per_league/",
        "SHAP plots:        outputs_rf_lagged/shap_*.png",
        "Raw / native model outputs: outputs_rf_lagged/raw_results/",
        "  - native_feature_importance.csv/.png  (RF MDI, the native",
        "    analog of regression coefficients)",
        "  - oob_summary.json, oob_predictions.csv  (training-side eval",
        "    native to RF)",
        "  - residual_diagnostics.png  (regression analog of confusion",
        "    matrix: histogram, residuals vs predicted/actual, QQ plot)",
        "  - residuals_by_league.png   (where the model fails, by league)",
        "  - prediction_bucket_matrix.csv/.png  (actual vs predicted",
        "    quartile buckets — mirror this across all models for",
        "    consistent cross-model comparison)",
        "  - raw_metrics.json  (canonical per-model metrics schema)",
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

    print(f"Splitting dev / test ({HOLDOUT_SEASONS})...")
    dev_raw, test_raw = split_dev_test(lagged)
    print(f"  dev={len(dev_raw)} test={len(test_raw)}")

    print("Fitting preprocessor on dev only...")
    dev_pp, prep_arts = fit_preprocessor(dev_raw)
    test_pp = apply_preprocessor(test_raw, prep_arts)

    X_dev, feat_names, oh_cols = build_feature_matrix(dev_pp, prep_arts)
    X_test, _, _ = build_feature_matrix(test_pp, prep_arts, train_onehot_cols=oh_cols)
    y_dev = dev_pp["Points"].values
    y_test = test_pp["Points"].values

    print(f"Building walk-forward folds (min_train_seasons={CV_MIN_TRAIN_SEASONS})...")
    folds = make_walk_forward_folds(dev_pp)
    print(f"  {len(folds)} folds:")
    for i, (tr_m, va_m, lbl) in enumerate(folds, 1):
        print(f"    fold {i}: train n={tr_m.sum()}, val={lbl} n={va_m.sum()}")

    print(f"Running Optuna ({N_TRIALS} trials) with walk-forward CV...")
    study = tune_rf_walk_forward(X_dev, y_dev, folds)
    best_params = study.best_params
    print(f"  Best mean-CV RMSE: {study.best_value:.4f}")
    print(f"  Best params:      {best_params}")

    print("Refitting best model on full dev set...")
    rf = RandomForestRegressor(
        **best_params, random_state=RANDOM_STATE, n_jobs=-1,
        oob_score=True, bootstrap=True,
    )
    rf.fit(X_dev, y_dev)

    oob_resid = y_dev - rf.oob_prediction_
    residual_std = float(np.std(oob_resid, ddof=1))
    print(f"  OOB residual SD: {residual_std:.3f}")

    print(f"Generating predictions with {INTERVAL_PROB:.0%} intervals...")
    pred_summary = predict_with_intervals(rf, X_test, residual_std=residual_std)

    print("Saving outputs (+ SHAP, permutation importance, per-league)...")
    diagnostics = save_outputs(
        rf, best_params, study, prep_arts, dev_pp, test_pp, folds,
        X_test, y_test, feat_names, pred_summary, oh_cols, residual_std,
    )

    print("\n=== HOLDOUT METRICS ===")
    for k, v in diagnostics.items():
        print(f"  {k}: {v}")
    print(f"\nAll outputs saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
