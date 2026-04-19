"""
Elastic net regression with lagged predictors, per league.

Each row predicts a club's current-season points per game using that same
club's previous-season features. A separate model is fitted per league, with
hyperparameters selected via cross-validation on training data only.
"""

import json
import re
import warnings
from dataclasses import dataclass

import optuna
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from pipeline_config import INPUT_PATH, VAL_SEASON, build_run_config, parse_mode_args, save_run_config

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

RAW_REQUIRED_COLUMNS = [
    "Season",
    "Club",
    "Points",
    "Squad Value",
    "Avg. Squad Age",
    "Goals For",
    "Goals Against",
]
RAW_OPTIONAL_COLUMNS = ["Expenditure", "Income"]
RAW_NUMERIC_COLUMNS = ["Avg. Squad Age", "Points", "Goals For", "Goals Against"]
RAW_FINANCIAL_COLUMNS = ["Expenditure", "Income", "Squad Value"]

SOURCE_FEATURE_COLUMNS = [
    "Expenditure",
    "Income",
    "Squad Value",
    "Avg. Squad Age",
    "Goals For",
    "Goals Against",
    "Points",
]
PREVIOUS_FEATURE_COLUMNS = [f"Prev {col}" for col in SOURCE_FEATURE_COLUMNS]
PREVIOUS_FINANCIAL_COLUMNS = [f"Prev {col}" for col in RAW_FINANCIAL_COLUMNS]
PREVIOUS_OPTIONAL_COLUMNS = [f"Prev {col}" for col in RAW_OPTIONAL_COLUMNS]

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

ALPHA_LOW = 1e-4
ALPHA_HIGH = 1e2


# data loading 

def parse_money(value):
    if pd.isna(value):
        return np.nan
    s = str(value).strip().lower()
    if s in {"", "-", "nan", "none"}:
        return np.nan
    s = (
        s.replace("€", "")
        .replace("â‚¬", "")
        .replace("eur", "")
        .replace(",", "")
        .replace("bn", "b")
    )
    s = re.sub(r"[^0-9kmb\.\-]", "", s)
    if s == "":
        return np.nan
    multiplier = 1.0
    if s.endswith("k"):
        multiplier = 1_000.0
        s = s[:-1]
    elif s.endswith("m"):
        multiplier = 1_000_000.0
        s = s[:-1]
    elif s.endswith("b"):
        multiplier = 1_000_000_000.0
        s = s[:-1]
    try:
        return float(s) * multiplier
    except ValueError:
        return np.nan


def normalize_column_name(column_name):
    normalized = re.sub(r"\s+", " ", str(column_name).strip()).lower()
    return KNOWN_COLUMN_ALIASES.get(normalized, re.sub(r"\s+", " ", str(column_name).strip()))


def normalize_sheet_columns(df):
    renamed = df.copy()
    renamed.columns = [normalize_column_name(c) for c in renamed.columns]
    return renamed


def normalize_club_key(club_name):
    return re.sub(r"\s+", " ", str(club_name).strip()).lower()


def slugify_filename(value):
    slug = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower())
    return slug.strip("_") or "league"


def parse_season_start_year(season_value):
    season_str = str(season_value).strip()
    match = re.match(r"^\s*(\d{4})\s*[/\-]", season_str)
    if match:
        return int(match.group(1))
    fallback = re.search(r"(\d{4})", season_str)
    if fallback:
        return int(fallback.group(1))
    raise ValueError(f"Could not parse season start year from: {season_value}")


def ensure_expected_columns(df, sheet_name):
    frame = df.copy()
    for column in RAW_OPTIONAL_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    missing_required = [c for c in RAW_REQUIRED_COLUMNS if c not in frame.columns]
    if missing_required:
        raise ValueError(
            f"Sheet '{sheet_name}' is missing required columns after normalization: "
            f"{missing_required}"
        )
    return frame


def load_and_clean_dataset(path):
    sheet_dict = pd.read_excel(path, sheet_name=None, engine="odf")
    dfs = []
    for sheet_name, df in sheet_dict.items():
        temp = normalize_sheet_columns(df)
        temp = ensure_expected_columns(temp, sheet_name)
        temp["League"] = str(sheet_name).replace("_", " ").strip()
        for col in RAW_FINANCIAL_COLUMNS:
            temp[col] = temp[col].apply(parse_money)
        for col in RAW_NUMERIC_COLUMNS:
            temp[col] = pd.to_numeric(temp[col], errors="coerce")
        dfs.append(temp)
    df = pd.concat(dfs, ignore_index=True)
    df = df.dropna(
        subset=["Points", "Squad Value", "Avg. Squad Age", "Goals For", "Goals Against"]
    ).copy()
    return df


# lagged dataset

def build_lagged_dataset(df):
    """
    Produce one row per club-season where predictors are the previous season's
    values for the same club. Rows where the previous season is not consecutive
    (e.g. a gap year) are dropped.
    """
    panel = df.copy()
    panel["ClubKey"] = panel["Club"].map(normalize_club_key)
    panel["season_start_year"] = panel["Season"].map(parse_season_start_year)
    panel = panel.sort_values(["ClubKey", "season_start_year"]).copy()
    grouped = panel.groupby("ClubKey", sort=False)

    panel["Prev Season"] = grouped["Season"].shift(1)
    panel["Prev League"] = grouped["League"].shift(1)
    panel["prev_season_start_year"] = grouped["season_start_year"].shift(1)

    for col in SOURCE_FEATURE_COLUMNS:
        panel[f"Prev {col}"] = grouped[col].shift(1)

    panel["season_gap"] = panel["season_start_year"] - panel["prev_season_start_year"]
    lagged = panel[panel["season_gap"] == 1].copy()

    if lagged.empty:
        raise ValueError("No lagged rows were created. Check season formatting and club names.")

    lagged = lagged.drop(
        columns=["ClubKey", "season_start_year", "prev_season_start_year", "season_gap"]
    )
    return lagged


def split_train_val_holdout(df, val_season, holdout_season):
    train = df[(df["Season"] != val_season) & (df["Season"] != holdout_season)].copy()
    val = df[df["Season"] == val_season].copy()
    holdout = df[df["Season"] == holdout_season].copy()
    if train.empty:
        raise ValueError("Training split is empty.")
    if val.empty:
        raise ValueError(f"Val split is empty for season '{val_season}'.")
    if holdout.empty:
        raise ValueError(f"Holdout split is empty for season '{holdout_season}'.")
    return train, val, holdout


# --- preprocessing ---

@dataclass
class ImputationArtifacts:
    continuous_cols: list
    group_medians: pd.DataFrame
    global_medians: pd.Series
    imputation_group_col: str = "Prev League"


def fit_imputation_artifacts(train_df):
    """
    Compute group medians and global medians from the full training set.
    Groups by Prev League so that promoted/relegated clubs are matched to
    the division their lagged predictors came from.
    """
    frame = train_df.copy()

    for col in PREVIOUS_FINANCIAL_COLUMNS:
        if col in frame.columns:
            frame[col] = np.log1p(frame[col].clip(lower=0))

    group_keys = frame["Prev League"].fillna(frame["League"])
    frame["_group_key"] = group_keys

    group_medians = frame.groupby("_group_key")[PREVIOUS_FEATURE_COLUMNS].median()
    global_medians = frame[PREVIOUS_FEATURE_COLUMNS].median()

    return ImputationArtifacts(
        continuous_cols=PREVIOUS_FEATURE_COLUMNS,
        group_medians=group_medians,
        global_medians=global_medians,
    )


def build_feature_cols():
    """Base lagged features plus missingness flags for optional columns."""
    flag_cols = [f"{col}_missing" for col in PREVIOUS_OPTIONAL_COLUMNS]
    return PREVIOUS_FEATURE_COLUMNS + flag_cols


def apply_transforms(frame, feature_cols, imputation_artifacts, fit_scaler=None):
    """
    1. Add binary missingness flags for optional columns.
    2. Apply log1p to financial columns (clips negatives to 0 first).
    3. Impute with group medians fitted on training data (grouped by Prev League).
    4. Standardise (fit on training data).

    Pass fit_scaler from training to apply the same standardisation to holdout
    without re-fitting.
    """
    df = frame.copy()

    for col in PREVIOUS_OPTIONAL_COLUMNS:
        flag = f"{col}_missing"
        df[flag] = df[col].isna().astype(float) if col in df.columns else 0.0

    for col in PREVIOUS_FINANCIAL_COLUMNS:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))

    group_keys = df[imputation_artifacts.imputation_group_col].fillna(df["League"])
    df["_group_key"] = group_keys

    for gk in df["_group_key"].dropna().unique():
        idx = df["_group_key"] == gk
        medians = (
            imputation_artifacts.group_medians.loc[gk]
            if gk in imputation_artifacts.group_medians.index
            else imputation_artifacts.global_medians
        )
        df.loc[idx, imputation_artifacts.continuous_cols] = (
            df.loc[idx, imputation_artifacts.continuous_cols].fillna(medians)
        )

    df[imputation_artifacts.continuous_cols] = df[imputation_artifacts.continuous_cols].fillna(
        imputation_artifacts.global_medians
    )
    df = df.drop(columns=["_group_key"], errors="ignore")

    X = df[feature_cols].values.astype(float)

    if fit_scaler is None:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        scaler = fit_scaler
        X = scaler.transform(X)

    return X, scaler


# modelling 

def fit_league_model(train_league, val_league, feature_cols, imputation_artifacts, random_seed, n_trials):
    """
    Select hyperparameters using Optuna validated on the held-out val season,
    then refit on training data with the best parameters.
    """
    y_train_raw = train_league["Points"].values.astype(float)
    y_mean = float(y_train_raw.mean())
    y_std = float(y_train_raw.std())
    if y_std == 0:
        raise ValueError("Training target has zero variance; cannot standardise.")
    y_train = (y_train_raw - y_mean) / y_std

    X_train, scaler = apply_transforms(train_league, feature_cols, imputation_artifacts)
    X_val, _ = apply_transforms(val_league, feature_cols, imputation_artifacts, fit_scaler=scaler)
    y_val = (val_league["Points"].values.astype(float) - y_mean) / y_std

    def objective(trial):
        alpha = trial.suggest_float("alpha", ALPHA_LOW, ALPHA_HIGH, log=True)
        l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10_000)
        model.fit(X_train, y_train)
        return float(np.sqrt(mean_squared_error(y_val, model.predict(X_val))))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=random_seed),
    )
    study.optimize(objective, n_trials=n_trials)

    best = study.best_params
    val_rmse = float(study.best_value)
    en = ElasticNet(alpha=best["alpha"], l1_ratio=best["l1_ratio"], max_iter=10_000)
    en.fit(X_train, y_train)

    return en, scaler, y_mean, y_std, val_rmse, X_train


def _safe_r2(y_true, y_pred):
    if len(y_true) < 2 or np.isclose(np.var(y_true), 0.0):
        return None
    return float(r2_score(y_true, y_pred))


# outputs

def save_shap_outputs(en, X_train, X_holdout, feature_cols, league_dir):
    explainer = shap.LinearExplainer(en, X_train)
    shap_values = explainer.shap_values(X_holdout)

    plt.figure()
    shap.summary_plot(shap_values, X_holdout, feature_names=feature_cols, show=False)
    plt.tight_layout()
    plt.savefig(league_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, X_holdout, feature_names=feature_cols,
                      plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(league_dir / "shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).to_csv(
        league_dir / "shap_mean_abs.csv", index=False
    )


def save_coefficients_plot(coef_df, title, output_path):
    fig, ax = plt.subplots(figsize=(7, max(3, len(coef_df) * 0.45)))
    colors = ["steelblue" if c >= 0 else "salmon" for c in coef_df["Coefficient"]]
    ax.barh(coef_df["Feature"], coef_df["Coefficient"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Coefficient")
    plt.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def save_actual_vs_predicted_plot(y_true, y_pred, title, output_path):
    mn = min(y_true.min(), y_pred.min())
    mx = max(y_true.max(), y_pred.max())
    pad = 1.0 if np.isclose(mx, mn) else 0.05 * (mx - mn)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred)
    plt.plot([mn - pad, mx + pad], [mn - pad, mx + pad], linestyle="--")
    plt.xlabel("Actual Points")
    plt.ylabel("Predicted Points")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_league_outputs(
    league,
    en,
    feature_cols,
    val_rmse,
    y_train,
    y_train_pred,
    y_holdout,
    y_holdout_pred,
    league_dir,
    X_train=None,
    X_holdout=None,
):
    league_dir.mkdir(parents=True, exist_ok=True)

    coef_df = pd.DataFrame(
        {
            "Feature": feature_cols,
            "Coefficient": en.coef_,
            "AbsCoefficient": np.abs(en.coef_),
        }
    ).sort_values("AbsCoefficient", ascending=False)
    coef_df.to_csv(league_dir / "coefficients.csv", index=False)

    save_coefficients_plot(
        coef_df=coef_df.sort_values("Coefficient"),
        title=f"{league}: elastic net coefficients",
        output_path=league_dir / "coefficients.png",
    )

    has_holdout = y_holdout is not None and len(y_holdout) >= 2
    metrics = {
        "league": league,
        "n_train": len(y_train),
        "n_holdout": len(y_holdout) if y_holdout is not None else 0,
        "alpha": float(en.alpha),
        "l1_ratio": float(en.l1_ratio),
        "n_active_features": int((np.abs(en.coef_) > 0).sum()),
        "train_r2": float(r2_score(y_train, y_train_pred)),
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, y_train_pred))),
        "val_rmse": val_rmse,
        "holdout_rmse": float(np.sqrt(mean_squared_error(y_holdout, y_holdout_pred))) if has_holdout else None,
        "holdout_mae": float(mean_absolute_error(y_holdout, y_holdout_pred)) if has_holdout else None,
        "holdout_r2": _safe_r2(y_holdout, y_holdout_pred) if has_holdout else None,
    }

    with open(league_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if y_holdout is not None and len(y_holdout) > 0:
        pred_df = pd.DataFrame(
            {
                "Actual": y_holdout,
                "Predicted": y_holdout_pred,
                "Residual": y_holdout - y_holdout_pred,
            }
        )
        pred_df.to_csv(league_dir / "holdout_predictions.csv", index=False)

        if has_holdout:
            save_actual_vs_predicted_plot(
                y_true=y_holdout,
                y_pred=y_holdout_pred,
                title=f"{league}: actual vs predicted (holdout)",
                output_path=league_dir / "actual_vs_predicted.png",
            )

    if X_train is not None and X_holdout is not None and len(X_holdout) > 0:
        save_shap_outputs(en, X_train, X_holdout, feature_cols, league_dir)

    return metrics


# main 

def main():
    args = parse_mode_args("Elastic net regression with lagged predictors per league.")
    run_config = build_run_config(args.mode)
    save_run_config(run_config, run_config.output_dir)

    print("Loading and cleaning data...")
    df = load_and_clean_dataset(INPUT_PATH)

    print("Building lagged dataset (previous-season predictors)...")
    lagged_df = build_lagged_dataset(df)
    print(f"Lagged rows available: {len(lagged_df)}")

    print(f"Splitting train / val ({run_config.val_season}) / holdout ({run_config.holdout_season})...")
    train_raw, val_raw, holdout_raw = split_train_val_holdout(
        lagged_df, run_config.val_season, run_config.holdout_season
    )
    print(f"Train rows: {len(train_raw)} | Val rows: {len(val_raw)} | Holdout rows: {len(holdout_raw)}")
    print(f"Mode: {run_config.mode}\n")

    feature_cols = build_feature_cols()
    leagues = sorted(lagged_df["League"].unique())

    print("Fitting imputation artifacts on training data...")
    imputation_artifacts = fit_imputation_artifacts(train_raw)

    all_metrics = []
    all_coefs = []

    for league in leagues:
        print(f"=== {league} ===")

        train_league = train_raw[train_raw["League"] == league].copy()
        val_league = val_raw[val_raw["League"] == league].copy()
        holdout_league = holdout_raw[holdout_raw["League"] == league].copy()

        if len(train_league) < 10:
            print(f"  Skipping: only {len(train_league)} training rows (need >= 10).\n")
            continue
        if val_league.empty:
            print(f"  Skipping: no val rows for {league}.\n")
            continue

        en, scaler, y_mean, y_std, val_rmse, X_train = fit_league_model(
            train_league=train_league,
            val_league=val_league,
            feature_cols=feature_cols,
            imputation_artifacts=imputation_artifacts,
            random_seed=run_config.random_seed,
            n_trials=run_config.n_trials,
        )

        y_train = train_league["Points"].values.astype(float)
        y_train_pred = en.predict(X_train) * y_std + y_mean

        if len(holdout_league) >= 2:
            X_holdout, _ = apply_transforms(holdout_league, feature_cols, imputation_artifacts, fit_scaler=scaler)
            y_holdout = holdout_league["Points"].values.astype(float)
            y_holdout_pred = en.predict(X_holdout) * y_std + y_mean
        else:
            X_holdout = None
            y_holdout, y_holdout_pred = None, None
            print(f"  Warning: only {len(holdout_league)} holdout rows — skipping holdout evaluation.")

        league_dir = run_config.output_dir / slugify_filename(league)
        metrics = save_league_outputs(
            league=league,
            en=en,
            feature_cols=feature_cols,
            val_rmse=val_rmse,
            y_train=y_train,
            y_train_pred=y_train_pred,
            y_holdout=y_holdout,
            y_holdout_pred=y_holdout_pred,
            league_dir=league_dir,
            X_train=X_train,
            X_holdout=X_holdout,
        )

        all_metrics.append(metrics)
        all_coefs.append(
            pd.DataFrame(
                {
                    "League": league,
                    "Feature": feature_cols,
                    "Coefficient": en.coef_,
                }
            )
        )

        print(f"  Train R²:  {metrics['train_r2']:.3f}")
        print(f"  Val RMSE:  {metrics['val_rmse']:.4f} (standardised)")
        if metrics["holdout_r2"] is not None:
            print(f"  Holdout R²: {metrics['holdout_r2']:.3f} | RMSE: {metrics['holdout_rmse']:.4f} | MAE: {metrics['holdout_mae']:.4f}")
        print(f"  alpha={metrics['alpha']:.5f} | l1_ratio={metrics['l1_ratio']:.2f} | active features: {metrics['n_active_features']}/{len(feature_cols)}")
        print()

    if all_metrics:
        pd.DataFrame(all_metrics).to_csv(run_config.output_dir / "overall_metrics.csv", index=False)

    if all_coefs:
        pd.concat(all_coefs, ignore_index=True).to_csv(
            run_config.output_dir / "all_coefficients.csv", index=False
        )

    print(f"Outputs saved to: {run_config.output_dir.resolve()}")


if __name__ == "__main__":
    main()
