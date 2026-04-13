"""
Bayesian football model focused only on the main research question.

This script builds a lagged club-season panel so that each row predicts the
current season result using the same club's previous-season features.

Outputs are saved only at the league level, because the goal is to answer:
which previous-season factors matter most for predicting next-season
performance in each league?
"""

import json
import re
import warnings
from dataclasses import dataclass

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from pipeline_config import (
    HOLDOUT_SEASON,
    INPUT_PATH,
    build_run_config,
    parse_mode_args,
    save_run_config,
)

warnings.filterwarnings("ignore")


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
RAW_NUMERIC_COLUMNS = [
    "Avg. Squad Age",
    "Points",
    "Goals For",
    "Goals Against",
]
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
CONTINUOUS_COLUMNS = PREVIOUS_FEATURE_COLUMNS
PREPROCESSING_GROUP_COLUMN = "Prev League"
DEFAULT_HDI_PROB = 0.94

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

ENGLISH_LEAGUES = {
    "Premier League",
    "Championship",
    "League One",
    "League Two",
}


@dataclass
class PreprocessingArtifacts:
    feature_cols: list
    continuous_cols: list
    x_means: pd.Series
    x_stds: pd.Series
    y_mean: float
    y_std: float
    group_medians: pd.DataFrame
    global_medians: pd.Series
    imputation_group_col: str


def parse_money(value):
    """
    Convert strings like 'EUR35.46m', '1.16bn', '450k', '-', '' to float.
    """
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

    missing_required = [column for column in RAW_REQUIRED_COLUMNS if column not in frame.columns]
    if missing_required:
        raise ValueError(
            f"Sheet '{sheet_name}' is missing required columns after normalization: "
            f"{missing_required}"
        )

    expected_columns = RAW_REQUIRED_COLUMNS + RAW_OPTIONAL_COLUMNS
    for column in expected_columns:
        if column not in frame.columns:
            frame[column] = np.nan

    return frame


def load_and_clean_dataset(path):
    """
    Read all league sheets, normalize minor header inconsistencies, and coerce types.
    """
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


def maybe_filter_development_leagues(df, run_config):
    """
    In development mode, restrict the dataset to English leagues only so
    quick test runs stay focused.
    """
    if run_config.mode != "development":
        return df

    filtered = df[df["League"].isin(ENGLISH_LEAGUES)].copy()

    if filtered.empty:
        raise ValueError(
            "Development-mode English league filter removed all rows. "
            f"Available leagues were: {sorted(df['League'].dropna().unique())}"
        )

    print(
        "Development mode: restricting data to English leagues only: "
        f"{sorted(ENGLISH_LEAGUES)}"
    )
    print(f"Rows before filter: {len(df)} | after filter: {len(filtered)}")

    return filtered


def build_lagged_dataset(df):
    """
    Build one row per club-season where predictors come from the immediately
    previous season and the target is the current season result.
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


def split_train_holdout(df, holdout_season):
    train = df[df["Season"] != holdout_season].copy()
    holdout = df[df["Season"] == holdout_season].copy()

    if train.empty:
        raise ValueError("Training split is empty after applying the holdout season filter.")
    if holdout.empty:
        raise ValueError("Holdout split is empty for the requested holdout season.")

    return train, holdout


def fill_continuous_by_group_medians(transformed, artifacts):
    group_labels = transformed[artifacts.imputation_group_col]
    filled = transformed[artifacts.continuous_cols].copy()

    unique_groups = [group for group in pd.Series(group_labels).dropna().unique()]
    for group in unique_groups:
        idx = group_labels == group
        medians = (
            artifacts.group_medians.loc[group]
            if group in artifacts.group_medians.index
            else artifacts.global_medians
        )
        filled.loc[idx, artifacts.continuous_cols] = filled.loc[idx, artifacts.continuous_cols].fillna(
            medians
        )

    filled = filled.fillna(artifacts.global_medians)
    transformed[artifacts.continuous_cols] = filled
    return transformed



def fit_preprocessor(train_df):
    """
    Learn preprocessing state from training data only.
    Imputation is grouped by previous-season league so promoted/relegated clubs
    are matched to the division their lagged predictors came from.
    """
    train = train_df.copy()

    missing_flag_cols = []
    for col in PREVIOUS_OPTIONAL_COLUMNS:
        flag = f"{col}_missing"
        train[flag] = train[col].isna().astype(int)
        missing_flag_cols.append(flag)

    medians_frame = train.copy()
    for col in PREVIOUS_FINANCIAL_COLUMNS:
        medians_frame[col] = np.log1p(medians_frame[col])

    grouping_series = medians_frame[PREPROCESSING_GROUP_COLUMN].fillna(medians_frame["League"])
    medians_frame["_group_key"] = grouping_series

    group_medians = medians_frame.groupby("_group_key")[CONTINUOUS_COLUMNS].median()
    global_medians = medians_frame[CONTINUOUS_COLUMNS].median()

    train = apply_preprocessor(
        train,
        PreprocessingArtifacts(
            feature_cols=CONTINUOUS_COLUMNS + missing_flag_cols,
            continuous_cols=CONTINUOUS_COLUMNS,
            x_means=pd.Series(dtype=float),
            x_stds=pd.Series(dtype=float),
            y_mean=np.nan,
            y_std=np.nan,
            group_medians=group_medians,
            global_medians=global_medians,
            imputation_group_col=PREPROCESSING_GROUP_COLUMN,
        ),
        include_target=False,
    )

    x_means = train[CONTINUOUS_COLUMNS].mean()
    x_stds = train[CONTINUOUS_COLUMNS].std().replace(0, 1)
    train[CONTINUOUS_COLUMNS] = (train[CONTINUOUS_COLUMNS] - x_means) / x_stds

    y_mean = train["Points"].mean()
    y_std = train["Points"].std()
    if y_std == 0:
        raise ValueError("Training target has zero variance; cannot standardise.")

    train["y_std"] = (train["Points"] - y_mean) / y_std

    artifacts = PreprocessingArtifacts(
        feature_cols=CONTINUOUS_COLUMNS + missing_flag_cols,
        continuous_cols=CONTINUOUS_COLUMNS,
        x_means=x_means,
        x_stds=x_stds,
        y_mean=float(y_mean),
        y_std=float(y_std),
        group_medians=group_medians,
        global_medians=global_medians,
        imputation_group_col=PREPROCESSING_GROUP_COLUMN,
    )

    return train, artifacts


def apply_preprocessor(frame, artifacts, include_target=True):
    transformed = frame.copy()

    for col in PREVIOUS_OPTIONAL_COLUMNS:
        if col not in transformed.columns:
            transformed[col] = np.nan
        flag = f"{col}_missing"
        transformed[flag] = transformed[col].isna().astype(int)

    for col in PREVIOUS_FINANCIAL_COLUMNS:
        transformed[col] = np.log1p(transformed[col])

    group_keys = transformed[artifacts.imputation_group_col].fillna(transformed["League"])
    transformed["_group_key"] = group_keys

    for group_key in transformed["_group_key"].dropna().unique():
        idx = transformed["_group_key"] == group_key
        medians = (
            artifacts.group_medians.loc[group_key]
            if group_key in artifacts.group_medians.index
            else artifacts.global_medians
        )
        transformed.loc[idx, artifacts.continuous_cols] = transformed.loc[
            idx, artifacts.continuous_cols
        ].fillna(medians)

    transformed[artifacts.continuous_cols] = transformed[artifacts.continuous_cols].fillna(
        artifacts.global_medians
    )

    transformed = transformed.drop(columns=["_group_key"], errors="ignore")

    if not artifacts.x_means.empty:
        transformed[artifacts.continuous_cols] = (
            transformed[artifacts.continuous_cols] - artifacts.x_means
        ) / artifacts.x_stds

    if include_target:
        transformed["y_std"] = (
            transformed["Points"] - artifacts.y_mean
        ) / artifacts.y_std

    return transformed

def build_and_fit_model(train_df, feature_cols, run_config):
    """
    Hierarchical model with league-specific intercepts and slopes.
    """
    leagues = sorted(train_df["League"].unique())
    league_to_idx = {league: i for i, league in enumerate(leagues)}

    group_idx = train_df["League"].map(league_to_idx).values
    X = train_df[feature_cols].values.astype(float)
    y = train_df["y_std"].values.astype(float)

    coords = {"group": leagues, "predictor": feature_cols}

    with pm.Model(coords=coords) as model:
        mu_alpha = pm.Normal("mu_alpha", mu=0.0, sigma=1.0)
        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=1.0)

        mu_beta = pm.Normal("mu_beta", mu=0.0, sigma=0.7, dims="predictor")
        sigma_beta = pm.HalfNormal("sigma_beta", sigma=0.5, dims="predictor")

        z_alpha = pm.Normal("z_alpha", mu=0.0, sigma=1.0, dims="group")
        alpha = pm.Deterministic("alpha", mu_alpha + sigma_alpha * z_alpha, dims="group")

        z_beta = pm.Normal("z_beta", mu=0.0, sigma=1.0, dims=("group", "predictor"))
        beta = pm.Deterministic(
            "beta",
            mu_beta + sigma_beta * z_beta,
            dims=("group", "predictor"),
        )

        sigma = pm.HalfNormal("sigma", sigma=1.0)
        mu = alpha[group_idx] + (beta[group_idx] * X).sum(axis=1)

        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

        idata = pm.sample(
            draws=run_config.draws,
            tune=run_config.tune,
            chains=run_config.chains,
            cores=1,
            target_accept=run_config.target_accept,
            random_seed=run_config.random_seed,
            progressbar=run_config.progressbar,
            return_inferencedata=True,
        )

    return model, idata, league_to_idx


def predict_holdout(
    idata,
    holdout_df,
    feature_cols,
    league_to_idx,
    y_mean,
    y_std,
    interval_prob=0.94,
    return_samples=False,
):
    X_holdout = holdout_df[feature_cols].values.astype(float)
    group_idx_holdout = holdout_df["League"].map(league_to_idx).values

    if np.isnan(group_idx_holdout).any():
        missing_leagues = sorted(holdout_df.loc[pd.isna(group_idx_holdout), "League"].unique())
        raise ValueError(
            f"Holdout contains unseen leagues with no trained group effect: {missing_leagues}"
        )
    group_idx_holdout = group_idx_holdout.astype(int)

    alpha_samples = (
        idata.posterior["alpha"]
        .stack(sample=("chain", "draw"))
        .transpose("sample", "group")
        .values
    )
    beta_samples = (
        idata.posterior["beta"]
        .stack(sample=("chain", "draw"))
        .transpose("sample", "group", "predictor")
        .values
    )
    sigma_samples = (
        idata.posterior["sigma"]
        .stack(sample=("chain", "draw"))
        .values
    )  # (n_samples,)

    selected_alpha = alpha_samples[:, group_idx_holdout]
    selected_beta = beta_samples[:, group_idx_holdout, :]

    mu_std_samples = selected_alpha + np.einsum("sop,op->so", selected_beta, X_holdout)

    # posterior predictive samples for observed outcomes
    y_std_samples = np.random.normal(
        loc=mu_std_samples,
        scale=sigma_samples[:, None],
    )

    pred_samples = y_std_samples * y_std + y_mean

    alpha_tail = (1.0 - interval_prob) / 2.0

    pred_mean = pred_samples.mean(axis=0)
    pred_sd = pred_samples.std(axis=0, ddof=1)
    pred_hdi_lower = np.quantile(pred_samples, alpha_tail, axis=0)
    pred_hdi_upper = np.quantile(pred_samples, 1.0 - alpha_tail, axis=0)

    summary = {
        "mean": pred_mean,
        "sd": pred_sd,
        "hdi_lower": pred_hdi_lower,
        "hdi_upper": pred_hdi_upper,
        "interval_prob": interval_prob,
    }

    if return_samples:
        summary["samples"] = pred_samples

    return summary


def safe_r2_value(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) < 2:
        return None
    if np.isclose(np.var(y_true), 0.0):
        return None

    return float(r2_score(y_true, y_pred))


def extract_league_coefficient_summary(idata, hdi_prob=0.94):
    beta = idata.posterior["beta"]
    stacked = beta.stack(sample=("chain", "draw")).transpose("group", "predictor", "sample")
    sample_values = np.asarray(stacked.values, dtype=float)

    means = sample_values.mean(axis=-1)
    sds = sample_values.std(axis=-1, ddof=1)

    # ArviZ can collapse dimensions unexpectedly when given a raw NumPy array
    # with more than one non-sample axis. Compute the HDI one coefficient at a time
    # so the output shape is always (n_groups, n_predictors, 2).
    hdi = np.empty(sample_values.shape[:-1] + (2,), dtype=float)
    for idx in np.ndindex(sample_values.shape[:-1]):
        hdi[idx] = az.hdi(sample_values[idx], hdi_prob=hdi_prob)

    prob_positive = (sample_values > 0).mean(axis=-1)
    prob_negative = (sample_values < 0).mean(axis=-1)

    rows = []
    for group_idx, league in enumerate(stacked.coords["group"].values.tolist()):
        for predictor_idx, feature in enumerate(stacked.coords["predictor"].values.tolist()):
            mean_value = float(means[group_idx, predictor_idx])
            hdi_lower = float(hdi[group_idx, predictor_idx, 0])
            hdi_upper = float(hdi[group_idx, predictor_idx, 1])
            robust_nonzero = bool((hdi_lower > 0) or (hdi_upper < 0))

            if hdi_lower > 0:
                direction = "positive"
            elif hdi_upper < 0:
                direction = "negative"
            else:
                direction = "mixed"

            rows.append(
                {
                    "League": str(league),
                    "Feature": str(feature),
                    "MeanCoefficient": mean_value,
                    "AbsMeanCoefficient": abs(mean_value),
                    "SD": float(sds[group_idx, predictor_idx]),
                    "HDILower": hdi_lower,
                    "HDIUpper": hdi_upper,
                    "ProbabilityPositive": float(prob_positive[group_idx, predictor_idx]),
                    "ProbabilityNegative": float(prob_negative[group_idx, predictor_idx]),
                    "RobustNonZero": robust_nonzero,
                    "Direction": direction,
                    "IntervalProbability": hdi_prob,
                }
            )

    summary_df = pd.DataFrame(rows)
    summary_df["AbsMeanRank"] = summary_df.groupby("League")["AbsMeanCoefficient"].rank(
        method="dense", ascending=False
    )

    robust_rank_df = summary_df.sort_values(
        ["League", "RobustNonZero", "AbsMeanCoefficient"],
        ascending=[True, False, False],
    ).copy()
    robust_rank_df["RobustRank"] = robust_rank_df.groupby("League").cumcount() + 1
    summary_df = summary_df.merge(
        robust_rank_df[["League", "Feature", "RobustRank"]],
        on=["League", "Feature"],
        how="left",
    )

    return summary_df.sort_values(
        ["League", "RobustNonZero", "AbsMeanCoefficient"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def extract_league_coefficient_means(idata):
    return extract_league_coefficient_summary(idata)[["League", "Feature", "MeanCoefficient"]].copy()


def summarise_feature_stability(coefficient_summary):
    if coefficient_summary.empty:
        return pd.DataFrame(
            columns=[
                "Feature",
                "NumLeagues",
                "NumRobustNonZero",
                "NumRobustPositive",
                "NumRobustNegative",
                "MeanAbsCoefficient",
                "MeanCoefficient",
                "MeanProbabilityPositive",
            ]
        )

    grouped = coefficient_summary.groupby("Feature")
    summary_df = pd.DataFrame(
        {
            "Feature": grouped.size().index,
            "NumLeagues": grouped["League"].nunique().values,
            "NumRobustNonZero": grouped["RobustNonZero"].sum().astype(int).values,
            "NumRobustPositive": grouped.apply(
                lambda g: int(((g["Direction"] == "positive") & g["RobustNonZero"]).sum())
            ).values,
            "NumRobustNegative": grouped.apply(
                lambda g: int(((g["Direction"] == "negative") & g["RobustNonZero"]).sum())
            ).values,
            "MeanAbsCoefficient": grouped["AbsMeanCoefficient"].mean().values,
            "MeanCoefficient": grouped["MeanCoefficient"].mean().values,
            "MeanProbabilityPositive": grouped["ProbabilityPositive"].mean().values,
        }
    )

    return summary_df.sort_values(
        ["NumRobustNonZero", "MeanAbsCoefficient"],
        ascending=[False, False],
    ).reset_index(drop=True)


def build_holdout_prediction_df(holdout_df, prediction_summary):
    pred_df = holdout_df[
        ["Season", "Prev Season", "League", "Prev League", "Club", "Points"]
    ].copy()
    pred_df["Predicted Points"] = prediction_summary["mean"]
    pred_df["Prediction SD"] = prediction_summary["sd"]
    pred_df["Prediction HDI Lower"] = prediction_summary["hdi_lower"]
    pred_df["Prediction HDI Upper"] = prediction_summary["hdi_upper"]
    pred_df["Residual"] = pred_df["Points"] - pred_df["Predicted Points"]
    pred_df["Within Prediction Interval"] = (
        (pred_df["Points"] >= pred_df["Prediction HDI Lower"])
        & (pred_df["Points"] <= pred_df["Prediction HDI Upper"])
    )
    return pred_df


def save_barh_plot(plot_df, y_col, x_col, title, output_path, figsize=(9, 6)):
    if plot_df.empty:
        return

    plt.figure(figsize=figsize)
    plt.barh(plot_df[y_col], plot_df[x_col])
    plt.axvline(0, linestyle="--")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_actual_vs_predicted_plot(pred_df, title, output_path):
    if pred_df.empty:
        return

    plt.figure(figsize=(6, 6))
    plt.scatter(pred_df["Points"], pred_df["Predicted Points"])

    mn = min(pred_df["Points"].min(), pred_df["Predicted Points"].min())
    mx = max(pred_df["Points"].max(), pred_df["Predicted Points"].max())
    pad = 1.0 if np.isclose(mx, mn) else 0.05 * (mx - mn)

    plt.plot([mn - pad, mx + pad], [mn - pad, mx + pad], linestyle="--")
    plt.xlabel("Actual Points")
    plt.ylabel("Predicted Points")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_per_league_outputs(idata, train_df, holdout_df, prediction_summary, base_output_dir, coefficient_hdi_prob=0.94):
    per_league_dir = base_output_dir / "per_league"
    per_league_dir.mkdir(parents=True, exist_ok=True)

    coefficient_summary = extract_league_coefficient_summary(idata, hdi_prob=coefficient_hdi_prob)
    coefficient_summary.to_csv(per_league_dir / "all_league_coefficients.csv", index=False)
    summarise_feature_stability(coefficient_summary).to_csv(
        per_league_dir / "cross_league_feature_summary.csv",
        index=False,
    )

    pred_df = build_holdout_prediction_df(holdout_df, prediction_summary)
    pred_df.to_csv(per_league_dir / "all_holdout_predictions.csv", index=False)

    metrics_rows = []

    for league in sorted(train_df["League"].unique()):
        league_dir = per_league_dir / slugify_filename(league)
        league_dir.mkdir(parents=True, exist_ok=True)

        league_coef_df = coefficient_summary[coefficient_summary["League"] == league].copy()
        league_coef_df.to_csv(league_dir / "coefficients.csv", index=False)
        league_coef_df[league_coef_df["RobustNonZero"]].to_csv(
            league_dir / "robust_predictors.csv",
            index=False,
        )
        save_barh_plot(
            plot_df=league_coef_df.sort_values("MeanCoefficient"),
            y_col="Feature",
            x_col="MeanCoefficient",
            title=f"{league}: posterior mean coefficients",
            output_path=league_dir / "coefficients.png",
        )

        league_holdout_df = pred_df[pred_df["League"] == league].copy()
        league_holdout_df.to_csv(league_dir / "holdout_predictions.csv", index=False)

        if league_holdout_df.empty:
            league_metrics = {
                "League": league,
                "train_rows": int((train_df["League"] == league).sum()),
                "holdout_rows": 0,
                "rmse": None,
                "mae": None,
                "r2": None,
                "prediction_interval_coverage": None,
            }
        else:
            y_true = league_holdout_df["Points"].values
            y_pred = league_holdout_df["Predicted Points"].values

            league_metrics = {
                "League": league,
                "train_rows": int((train_df["League"] == league).sum()),
                "holdout_rows": int(len(league_holdout_df)),
                "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "r2": safe_r2_value(y_true, y_pred),
                "prediction_interval_coverage": float(
                    league_holdout_df["Within Prediction Interval"].mean()
                ),
            }

            save_actual_vs_predicted_plot(
                pred_df=league_holdout_df,
                title=f"{league}: actual vs predicted",
                output_path=league_dir / "actual_vs_predicted.png",
            )

        with open(league_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(league_metrics, f, indent=2)

        metrics_rows.append(league_metrics)

    pd.DataFrame(metrics_rows).to_csv(per_league_dir / "per_league_metrics.csv", index=False)


def save_outputs(idata, train_df, holdout_df, prediction_summary, preprocessing_artifacts, run_config):
    output_dir = run_config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    save_run_config(run_config, output_dir)

    y_true = holdout_df["Points"].values
    predictions = prediction_summary["mean"]
    rmse = float(np.sqrt(mean_squared_error(y_true, predictions)))
    mae = float(mean_absolute_error(y_true, predictions))
    r2 = float(r2_score(y_true, predictions))
    divergences = int(idata.sample_stats["diverging"].sum().values)
    max_rhat = float(az.rhat(idata).to_array().max().values)

    pred_df = build_holdout_prediction_df(holdout_df, prediction_summary)
    prediction_interval_coverage = float(pred_df["Within Prediction Interval"].mean())

    diagnostics = {
        "mode": run_config.mode,
        "train_rows": int(len(train_df)),
        "holdout_rows": int(len(holdout_df)),
        "holdout_season": run_config.holdout_season,
        "predictor_timing": "previous_season",
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "prediction_interval_coverage": prediction_interval_coverage,
        "prediction_interval_probability": float(prediction_summary["interval_prob"]),
        "divergences": divergences,
        "max_rhat": max_rhat,
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    preprocessing_summary = {
        "mode": run_config.mode,
        "holdout_season": run_config.holdout_season,
        "predictor_timing": "previous_season",
        "feature_cols": preprocessing_artifacts.feature_cols,
        "continuous_cols": preprocessing_artifacts.continuous_cols,
        "imputation_group_col": preprocessing_artifacts.imputation_group_col,
        "x_means": {k: float(v) for k, v in preprocessing_artifacts.x_means.items()},
        "x_stds": {k: float(v) for k, v in preprocessing_artifacts.x_stds.items()},
        "y_mean": float(preprocessing_artifacts.y_mean),
        "y_std": float(preprocessing_artifacts.y_std),
    }
    with open(output_dir / "preprocessing_summary.json", "w", encoding="utf-8") as f:
        json.dump(preprocessing_summary, f, indent=2)

    league_summary = az.summary(idata, var_names=["alpha", "beta"], round_to=4)
    league_summary.to_csv(output_dir / "league_parameter_summary.csv")

    save_per_league_outputs(
        idata=idata,
        train_df=train_df,
        holdout_df=holdout_df,
        prediction_summary=prediction_summary,
        base_output_dir=output_dir,
        coefficient_hdi_prob=run_config.coefficient_hdi_prob,
    )

    report_lines = [
        "Bayesian hierarchical football model report",
        "Predictor timing: previous season",
        f"Mode: {run_config.mode}",
        f"Holdout season: {run_config.holdout_season}",
        f"Train rows: {len(train_df)}",
        f"Holdout rows: {len(holdout_df)}",
        "",
        "Overall holdout metrics",
        f"RMSE: {rmse:.6f}",
        f"MAE: {mae:.6f}",
        f"R2: {r2:.6f}",
        f"Prediction interval coverage ({prediction_summary['interval_prob']:.0%} interval): {prediction_interval_coverage:.6f}",
        "",
        "Sampler diagnostics",
        f"Divergences: {divergences}",
        f"Max R-hat: {max_rhat:.6f}",
        "",
        "Feature-uncertainty outputs saved:",
        str(output_dir / "per_league" / "all_league_coefficients.csv"),
        str(output_dir / "per_league" / "cross_league_feature_summary.csv"),
        "",
        "League-level outputs saved under:",
        str(output_dir / "per_league"),
    ]
    with open(output_dir / "report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    return diagnostics


def main():
    args = parse_mode_args("Fit the hierarchical Bayesian football model.")
    run_config = build_run_config(args.mode)
    np.random.seed(run_config.random_seed)

    print("Loading and cleaning data...")
    df = load_and_clean_dataset(INPUT_PATH)
    df = maybe_filter_development_leagues(df, run_config)

    print("Building club-season dataset with previous-season predictors...")
    lagged_df = build_lagged_dataset(df)
    print(f"Lagged rows available: {len(lagged_df)}")

    print(f"Creating train/holdout split with holdout season {HOLDOUT_SEASON}...")
    train_raw, holdout_raw = split_train_holdout(lagged_df, run_config.holdout_season)

    print("Fitting preprocessing on training data only...")
    train_df, preprocessing_artifacts = fit_preprocessor(train_raw)
    holdout_df = apply_preprocessor(holdout_raw, preprocessing_artifacts)

    print(f"Train rows: {len(train_df)}")
    print(f"Holdout rows ({run_config.holdout_season}): {len(holdout_df)}")
    print(f"Running mode: {run_config.mode}")

    print("Fitting Bayesian hierarchical model...")
    _, idata, league_to_idx = build_and_fit_model(
        train_df=train_df,
        feature_cols=preprocessing_artifacts.feature_cols,
        run_config=run_config,
    )

    print("Predicting holdout season...")
    prediction_summary = predict_holdout(
        idata=idata,
        holdout_df=holdout_df,
        feature_cols=preprocessing_artifacts.feature_cols,
        league_to_idx=league_to_idx,
        y_mean=preprocessing_artifacts.y_mean,
        y_std=preprocessing_artifacts.y_std,
        interval_prob=run_config.prediction_hdi_prob,
    )

    print("Saving outputs...")
    diagnostics = save_outputs(
        idata=idata,
        train_df=train_df,
        holdout_df=holdout_df,
        prediction_summary=prediction_summary,
        preprocessing_artifacts=preprocessing_artifacts,
        run_config=run_config,
    )

    print("\n=== HOLDOUT METRICS ===")
    for k, v in diagnostics.items():
        print(f"{k}: {v}")

    print(f"\nSaved outputs to: {run_config.output_dir.resolve()}")


if __name__ == "__main__":
    main()
