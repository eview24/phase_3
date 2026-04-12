"""
Debugging version of Bayesian formulation comparison.

This version only fits the normal likelihood with league-specific varying
slopes so the debug run is faster and focused on the current issue.
"""

import json

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from bayes_model import (
    apply_preprocessor,
    build_holdout_prediction_df,
    build_lagged_dataset,
    extract_league_coefficient_summary,
    fit_preprocessor,
    load_and_clean_dataset,
    maybe_filter_development_leagues,
    predict_holdout,
    safe_r2_value,
    save_actual_vs_predicted_plot,
    save_barh_plot,
    slugify_filename,
    split_train_holdout,
    summarise_feature_stability,
)
from pipeline_config import (
    HOLDOUT_SEASON,
    INPUT_PATH,
    build_run_config,
    parse_mode_args,
    save_run_config,
)


MODEL_SPECS = [
    {
        "model_name": "normal_varying_slopes",
        "likelihood": "normal",
    }
]


def build_and_fit_specification(train_df, feature_cols, likelihood, run_config):
    leagues = sorted(train_df["League"].unique())
    league_to_idx = {league: i for i, league in enumerate(leagues)}

    group_idx = train_df["League"].map(league_to_idx).values
    X = train_df[feature_cols].values.astype(float)
    y = train_df["y_std"].values.astype(float)

    coords = {"group": leagues, "predictor": feature_cols}

    with pm.Model(coords=coords):
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

    return idata, league_to_idx


def compute_per_league_metrics(train_df, pred_df):
    metrics_rows = []

    for league in sorted(train_df["League"].unique()):
        league_pred_df = pred_df[pred_df["League"] == league].copy()

        if league_pred_df.empty:
            row = {
                "League": league,
                "train_rows": int((train_df["League"] == league).sum()),
                "holdout_rows": 0,
                "rmse": None,
                "mae": None,
                "r2": None,
                "prediction_interval_coverage": None,
            }
        else:
            y_true = league_pred_df["Season Result"].values
            y_pred = league_pred_df["Predicted Season Result"].values

            row = {
                "League": league,
                "train_rows": int((train_df["League"] == league).sum()),
                "holdout_rows": int(len(league_pred_df)),
                "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "r2": safe_r2_value(y_true, y_pred),
                "prediction_interval_coverage": float(
                    league_pred_df["Within Prediction Interval"].mean()
                ),
            }

        metrics_rows.append(row)

    return pd.DataFrame(metrics_rows)


def save_per_league_outputs(idata, train_df, pred_df, model_output_dir, coefficient_hdi_prob=0.94):
    per_league_dir = model_output_dir / "per_league"
    per_league_dir.mkdir(parents=True, exist_ok=True)

    coefficient_summary = extract_league_coefficient_summary(idata, hdi_prob=coefficient_hdi_prob)
    coefficient_summary.to_csv(per_league_dir / "all_league_coefficients.csv", index=False)
    summarise_feature_stability(coefficient_summary).to_csv(
        per_league_dir / "cross_league_feature_summary.csv",
        index=False,
    )

    pred_df.to_csv(per_league_dir / "all_holdout_predictions.csv", index=False)

    per_league_metrics = compute_per_league_metrics(train_df, pred_df)
    per_league_metrics.to_csv(per_league_dir / "per_league_metrics.csv", index=False)

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

        league_pred_df = pred_df[pred_df["League"] == league].copy()
        league_pred_df.to_csv(league_dir / "holdout_predictions.csv", index=False)

        metrics_row = per_league_metrics[per_league_metrics["League"] == league].iloc[0].to_dict()
        with open(league_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics_row, f, indent=2)

        if not league_pred_df.empty:
            save_actual_vs_predicted_plot(
                pred_df=league_pred_df,
                title=f"{league}: actual vs predicted",
                output_path=league_dir / "actual_vs_predicted.png",
            )

    return per_league_metrics


def debug_prediction_summary(pred_df, model_output_dir):
    summary = {
        "num_holdout_rows": int(len(pred_df)),
        "actual_min": float(pred_df["Season Result"].min()),
        "actual_max": float(pred_df["Season Result"].max()),
        "actual_mean": float(pred_df["Season Result"].mean()),
        "predicted_min": float(pred_df["Predicted Season Result"].min()),
        "predicted_max": float(pred_df["Predicted Season Result"].max()),
        "predicted_mean": float(pred_df["Predicted Season Result"].mean()),
        "residual_min": float(pred_df["Residual"].min()),
        "residual_max": float(pred_df["Residual"].max()),
        "residual_mean": float(pred_df["Residual"].mean()),
        "prediction_interval_coverage": float(pred_df["Within Prediction Interval"].mean()),
    }

    with open(model_output_dir / "debug_prediction_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    top10 = pred_df.copy()
    top10["AbsResidual"] = top10["Residual"].abs()
    top10 = top10.sort_values("AbsResidual", ascending=False).head(10)
    top10.to_csv(model_output_dir / "debug_top10_worst_predictions.csv", index=True)


def debug_worst_holdout_prediction(
    model_name,
    idata,
    holdout_raw,
    holdout_df,
    prediction_summary,
    feature_cols,
    league_to_idx,
    preprocessing_artifacts,
    model_output_dir,
    coefficient_hdi_prob,
):
    pred_df = build_holdout_prediction_df(holdout_raw, prediction_summary).copy()
    pred_df["AbsResidual"] = pred_df["Residual"].abs()

    worst_idx = pred_df["AbsResidual"].idxmax()
    summary_row = pred_df.loc[worst_idx]

    raw_row = holdout_raw.loc[worst_idx]
    transformed_row = holdout_df.loc[worst_idx]

    league = str(raw_row["League"])
    group_idx = league_to_idx[league]

    alpha_mean = float(idata.posterior["alpha"].mean(dim=("chain", "draw")).values[group_idx])

    beta_summary = extract_league_coefficient_summary(idata, hdi_prob=coefficient_hdi_prob)
    beta_summary = beta_summary[beta_summary["League"] == league].set_index("Feature")

    transformed_x = transformed_row[feature_cols].astype(float).values

    original_values = []
    for feature in feature_cols:
        if feature in raw_row.index:
            original_values.append(raw_row[feature])
        else:
            original_values.append(transformed_row[feature])

    contributions_df = pd.DataFrame(
        {
            "Feature": feature_cols,
            "OriginalValue": original_values,
            "TransformedValue": transformed_x,
            "PosteriorMeanBeta": [float(beta_summary.loc[f, "MeanCoefficient"]) for f in feature_cols],
            "BetaHDILower": [float(beta_summary.loc[f, "HDILower"]) for f in feature_cols],
            "BetaHDIUpper": [float(beta_summary.loc[f, "HDIUpper"]) for f in feature_cols],
            "ProbabilityPositive": [float(beta_summary.loc[f, "ProbabilityPositive"]) for f in feature_cols],
            "RobustNonZero": [bool(beta_summary.loc[f, "RobustNonZero"]) for f in feature_cols],
        }
    )
    contributions_df["Contribution"] = (
        contributions_df["TransformedValue"] * contributions_df["PosteriorMeanBeta"]
    )
    contributions_df = contributions_df.sort_values(
        by="Contribution",
        key=lambda s: s.abs(),
        ascending=False,
    )

    beta_sum = float(contributions_df["Contribution"].sum())
    pred_std = float(alpha_mean + beta_sum)
    reconstructed_prediction = float(
        pred_std * preprocessing_artifacts.y_std + preprocessing_artifacts.y_mean
    )

    inputs_df = pd.DataFrame(
        {
            "Feature": feature_cols,
            "OriginalValue": original_values,
            "TransformedValue": transformed_x,
        }
    )
    inputs_df.to_csv(model_output_dir / "debug_worst_prediction_inputs.csv", index=False)
    contributions_df.to_csv(
        model_output_dir / "debug_worst_prediction_contributions.csv",
        index=False,
    )

    debug_summary = {
        "model_name": model_name,
        "worst_row_index": int(worst_idx),
        "club": str(raw_row["Club"]),
        "league": league,
        "season": str(raw_row["Season"]),
        "prev_season": str(raw_row["Prev Season"]),
        "actual_season_result": float(summary_row["Season Result"]),
        "predicted_season_result": float(summary_row["Predicted Season Result"]),
        "prediction_hdi_lower": float(summary_row["Prediction HDI Lower"]),
        "prediction_hdi_upper": float(summary_row["Prediction HDI Upper"]),
        "within_prediction_interval": bool(summary_row["Within Prediction Interval"]),
        "residual": float(summary_row["Residual"]),
        "abs_residual": float(summary_row["AbsResidual"]),
        "alpha_mean": alpha_mean,
        "sum_of_beta_contributions": beta_sum,
        "pred_std": pred_std,
        "y_mean": float(preprocessing_artifacts.y_mean),
        "y_std": float(preprocessing_artifacts.y_std),
        "reconstructed_prediction": reconstructed_prediction,
    }

    with open(model_output_dir / "debug_worst_prediction.json", "w", encoding="utf-8") as f:
        json.dump(debug_summary, f, indent=2)


def summarise_run(model_name, idata, pred_df, per_league_metrics, run_config):
    y_true = pred_df["Season Result"].values
    y_pred = pred_df["Predicted Season Result"].values

    mean_league_rmse = (
        float(per_league_metrics["rmse"].dropna().mean())
        if per_league_metrics["rmse"].notna().any()
        else None
    )
    mean_league_mae = (
        float(per_league_metrics["mae"].dropna().mean())
        if per_league_metrics["mae"].notna().any()
        else None
    )
    mean_league_r2 = (
        float(per_league_metrics["r2"].dropna().mean())
        if per_league_metrics["r2"].notna().any()
        else None
    )
    mean_league_coverage = (
        float(per_league_metrics["prediction_interval_coverage"].dropna().mean())
        if per_league_metrics["prediction_interval_coverage"].notna().any()
        else None
    )

    return {
        "model_name": model_name,
        "mode": run_config.mode,
        "predictor_timing": "previous_season",
        "holdout_season": run_config.holdout_season,
        "overall_rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "overall_mae": float(mean_absolute_error(y_true, y_pred)),
        "overall_r2": float(r2_score(y_true, y_pred)),
        "overall_prediction_interval_coverage": float(pred_df["Within Prediction Interval"].mean()),
        "mean_league_rmse": mean_league_rmse,
        "mean_league_mae": mean_league_mae,
        "mean_league_r2": mean_league_r2,
        "mean_league_prediction_interval_coverage": mean_league_coverage,
        "num_leagues_with_holdout": int((per_league_metrics["holdout_rows"] > 0).sum()),
        "divergences": int(idata.sample_stats["diverging"].sum().values),
        "max_rhat": float(az.rhat(idata).to_array().max().values),
    }


def save_model_outputs(
    model_name,
    idata,
    train_df,
    holdout_raw,
    holdout_df,
    prediction_summary,
    summary_row,
    preprocessing_artifacts,
    league_to_idx,
    comparison_output_dir,
    coefficient_hdi_prob,
):
    model_output_dir = comparison_output_dir / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)

    with open(model_output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary_row, f, indent=2)

    league_summary = az.summary(idata, var_names=["alpha", "beta"], round_to=4)
    league_summary.to_csv(model_output_dir / "league_parameter_summary.csv")

    pred_df = build_holdout_prediction_df(holdout_raw, prediction_summary)
    pred_df.to_csv(model_output_dir / "holdout_predictions.csv", index=False)

    per_league_metrics = save_per_league_outputs(
        idata=idata,
        train_df=train_df,
        pred_df=pred_df,
        model_output_dir=model_output_dir,
        coefficient_hdi_prob=coefficient_hdi_prob,
    )

    debug_prediction_summary(pred_df=pred_df, model_output_dir=model_output_dir)

    debug_worst_holdout_prediction(
        model_name=model_name,
        idata=idata,
        holdout_raw=holdout_raw,
        holdout_df=holdout_df,
        prediction_summary=prediction_summary,
        feature_cols=preprocessing_artifacts.feature_cols,
        league_to_idx=league_to_idx,
        preprocessing_artifacts=preprocessing_artifacts,
        model_output_dir=model_output_dir,
        coefficient_hdi_prob=coefficient_hdi_prob,
    )

    report_lines = [
        f"Model: {model_name}",
        "Predictor timing: previous season",
        f"Holdout season: {summary_row['holdout_season']}",
        "",
        "Overall holdout metrics",
        f"RMSE: {summary_row['overall_rmse']:.6f}",
        f"MAE: {summary_row['overall_mae']:.6f}",
        f"R2: {summary_row['overall_r2']:.6f}",
        f"Overall prediction interval coverage: {summary_row['overall_prediction_interval_coverage']:.6f}",
        "",
        "Mean per-league holdout metrics",
        f"Mean league RMSE: {summary_row['mean_league_rmse']}",
        f"Mean league MAE: {summary_row['mean_league_mae']}",
        f"Mean league R2: {summary_row['mean_league_r2']}",
        f"Mean league prediction interval coverage: {summary_row['mean_league_prediction_interval_coverage']}",
        "",
        "Sampler diagnostics",
        f"Divergences: {summary_row['divergences']}",
        f"Max R-hat: {summary_row['max_rhat']:.6f}",
        "",
        "League-level metrics",
        per_league_metrics.to_string(index=False),
        "",
        "Debug files saved:",
        "debug_prediction_summary.json",
        "debug_top10_worst_predictions.csv",
        "debug_worst_prediction.json",
        "debug_worst_prediction_inputs.csv",
        "debug_worst_prediction_contributions.csv",
    ]
    with open(model_output_dir / "report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    return per_league_metrics


def main():
    args = parse_mode_args("Debug Bayesian normal formulation for the league-specific question.")
    run_config = build_run_config(args.mode)
    comparison_output_dir = run_config.output_dir / "model_comparison_debug"
    comparison_output_dir.mkdir(parents=True, exist_ok=True)
    save_run_config(run_config, comparison_output_dir)

    np.random.seed(run_config.random_seed)

    print("Loading and cleaning data...")
    df = load_and_clean_dataset(INPUT_PATH)
    df = maybe_filter_development_leagues(df, run_config)

    print("Building club-season dataset with previous-season predictors...")
    lagged_df = build_lagged_dataset(df)
    print(f"Lagged rows available: {len(lagged_df)}")

    print(f"Creating shared train/holdout split with holdout season {HOLDOUT_SEASON}...")
    train_raw, holdout_raw = split_train_holdout(lagged_df, run_config.holdout_season)

    print("Fitting preprocessing on training data only...")
    train_df, preprocessing_artifacts = fit_preprocessor(train_raw)
    holdout_df = apply_preprocessor(holdout_raw, preprocessing_artifacts)

    print(f"Train rows: {len(train_df)}")
    print(f"Holdout rows ({run_config.holdout_season}): {len(holdout_df)}")
    print(f"Running mode: {run_config.mode}")

    summary_rows = []

    for specification in MODEL_SPECS:
        model_name = specification["model_name"]
        print(f"\n=== Fitting {model_name} ===")

        idata, league_to_idx = build_and_fit_specification(
            train_df=train_df,
            feature_cols=preprocessing_artifacts.feature_cols,
            likelihood=specification["likelihood"],
            run_config=run_config,
        )

        prediction_summary = predict_holdout(
            idata=idata,
            holdout_df=holdout_df,
            feature_cols=preprocessing_artifacts.feature_cols,
            league_to_idx=league_to_idx,
            y_mean=preprocessing_artifacts.y_mean,
            y_std=preprocessing_artifacts.y_std,
            interval_prob=run_config.prediction_hdi_prob,
        )

        pred_df = build_holdout_prediction_df(holdout_raw, prediction_summary)
        per_league_metrics = compute_per_league_metrics(train_df, pred_df)

        summary_row = summarise_run(
            model_name=model_name,
            idata=idata,
            pred_df=pred_df,
            per_league_metrics=per_league_metrics,
            run_config=run_config,
        )
        summary_rows.append(summary_row)

        save_model_outputs(
            model_name=model_name,
            idata=idata,
            train_df=train_df,
            holdout_raw=holdout_raw,
            holdout_df=holdout_df,
            prediction_summary=prediction_summary,
            summary_row=summary_row,
            preprocessing_artifacts=preprocessing_artifacts,
            league_to_idx=league_to_idx,
            comparison_output_dir=comparison_output_dir,
            coefficient_hdi_prob=run_config.coefficient_hdi_prob,
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(comparison_output_dir / "model_comparison_summary.csv", index=False)
    summary_df.to_json(
        comparison_output_dir / "model_comparison_summary.json",
        orient="records",
        indent=2,
    )

    best_model = summary_df.iloc[0].to_dict()
    with open(comparison_output_dir / "recommended_model.json", "w", encoding="utf-8") as f:
        json.dump(best_model, f, indent=2)

    report_lines = [
        "Bayesian normal-formulation debugging run",
        "Question focus: previous-season factors by league",
        f"Mode: {run_config.mode}",
        f"Holdout season: {run_config.holdout_season}",
        "",
        summary_df.to_string(index=False),
        "",
        "Per-model debug outputs saved under:",
        str(comparison_output_dir),
    ]

    with open(comparison_output_dir / "model_comparison_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    print("\n=== DEBUG RUN SUMMARY ===")
    print(summary_df.to_string(index=False))
    print(f"Saved debug outputs to: {comparison_output_dir.resolve()}")


if __name__ == "__main__":
    main()
