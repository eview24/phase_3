"""
Lean Bayesian formulation comparison for the main research question.

This script compares only the two formulations that can directly answer:
which previous-season factors matter most for predicting next-season
performance in each league?

Compared formulations:
1. normal likelihood with league-specific varying slopes
2. student-t likelihood with league-specific varying slopes

It uses the same lagged previous-season pipeline as bayes_model.py and saves
only the outputs needed to choose the better formulation.
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
    },
    {
        "model_name": "student_t_varying_slopes",
        "likelihood": "student_t",
    },
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

        if likelihood == "student_t":
            nu_offset = pm.Exponential("nu_offset", lam=1 / 30)
            nu = pm.Deterministic("nu", nu_offset + 2.0)
            pm.StudentT("y_obs", nu=nu, mu=mu, sigma=sigma, observed=y)
        else:
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


def is_well_converged(summary_row, mode):
    rhat_threshold = 1.05 if mode == "development" else 1.01
    return (summary_row["divergences"] == 0) and (summary_row["max_rhat"] <= rhat_threshold)


def save_model_outputs(
    model_name,
    idata,
    train_df,
    holdout_df,
    prediction_summary,
    summary_row,
    comparison_output_dir,
    coefficient_hdi_prob,
):
    model_output_dir = comparison_output_dir / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)

    with open(model_output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary_row, f, indent=2)

    league_summary = az.summary(idata, var_names=["alpha", "beta"], round_to=4)
    league_summary.to_csv(model_output_dir / "league_parameter_summary.csv")

    pred_df = build_holdout_prediction_df(holdout_df, prediction_summary)
    pred_df.to_csv(model_output_dir / "holdout_predictions.csv", index=False)

    per_league_metrics = save_per_league_outputs(
        idata=idata,
        train_df=train_df,
        pred_df=pred_df,
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
    ]
    with open(model_output_dir / "report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    return per_league_metrics


def main():
    args = parse_mode_args("Compare Bayesian formulations for the league-specific question.")
    run_config = build_run_config(args.mode)
    comparison_output_dir = run_config.output_dir / "model_comparison"
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
            holdout_df=holdout_raw,
            prediction_summary=prediction_summary,
            summary_row=summary_row,
            comparison_output_dir=comparison_output_dir,
            coefficient_hdi_prob=run_config.coefficient_hdi_prob,
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df["well_converged"] = summary_df.apply(
        lambda row: is_well_converged(row.to_dict(), run_config.mode),
        axis=1,
    )
    summary_df = summary_df.sort_values(
        by=[
            "well_converged",
            "mean_league_rmse",
            "overall_rmse",
            "mean_league_mae",
            "overall_mae",
            "overall_r2",
            "max_rhat",
            "divergences",
        ],
        ascending=[False, True, True, True, True, False, True, True],
    )

    summary_df.to_csv(comparison_output_dir / "model_comparison_summary.csv", index=False)
    summary_df.to_json(
        comparison_output_dir / "model_comparison_summary.json",
        orient="records",
        indent=2,
    )

    best_model = summary_df.iloc[0].to_dict()
    with open(comparison_output_dir / "recommended_model.json", "w", encoding="utf-8") as f:
        json.dump(best_model, f, indent=2)

    rhat_threshold = 1.05 if run_config.mode == "development" else 1.01
    report_lines = [
        "Bayesian formulation comparison",
        "Question focus: previous-season factors by league",
        f"Mode: {run_config.mode}",
        f"Holdout season: {run_config.holdout_season}",
        "",
        "Formulations compared:",
    ]
    report_lines.extend(f"- {spec['model_name']}" for spec in MODEL_SPECS)
    report_lines.extend(
        [
            "",
            "Ranking priority:",
            f"0. well_converged (divergences = 0 and max_rhat <= {rhat_threshold})",
            "1. mean_league_rmse",
            "2. overall_rmse",
            "3. mean_league_mae",
            "4. overall_mae",
            "5. overall_r2",
            "6. max_rhat",
            "7. divergences",
            "",
            "Recommended formulation:",
            str(best_model["model_name"]),
            "",
            summary_df.to_string(index=False),
        ]
    )

    with open(comparison_output_dir / "model_comparison_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    print("\n=== MODEL COMPARISON SUMMARY ===")
    print(summary_df.to_string(index=False))
    print(f"\nRecommended formulation: {best_model['model_name']}")
    print(f"Saved outputs to: {comparison_output_dir.resolve()}")


if __name__ == "__main__":
    main()
