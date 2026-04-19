"""
Comparison plots between the elastic net, Bayesian, and random forest models.

Reads saved outputs from all three pipelines — no model rerunning needed.
All models must have been run before running this script.

Outputs saved to: comparison_outputs/<mode>/
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

LEAGUE_ORDER = [
    "Premier League", "Championship", "League One", "League Two",
    "LaLiga", "Bundesliga", "Ligue One", "Serie A",
]

DEFAULT_COLOUR = "black"
MODEL_COLOURS = {"Elastic Net": "blue", "Bayesian": "red", "Random Forest": "green"}


def parse_args():
    parser = argparse.ArgumentParser(description="Comparison plots for elastic net vs Bayesian vs Random Forest.")
    parser.add_argument(
        "--mode",
        choices=["development", "final"],
        default="development",
        help="Which run mode's outputs to compare.",
    )
    return parser.parse_args()


def get_paths(mode):
    en_base = ROOT / "football_elastic_net" / "outputs" / mode
    bayes_base = ROOT / "football_bayes_project" / "outputs" / mode
    rf_base = ROOT / "outputs_rf_lagged"
    output_dir = ROOT / "comparison_outputs" / mode
    output_dir.mkdir(parents=True, exist_ok=True)
    return en_base, bayes_base, rf_base, output_dir


def slugify(value):
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def sort_leagues(leagues):
    known = [l for l in LEAGUE_ORDER if l in leagues]
    unknown = sorted([l for l in leagues if l not in LEAGUE_ORDER])
    return known + unknown


def load_metrics(en_base, bayes_base, rf_base):
    en = pd.read_csv(en_base / "overall_metrics.csv")
    en = en.rename(columns={"league": "League", "holdout_r2": "r2", "holdout_rmse": "rmse", "holdout_mae": "mae"})
    en = en[["League", "r2", "rmse", "mae"]]

    bayes = pd.read_csv(bayes_base / "per_league" / "per_league_metrics.csv")
    bayes = bayes[["League", "r2", "rmse", "mae"]]

    rf_path = rf_base / "per_league" / "per_league_metrics.csv"
    rf = pd.read_csv(rf_path)[["League", "r2", "rmse", "mae"]] if rf_path.exists() else pd.DataFrame(columns=["League", "r2", "rmse", "mae"])

    return en, bayes, rf


def load_coefficients(en_base, bayes_base):
    en = pd.read_csv(en_base / "all_coefficients.csv")
    en = en.rename(columns={"Coefficient": "coef"})

    bayes = pd.read_csv(bayes_base / "per_league" / "all_league_coefficients.csv")
    bayes = bayes.rename(columns={"MeanCoefficient": "coef", "Prev Season Result": "Prev Points"})

    return en, bayes


def load_holdout_predictions(en_base, bayes_base, rf_base):
    en_coefs_raw = pd.read_csv(en_base / "all_coefficients.csv")
    slug_to_name = {slugify(l): l for l in en_coefs_raw["League"].unique()}

    en_frames = []
    for league_dir in sorted(en_base.iterdir()):
        if not league_dir.is_dir():
            continue
        path = league_dir / "holdout_predictions.csv"
        if path.exists():
            df = pd.read_csv(path)
            df["League"] = slug_to_name.get(league_dir.name, league_dir.name)
            en_frames.append(df)

    en = pd.concat(en_frames, ignore_index=True) if en_frames else pd.DataFrame()
    en = en.rename(columns={"Actual": "Points", "Predicted": "Predicted Points"})
    if not en.empty and "Residual" not in en.columns:
        en["Residual"] = en["Points"] - en["Predicted Points"]

    bayes_path = bayes_base / "per_league" / "all_holdout_predictions.csv"
    bayes = pd.read_csv(bayes_path) if bayes_path.exists() else pd.DataFrame()
    if not bayes.empty:
        bayes = bayes.rename(columns={
            "Season Result": "Points",
            "Predicted Season Result": "Predicted Points",
        })

    rf_path = rf_base / "per_league" / "all_holdout_predictions.csv"
    rf = pd.read_csv(rf_path) if rf_path.exists() else pd.DataFrame()
    if not rf.empty and "Residual" not in rf.columns:
        rf["Residual"] = rf["Points"] - rf["Predicted Points"]

    return en, bayes, rf


def plot_metric_comparison(en_metrics, bayes_metrics, rf_metrics, metric, ylabel, title, output_path, leagues):
    en_vals = [en_metrics.loc[en_metrics["League"] == l, metric].values[0]
               if l in en_metrics["League"].values else np.nan for l in leagues]
    bayes_vals = [bayes_metrics.loc[bayes_metrics["League"] == l, metric].values[0]
                  if l in bayes_metrics["League"].values else np.nan for l in leagues]
    rf_vals = [rf_metrics.loc[rf_metrics["League"] == l, metric].values[0]
               if (not rf_metrics.empty and l in rf_metrics["League"].values) else np.nan for l in leagues]

    x = np.arange(len(leagues))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(8, len(leagues) * 1.5), 5))
    bars_en = ax.bar(x - width, en_vals, width, label="Elastic Net", color=MODEL_COLOURS["Elastic Net"], alpha=0.85)
    bars_bayes = ax.bar(x, bayes_vals, width, label="Bayesian", color=MODEL_COLOURS["Bayesian"], alpha=0.85)
    bars_rf = ax.bar(x + width, rf_vals, width, label="Random Forest", color=MODEL_COLOURS["Random Forest"], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(leagues, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.axhline(0, color="black", linewidth=0.6)

    for bar in list(bars_en) + list(bars_bayes) + list(bars_rf):
        h = bar.get_height()
        if not np.isnan(h):
            ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def plot_coefficient_heatmaps(en_coefs, bayes_coefs, output_path, leagues):
    feature_order = [
        "Prev Squad Value", "Prev Goals For", "Prev Goals Against",
        "Prev Expenditure", "Prev Income", "Prev Avg. Squad Age",
        "Prev Points", "Prev Expenditure_missing", "Prev Income_missing",
    ]

    def to_matrix(coefs, coef_col):
        pivot = coefs.pivot_table(index="Feature", columns="League", values=coef_col, aggfunc="first")
        pivot = pivot.reindex(
            index=[f for f in feature_order if f in pivot.index],
            columns=[l for l in leagues if l in pivot.columns],
        )
        return pivot

    en_mat = to_matrix(en_coefs, "coef")
    bayes_mat = to_matrix(bayes_coefs, "coef")

    n_leagues = len(leagues)
    fig, axes = plt.subplots(1, 2, figsize=(max(14, n_leagues * 1.8), 6))

    for ax, mat, title in zip(axes, [en_mat, bayes_mat], ["Elastic Net coefficients", "Bayesian posterior mean coefficients"]):
        vmax = np.nanmax(np.abs(mat.values))
        vmin = -vmax
        im = ax.imshow(mat.values, aspect="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(mat.columns)))
        ax.set_xticklabels(mat.columns, rotation=30, ha="right", fontsize=9)
        ax.set_yticks(range(len(mat.index)))
        ax.set_yticklabels(mat.index, fontsize=9)
        ax.set_title(title, fontsize=11)
        plt.colorbar(im, ax=ax, shrink=0.8)

        for i in range(len(mat.index)):
            for j in range(len(mat.columns)):
                val = mat.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val + 0.0:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if abs(val) > 0.6 * vmax else "black")

    plt.suptitle("Feature coefficients by league (Elastic Net vs Bayesian)", fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def plot_coefficient_comparison(en_coefs, bayes_coefs, output_path, leagues):
    feature_order = [
        "Prev Squad Value", "Prev Goals For", "Prev Goals Against",
        "Prev Expenditure", "Prev Income", "Prev Avg. Squad Age",
        "Prev Points", "Prev Expenditure_missing", "Prev Income_missing",
    ]

    n_cols = min(len(leagues), 3)
    n_rows = int(np.ceil(len(leagues) / n_cols))
    bar_height = 0.35
    subplot_h = max(3.5, len(feature_order) * bar_height * 2 + 1.5)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 7, n_rows * subplot_h))
    axes = np.atleast_1d(axes).flatten()

    for idx, league in enumerate(leagues):
        ax = axes[idx]

        en_league = en_coefs[en_coefs["League"] == league].set_index("Feature")["coef"]
        bayes_league = bayes_coefs[bayes_coefs["League"] == league].set_index("Feature")["coef"]

        features = [f for f in feature_order if f in en_league.index or f in bayes_league.index]
        features.sort(key=lambda f: (abs(en_league.get(f, 0)) + abs(bayes_league.get(f, 0))) / 2)

        en_vals = [en_league.get(f, 0) for f in features]
        bayes_vals = [bayes_league.get(f, 0) for f in features]

        y = np.arange(len(features))

        ax.barh(y + bar_height / 2, en_vals, bar_height,
                color=MODEL_COLOURS["Elastic Net"], alpha=0.85, label="Elastic Net")
        ax.barh(y - bar_height / 2, bayes_vals, bar_height,
                color=MODEL_COLOURS["Bayesian"], alpha=0.85, label="Bayesian")

        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(features, fontsize=8)
        ax.set_title(league, fontsize=10)
        ax.set_xlabel("Coefficient", fontsize=8)
        if idx == 0:
            ax.legend(fontsize=8, loc="lower right")

    for idx in range(len(leagues), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("Feature coefficients: Elastic Net vs Bayesian", fontsize=13)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def _grid_dims(n_leagues, max_cols=4):
    n_cols = min(n_leagues, max_cols)
    n_rows = int(np.ceil(n_leagues / n_cols))
    return n_cols, n_rows


def plot_residuals(en_preds, bayes_preds, rf_preds, output_path, leagues):
    n_models = 3
    n_cols, n_rows = _grid_dims(len(leagues))
    fig, axes = plt.subplots(n_models * n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4 * n_models))
    axes = np.atleast_2d(axes)
    fig.suptitle("Residuals by league (Actual − Predicted)", fontsize=13)

    model_list = [
        (en_preds, "Elastic Net"),
        (bayes_preds, "Bayesian"),
        (rf_preds, "Random Forest"),
    ]

    for idx, league in enumerate(leagues):
        col = idx % n_cols
        for model_row, (preds, model_name) in enumerate(model_list):
            row = (idx // n_cols) * n_models + model_row
            ax = axes[row, col]

            league_df = preds[preds["League"] == league] if not preds.empty else pd.DataFrame()

            if league_df.empty:
                ax.set_visible(False)
                continue

            ax.scatter(league_df["Points"], league_df["Residual"],
                       color=MODEL_COLOURS[model_name], alpha=0.7, s=30)
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

            if col == 0:
                ax.set_ylabel(f"{model_name}\nResidual (points)", fontsize=9)
            if model_row == 0:
                ax.set_title(league, fontsize=10)
            if model_row == n_models - 1:
                ax.set_xlabel("Actual Points", fontsize=9)

    for idx in range(len(leagues), n_rows * n_cols):
        col = idx % n_cols
        for r in range(n_models):
            row = (idx // n_cols) * n_models + r
            if row < axes.shape[0]:
                axes[row, col].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def plot_actual_vs_predicted(en_preds, bayes_preds, rf_preds, output_path, leagues,
                             en_metrics=None, bayes_metrics=None, rf_metrics=None):
    n_cols, n_rows = _grid_dims(len(leagues))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    axes = np.atleast_1d(axes).flatten()
    fig.suptitle("Actual vs predicted by league", fontsize=13)

    for idx, league in enumerate(leagues):
        ax = axes[idx]

        all_empty = True
        mn_vals, mx_vals = [], []

        for preds, model_name, metrics in [
            (en_preds, "Elastic Net", en_metrics),
            (bayes_preds, "Bayesian", bayes_metrics),
            (rf_preds, "Random Forest", rf_metrics),
        ]:
            league_df = preds[preds["League"] == league] if not preds.empty else pd.DataFrame()
            if league_df.empty:
                continue

            all_empty = False

            r2 = None
            if metrics is not None and not metrics.empty:
                row_match = metrics[metrics["League"] == league]
                if not row_match.empty:
                    r2 = row_match["r2"].values[0]

            label = f"{model_name} (R²={r2:.3f})" if r2 is not None and not np.isnan(r2) else model_name
            ax.scatter(league_df["Points"], league_df["Predicted Points"],
                       color=MODEL_COLOURS[model_name], alpha=0.7, s=30, label=label)

            mn_vals += [league_df["Points"].min(), league_df["Predicted Points"].min()]
            mx_vals += [league_df["Points"].max(), league_df["Predicted Points"].max()]

        if all_empty:
            ax.set_visible(False)
            continue

        mn, mx = min(mn_vals), max(mx_vals)
        pad = 0.05 * (mx - mn) if not np.isclose(mx, mn) else 1.0
        ax.plot([mn - pad, mx + pad], [mn - pad, mx + pad], "k--", linewidth=0.8)

        ax.set_title(league, fontsize=10)
        ax.set_xlabel("Actual Points", fontsize=9)
        ax.set_ylabel("Predicted Points", fontsize=9)
        ax.legend(fontsize=7, loc="upper left")

    for idx in range(len(leagues), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def main():
    args = parse_args()
    en_base, bayes_base, rf_base, output_dir = get_paths(args.mode)

    missing = [p for p in [en_base, bayes_base] if not p.exists()]
    if missing:
        for p in missing:
            print(f"Missing outputs directory: {p}")
        print("Run elastic net and Bayesian models first.")
        return

    rf_available = (rf_base / "per_league" / "per_league_metrics.csv").exists()
    if not rf_available:
        print("Warning: random forest outputs not found at outputs_rf_lagged/ — RF will be skipped.")
        print("Run: python random_forest_lagged.py")

    print(f"Loading outputs (mode = {args.mode})...")
    en_metrics, bayes_metrics, rf_metrics = load_metrics(en_base, bayes_base, rf_base)
    en_coefs, bayes_coefs = load_coefficients(en_base, bayes_base)
    en_preds, bayes_preds, rf_preds = load_holdout_predictions(en_base, bayes_base, rf_base)

    all_leagues = set(en_metrics["League"]) | set(bayes_metrics["League"])
    if not rf_metrics.empty:
        all_leagues |= set(rf_metrics["League"])
    leagues = sort_leagues(all_leagues)
    print(f"Leagues found: {leagues}")

    print("Generating plots...")

    plot_metric_comparison(
        en_metrics, bayes_metrics, rf_metrics,
        metric="r2", ylabel="Holdout R²",
        title="Holdout R² by league: Elastic Net vs Bayesian vs Random Forest",
        output_path=output_dir / "r2_comparison.png",
        leagues=leagues,
    )

    plot_metric_comparison(
        en_metrics, bayes_metrics, rf_metrics,
        metric="rmse", ylabel="Holdout RMSE (points)",
        title="Holdout RMSE by league: Elastic Net vs Bayesian vs Random Forest",
        output_path=output_dir / "rmse_comparison.png",
        leagues=leagues,
    )

    plot_metric_comparison(
        en_metrics, bayes_metrics, rf_metrics,
        metric="mae", ylabel="Holdout MAE (points)",
        title="Holdout MAE by league: Elastic Net vs Bayesian vs Random Forest",
        output_path=output_dir / "mae_comparison.png",
        leagues=leagues,
    )

    english_leagues = [l for l in ["Premier League", "Championship", "League One", "League Two"] if l in leagues]
    european_leagues = [l for l in ["Premier League", "LaLiga", "Bundesliga", "Ligue One", "Serie A"] if l in leagues]

    plot_coefficient_heatmaps(
        en_coefs, bayes_coefs,
        output_path=output_dir / "coefficient_heatmaps_english.png",
        leagues=english_leagues,
    )

    plot_coefficient_heatmaps(
        en_coefs, bayes_coefs,
        output_path=output_dir / "coefficient_heatmaps_european.png",
        leagues=european_leagues,
    )

    plot_coefficient_comparison(
        en_coefs, bayes_coefs,
        output_path=output_dir / "coefficient_comparison.png",
        leagues=leagues,
    )

    plot_residuals(
        en_preds, bayes_preds, rf_preds,
        output_path=output_dir / "residuals.png",
        leagues=leagues,
    )

    plot_actual_vs_predicted(
        en_preds, bayes_preds, rf_preds,
        output_path=output_dir / "actual_vs_predicted.png",
        leagues=leagues,
        en_metrics=en_metrics,
        bayes_metrics=bayes_metrics,
        rf_metrics=rf_metrics,
    )

    print(f"\nAll plots saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
