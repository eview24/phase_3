"""
Random Forest with CV, Optuna tuning, permutation importance and SHAP.

Improvements over the original random_forest.py:
  1. Strips whitespace from column names (fixes the Championship crash).
  2. Replaces single train/test split with 5-fold CV for model selection.
  3. Adds baselines (DummyRegressor, Ridge, default RF) for context.
  4. Tunes RF hyperparameters with Optuna on CV R2.
  5. Reports both impurity-based and permutation feature importances.
  6. Produces TreeSHAP summary + bar plots per league.
  7. Also trains a pooled model (all leagues, league one-hot encoded).
  8. Saves metrics, best params and plots to ./outputs_rf/.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shap
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
N_SPLITS = 5
N_TRIALS = 40          # Optuna trials per model
MIN_ROWS = 40          # Skip leagues smaller than this
DATA_PATH = "dataset.ods"
OUTPUT_DIR = Path("outputs_rf")
OUTPUT_DIR.mkdir(exist_ok=True)

FEATURES = ["Expenditure", "Income", "Squad Value", "Avg. Squad Age"]
TARGET = "Points per game"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _to_numeric(s: pd.Series) -> pd.Series:
    """Parse values like '€12.5m' or '€450k' into floats."""
    return pd.to_numeric(
        s.astype(str)
        .str.replace("€", "", regex=False)
        .str.replace("m", "e6", regex=False)
        .str.replace("k", "e3", regex=False),
        errors="coerce",
    )


def load_league(file_path: str, sheet: str) -> pd.DataFrame:
    df = pd.read_excel(file_path, engine="odf", sheet_name=sheet)
    df.columns = df.columns.str.strip()  # fixes the Championship KeyError
    df = df[FEATURES + [TARGET]].copy()
    for c in df.columns:
        df[c] = _to_numeric(df[c])
    return df.dropna().reset_index(drop=True)


def load_pooled(file_path: str) -> tuple[pd.DataFrame, list[str]]:
    xls = pd.ExcelFile(file_path, engine="odf")
    frames = []
    for sheet in xls.sheet_names:
        d = load_league(file_path, sheet)
        d["league"] = sheet
        frames.append(d)
    pooled = pd.concat(frames, ignore_index=True)
    pooled = pd.get_dummies(pooled, columns=["league"], prefix="lg", dtype=float)
    lg_cols = [c for c in pooled.columns if c.startswith("lg_")]
    return pooled, lg_cols


# ---------------------------------------------------------------------------
# Modelling helpers
# ---------------------------------------------------------------------------
def cv_r2(model, X, y) -> tuple[float, float]:
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    # n_jobs=1 inside caller's model, -1 on CV to avoid oversubscription
    scores = cross_val_score(model, X, y, scoring="r2", cv=kf, n_jobs=-1)
    return float(scores.mean()), float(scores.std())


def tune_rf(X_train, y_train, n_trials: int = N_TRIALS) -> optuna.Study:
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_float("max_features", 0.3, 1.0),
            "max_samples": trial.suggest_float("max_samples", 0.5, 1.0),
            "random_state": RANDOM_STATE,
            "n_jobs": 1,  # parallelise over CV folds instead, avoids oversubscription
        }
        model = RandomForestRegressor(**params)
        return cross_val_score(
            model, X_train, y_train, scoring="r2", cv=kf, n_jobs=-1
        ).mean()

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study


def evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    return {
        "r2": float(r2_score(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
    }


def save_shap(best_rf, X_test, feature_names, tag: str) -> None:
    explainer = shap.TreeExplainer(
        best_rf, feature_perturbation="tree_path_dependent"
    )
    shap_values = explainer.shap_values(X_test)

    plt.figure()
    shap.summary_plot(
        shap_values, X_test, feature_names=feature_names,
        show=False, max_display=12,
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"shap_summary_{tag}.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(
        shap_values, X_test, feature_names=feature_names,
        plot_type="bar", show=False, max_display=12,
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"shap_bar_{tag}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Dependence plot for top (non-dummy) feature
    mean_abs = np.abs(shap_values).mean(axis=0)
    # Prefer non-league features for dependence plot
    core_idx = [i for i, f in enumerate(feature_names) if not f.startswith("lg_")]
    top = int(core_idx[np.argmax(mean_abs[core_idx])])
    plt.figure()
    shap.dependence_plot(
        top, shap_values, X_test, feature_names=feature_names, show=False,
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"shap_dependence_{tag}.png", dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Per-league run
# ---------------------------------------------------------------------------
def run_league(sheet: str, file_path: str) -> dict | None:
    print(f"\n===== {sheet} =====")
    df = load_league(file_path, sheet)
    n = len(df)
    print(f"Rows after cleaning: {n}")
    if n < MIN_ROWS:
        print(f"Skipping: fewer than {MIN_ROWS} rows.")
        return None

    X = df[FEATURES].values
    y = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # Baselines evaluated with CV on the training split
    baselines = {
        "dummy_mean": DummyRegressor(strategy="mean"),
        "ridge": Ridge(random_state=RANDOM_STATE),
        "rf_default": RandomForestRegressor(
            n_estimators=500, random_state=RANDOM_STATE, n_jobs=1
        ),
    }
    rows = []
    for name, mdl in baselines.items():
        m, s = cv_r2(mdl, X_train, y_train)
        rows.append({"model": name, "cv_r2_mean": m, "cv_r2_std": s})

    # Optuna-tuned RF
    study = tune_rf(X_train, y_train)
    best_params = study.best_params
    rows.append({"model": "rf_optuna", "cv_r2_mean": study.best_value, "cv_r2_std": None})

    # Refit on full training set, evaluate on held-out test
    best_rf = RandomForestRegressor(
        **best_params, random_state=RANDOM_STATE, n_jobs=-1
    )
    best_rf.fit(X_train, y_train)
    test_metrics = evaluate(best_rf, X_test, y_test)

    # Feature importances
    perm = permutation_importance(
        best_rf, X_test, y_test,
        n_repeats=30, random_state=RANDOM_STATE, n_jobs=-1,
    )
    imp_df = pd.DataFrame({
        "feature": FEATURES,
        "perm_importance_mean": perm.importances_mean,
        "perm_importance_std": perm.importances_std,
        "impurity_importance": best_rf.feature_importances_,
    }).sort_values("perm_importance_mean", ascending=False)

    # SHAP
    save_shap(best_rf, X_test, FEATURES, tag=sheet)

    # Persist
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / f"baselines_{sheet}.csv", index=False)
    imp_df.to_csv(OUTPUT_DIR / f"importance_{sheet}.csv", index=False)
    with open(OUTPUT_DIR / f"best_params_{sheet}.json", "w") as f:
        json.dump(
            {"params": best_params, "cv_r2": study.best_value, "test": test_metrics},
            f, indent=2,
        )

    print("Baselines (5-fold CV R² on train):")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Best Optuna params: {best_params}")
    print(
        f"Held-out test: R² = {test_metrics['r2']:.3f}, "
        f"RMSE = {test_metrics['rmse']:.3f}"
    )
    print("Importances:")
    print(imp_df.to_string(index=False))

    return {
        "league": sheet,
        "n_rows": n,
        "rf_default_cv_r2": next(r["cv_r2_mean"] for r in rows if r["model"] == "rf_default"),
        "rf_optuna_cv_r2": study.best_value,
        "test_r2": test_metrics["r2"],
        "test_rmse": test_metrics["rmse"],
    }


# ---------------------------------------------------------------------------
# Pooled run
# ---------------------------------------------------------------------------
def run_pooled(file_path: str) -> dict:
    print("\n===== POOLED (all leagues, one-hot league feature) =====")
    df, lg_cols = load_pooled(file_path)
    feats = FEATURES + lg_cols
    X = df[feats].values
    y = df[TARGET].values
    print(f"Rows: {len(df)}, Features: {len(feats)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # Baselines
    baselines = {
        "dummy_mean": DummyRegressor(strategy="mean"),
        "ridge": Ridge(random_state=RANDOM_STATE),
        "rf_default": RandomForestRegressor(
            n_estimators=500, random_state=RANDOM_STATE, n_jobs=1
        ),
    }
    rows = []
    for name, mdl in baselines.items():
        m, s = cv_r2(mdl, X_train, y_train)
        rows.append({"model": name, "cv_r2_mean": m, "cv_r2_std": s})

    study = tune_rf(X_train, y_train)
    best_params = study.best_params
    rows.append({"model": "rf_optuna", "cv_r2_mean": study.best_value, "cv_r2_std": None})

    best_rf = RandomForestRegressor(
        **best_params, random_state=RANDOM_STATE, n_jobs=-1
    )
    best_rf.fit(X_train, y_train)
    test_metrics = evaluate(best_rf, X_test, y_test)

    perm = permutation_importance(
        best_rf, X_test, y_test,
        n_repeats=30, random_state=RANDOM_STATE, n_jobs=-1,
    )
    imp_df = pd.DataFrame({
        "feature": feats,
        "perm_importance_mean": perm.importances_mean,
        "perm_importance_std": perm.importances_std,
        "impurity_importance": best_rf.feature_importances_,
    }).sort_values("perm_importance_mean", ascending=False)

    save_shap(best_rf, X_test, feats, tag="POOLED")

    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "baselines_POOLED.csv", index=False)
    imp_df.to_csv(OUTPUT_DIR / "importance_POOLED.csv", index=False)
    with open(OUTPUT_DIR / "best_params_POOLED.json", "w") as f:
        json.dump(
            {"params": best_params, "cv_r2": study.best_value, "test": test_metrics},
            f, indent=2,
        )

    print("Baselines (5-fold CV R² on train):")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Best Optuna params: {best_params}")
    print(
        f"Held-out test: R² = {test_metrics['r2']:.3f}, "
        f"RMSE = {test_metrics['rmse']:.3f}"
    )
    print("Top importances:")
    print(imp_df.head(10).to_string(index=False))

    return {
        "league": "POOLED",
        "n_rows": len(df),
        "rf_default_cv_r2": next(r["cv_r2_mean"] for r in rows if r["model"] == "rf_default"),
        "rf_optuna_cv_r2": study.best_value,
        "test_r2": test_metrics["r2"],
        "test_rmse": test_metrics["rmse"],
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    xls = pd.ExcelFile(DATA_PATH, engine="odf")
    print(f"Sheets: {xls.sheet_names}")

    summary: list[dict] = []
    for sheet in xls.sheet_names:
        res = run_league(sheet, DATA_PATH)
        if res:
            summary.append(res)

    summary.append(run_pooled(DATA_PATH))

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(OUTPUT_DIR / "summary.csv", index=False)

    print("\n===== SUMMARY =====")
    print(summary_df.round(3).to_string(index=False))
    print(f"\nAll outputs saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
