# Bayesian Specification Comparison Report

No winning specification can be identified yet from saved outputs, because the comparison artifacts are not present in `outputs/model_comparison/`.

At the time of inspection, the expected summary files were missing:

- `outputs/model_comparison/model_comparison_summary.csv`
- `outputs/model_comparison/model_comparison_report.txt`

Because those saved evaluation results are not available, there is no evidence on disk yet for which of the four specifications performed best on the `2024/25` holdout season.

To complete this report, run:

```bash
python src/optimize_bayes_models.py
```

Then this report should be updated from the saved comparison metrics, with the best model chosen using the shared temporal holdout results and sampler diagnostics.
