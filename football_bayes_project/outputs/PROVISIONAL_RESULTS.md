# Provisional Outputs Note

Current outputs in this repository are provisional.

- They are based on the current English-only dataset in `data/dataset.ods`.
- `development` mode outputs are for workflow checks and quick model comparison, not final scientific interpretation.
- Any metrics, coefficient summaries, plots, or comparison rankings must be rerun after international league sheets are added.
- `final` mode should only be treated as a fuller rerun setting for the dataset currently available, not as a final research conclusion.

Once new league sheets are added, rerun at minimum:

- `python src/bayes_model.py --mode development`
- `python src/optimize_bayes_models.py --mode development`
- `python src/bayes_model.py --mode final`
- `python src/optimize_bayes_models.py --mode final`
