# Contributing

Thank you for your interest in contributing! Here's how to get started.

## Development setup

```bash
git clone https://github.com/PanagiotaGr/wafer-fault-detection-with-ml.git
cd wafer-fault-detection-with-ml

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements-dev.txt
```

## Running the tests

```bash
pytest tests/ -v
```

With coverage:

```bash
pytest tests/ --cov=src --cov-report=term-missing
```

## Code style

This project uses **ruff** for linting and **black** for formatting.

```bash
ruff check src/ tests/
black src/ tests/
```

Both are enforced in CI — PRs that fail lint will not be merged.

## Project structure

```
src/
  data/       loader.py, augmentation.py   ← all data I/O lives here
  models/     cnn.py, classical.py         ← model definitions
  training/   trainer.py, losses.py        ← training loop & loss functions
  eval/       metrics.py, plots.py         ← evaluation & visualisation
  utils.py                                 ← config, seeds, device helpers

tests/                                     ← unit tests (no dataset needed)
config.yaml                                ← all hyperparameters
```

**The key rule:** entry-point scripts (`wafer_pipeline.py`, etc.) must stay thin — they just wire together modules from `src/`. No duplicated logic across scripts.

## Adding a new model

1. Add a `build_my_model(cfg)` factory in `src/models/`.
2. Register it in `ALL_MODELS` (classical) or import it in the CNN pipeline.
3. Add at least one unit test in `tests/`.

## Adding a new loss function

1. Implement it in `src/training/losses.py`.
2. Register the name string in `build_loss()`.
3. Add it as a variant option in `config.yaml` under `few_shot.variants`.

## Submitting a pull request

1. Fork the repo and create a branch: `git checkout -b feature/my-change`
2. Make your changes, add tests.
3. Run `pytest` and the linters locally — everything must pass.
4. Open a PR against `main` with a clear description of what changed and why.

## Reporting issues

Please include: Python version, OS, and the full traceback. If related to the dataset, specify which preprocessing step failed.
