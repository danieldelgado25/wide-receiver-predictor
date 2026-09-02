# Fantasy WR Web App

A Flask + React web layer on top of
[`wide-receiver-predictor`](https://github.com/danieldelgado25/wide-receiver-predictor):
serves next-week PPR fantasy point projections for NFL wide receivers from a
trained Ridge regression model, through a REST API and a browsable dashboard.

This repo does **not** reimplement any data loading, feature engineering, or
target construction — it imports `wide-receiver-predictor`'s pipeline
directly (`build_training_dataset`) and adds a training/serving/UI layer
around it.

## Repo layout

```
fantasy-wr-webapp/
├── backend/
│   ├── training/
│   │   └── train_model.py       # offline: builds dataset, trains, saves artifact
│   ├── wr_webapp/
│   │   ├── config.py            # resolves path to wide-receiver-predictor checkout
│   │   ├── app.py                # Flask app factory
│   │   ├── services/
│   │   │   ├── wr_pipeline_service.py   # wraps build_training_dataset, caches results
│   │   │   ├── model_service.py         # loads/serves the trained model
│   │   │   └── projection_service.py    # "as of" row selection, filtering
│   │   └── api/routes.py         # /api/projections, /api/teams, /api/model-info, /api/health
│   ├── artifacts/                 # trained model lives here (wr_model_v1.joblib)
│   ├── tests/
│   └── run.py                     # dev entrypoint
└── frontend/
    ├── src/
    │   ├── api/projectionsApi.js  # the only file that knows the API's shape
    │   ├── components/            # Header, FilterBar, ProjectionsTable, ModelInfoBanner
    │   └── App.jsx                # owns filter state, orchestrates fetching
    └── package.json
```

## Setup

**Prerequisite:** clone `wide-receiver-predictor` as a sibling directory of
this repo (or set `WR_PREDICTOR_PATH` to point at your checkout elsewhere —
see `backend/wr_webapp/config.py`):

```
some-folder/
├── wide-receiver-predictor/
└── fantasy-wr-webapp/
```

### Backend

```bash
cd backend
pip install -r requirements.txt --break-system-packages   # or use a venv

# Train the model (pulls real NFL data via nflreadpy, ~1-2 min).
# Only needs to be re-run when you want to retrain — see "Retraining" below.
python training/train_model.py

# Run the API
python run.py    # http://localhost:5000
```

### Frontend

```bash
cd frontend
npm install
npm run dev       # http://localhost:5173
```

The frontend talks to `http://localhost:5000` by default; override with a
`VITE_API_BASE_URL` env var for other environments.

## Architecture notes

**Service layer.** Nothing outside `wr_pipeline_service.py` calls
`build_training_dataset` directly, and nothing outside `model_service.py`
calls `joblib.load` or `.predict()` on the raw model. `routes.py` stays thin:
parse the request, call a service, format JSON. This isolation is what makes
it possible to, e.g., swap the model artifact format or the upstream data
pipeline later while touching exactly one file.

**Model loading.** The trained model is a serialized artifact
(`artifacts/wr_model_v1.joblib`) — a dict containing the fitted sklearn
`Pipeline` plus the exact ordered list of feature columns it expects,
training metadata, and evaluation metrics. `model_service.py` loads it once
per process (thread-safe singleton) and explicitly reorders/selects columns
before every prediction, since sklearn Pipelines predict by column
*position*, not name — a silent column-order mismatch would produce wrong
predictions with no error.

**Frontend API consumption.** `src/api/projectionsApi.js` is the frontend's
mirror of the backend's service layer: the only file that knows the base URL
or endpoint shapes. `App.jsx` owns all filter state and the fetch
`useEffect`s; every other component is presentational (props in, JSX out),
which keeps data-fetching logic in one predictable place.

## What was fixed vs. the original pipeline

- **No model was ever serialized.** `02_ML_exploration.ipynb` trains in
  memory only. `training/train_model.py` adds the missing serialization step.
- **`merge_ff_opportunity=True` is broken** — a `polars.exceptions.SchemaError`
  from an `int`/`str` dtype mismatch on the season join. Left disabled
  (`merge_ff_opportunity=False`) rather than patched here, since fixing the
  upstream pipeline belongs in that repo, not this one.
- **The notebook's blanket `dropna()` silently discarded ~40% of rows** —
  `temp`/`wind` are null for every dome game, so dropping any row with a null
  feature disproportionately removed dome-team players from training
  entirely. `train_model.py` instead requires only the rolling-3 features and
  the target to be non-null, and median-imputes remaining nulls (mostly
  `temp`/`wind`) inside the model pipeline. This grew the training set from
  ~9,000 to ~12,700 rows.

## Model performance (2016-2024 data, time-based split)

| | Test MAE | Test RMSE | Test R² |
|---|---|---|---|
| Rolling-3-week average (baseline) | 5.17 | 7.10 | 0.173 |
| **Ridge regression (served model)** | **4.81** | **6.46** | **0.314** |

The model beats the naive "predict their recent average" baseline by ~9% on
RMSE — see `/api/model-info` for these numbers live, and the in-app banner
that surfaces this comparison rather than hiding it.

Full metrics (train/val/test, chosen alpha, training seasons) are saved
alongside the model at `backend/artifacts/wr_model_v1.metrics.json`.

## Retraining

Re-run `python training/train_model.py` from `backend/` whenever:
- A new NFL season/week of data becomes available and you want the model to
  see it (note: the *test set* is currently seasons 2023-2024 — retraining on
  more recent data means updating `TRAIN_SEASONS`/`VAL_SEASONS`/`TEST_SEASONS`
  in `train_model.py` to keep a genuine holdout).
- Upstream `wide-receiver-predictor` feature engineering changes.
- You want to try a different alpha grid, model family, or fix the
  `merge_ff_opportunity` join upstream and pull in those features.

The script overwrites `artifacts/wr_model_v1.joblib` in place; there's no
versioning beyond the filename right now — worth adding (e.g. a
`wr_model_v2.joblib` + a `CURRENT_VERSION` pointer) before this goes in front
of real users, so a bad retrain can be rolled back.
