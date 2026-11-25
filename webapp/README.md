# Web app — Medical Insurance Cost Predictor

A small Flask app that serves the case study's winning model so an applicant's profile can be entered in a
browser and the predicted annual charge returned, along with a per-feature contribution breakdown (exact,
because the deployed model is linear in its polynomial terms) and an uncertainty band of ± 1 test RMSE.

## Files

| File | Purpose |
|---|---|
| `app.py` | Flask application — routes `/`, `/api/predict`, `/api/model-info`, `/health` |
| `export_model.py` | re-trains the winning model on the same split and writes `model/model_bundle.joblib` |
| `model/model_bundle.joblib` | the deployed artefact (pipeline + metrics + ranges), ~5 KB |
| `templates/index.html` | single-page UI (no external CSS/JS, no CDN — works offline) |

## Run it

```bash
pip install -r requirements.txt          # flask is now included
python webapp/export_model.py            # optional — the bundle is committed
python webapp/app.py                     # http://localhost:5000
```

Use a different port with `PORT=8000 python webapp/app.py`.

## API

```bash
curl -X POST http://localhost:5000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"age":45,"sex":"male","bmi":34.8,"children":2,"smoker":"yes","region":"northeast"}'
```

```json
{
  "ok": true,
  "prediction": 43090.16,
  "baseline": 13131.55,
  "low": 38871.28, "high": 47309.04,
  "contributions": [{"feature": "Smoker", "value": 15220.79}, ...],
  "band": "above the dataset average"
}
```

`GET /api/model-info` returns the metrics table, dataset summary and valid input ranges; `GET /health`
is a one-line liveness probe.

Validation is strict on purpose: age must be 18–64, BMI 15.96–53.13 and children 0–5 — the ranges present
in the training data. Inputs outside them return HTTP 400 rather than a silently extrapolated number.

## Notes

* The model reproduces the report exactly (R² = 0.9031, RMSE = 4,218.88) because it is trained on the same
  855 rows with `random_state = 42`.
* Contributions add up to `prediction − baseline` to within rounding, so the chart is a decomposition,
  not a sampled approximation such as SHAP.
* For deployment elsewhere the app is a standard WSGI object (`app`), so Gunicorn works:
  `gunicorn -w 2 -b 0.0.0.0:8000 webapp.app:app`.
