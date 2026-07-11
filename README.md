# 🚦 Accident Risk & Hotspot Prediction — Streamlit App

A full software project (no hardware/Arduino) that:

1. **Trains on historical accident data** and automatically **compares
   multiple ML algorithms** (Logistic Regression, Decision Tree, Random
   Forest, Gradient Boosting, and XGBoost if installed) — picking whichever
   scores highest on weighted F1 on held-out data.
2. Uses your **live browser GPS location** (with a manual fallback) and
   **live weather** for that exact spot (via the free Open-Meteo API, no
   key needed) to predict **real-time accident severity risk**.
3. Shows a **hotspot heatmap** of historical accidents on an interactive
   map, plus your current position.

---

## 1. Install

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Generate historical data + train/compare models

```bash
python data/generate_historical_data.py   # builds a synthetic historical dataset
python train_model.py                     # trains + compares algorithms, saves the winner
```

You'll see a leaderboard like:

```
  Logistic Regression  | acc=0.560  f1=0.563  roc_auc=0.800
  Decision Tree        | acc=0.472  f1=0.490  roc_auc=0.699
  Random Forest        | acc=0.564  f1=0.563  roc_auc=0.795
  Gradient Boosting    | acc=0.565  f1=0.556  roc_auc=0.803

Winner (highest weighted F1): Logistic Regression
```

(Your own real dataset will likely produce different — usually stronger —
results for the tree-based / boosting models; see step 4 below.)

## 3. Run the app

```bash
streamlit run app.py
```

Your browser will prompt for location permission — allow it for live GPS,
or tick "Enter location manually" to type coordinates.

---

## 4. ⚠️ Use YOUR real dataset (important for a real project)

`data/generate_historical_data.py` creates **synthetic** data purely so
the whole pipeline runs immediately. For a real accident-hotspot project,
replace `data/historical_accidents.csv` with real records — e.g.:

- Your state/city Traffic Police open accident data
- Kaggle "US Accidents (2016–2023)" dataset
- data.gov.in road accident datasets

Just make sure your CSV has these columns (see `utils.py → FEATURE_COLUMNS`):

```
latitude, longitude, hour, day_of_week, month, is_weekend, is_night,
road_type_enc, speed_limit, junction, traffic_density, weather_enc,
temperature, precipitation, wind_speed, visibility, severity
```

- `road_type_enc` uses `utils.ROAD_TYPES` mapping (Highway=0, Urban Road=1,
  Residential=2, Rural Road=3)
- `weather_enc` uses `utils.WEATHER_CONDITIONS` mapping (Clear=0, Cloudy=1,
  Rain=2, Fog=3, Storm=4, Snow=5)
- `severity` is the target: 0=Low, 1=Medium, 2=High, 3=Severe

Then just re-run `python train_model.py` — everything else (comparison,
best-model selection, the live app) works unchanged.

---

## Project structure

```
accident-risk-app/
├── app.py                          # Streamlit app (live location + weather + prediction + map)
├── train_model.py                  # Trains & compares algorithms, saves the best one
├── utils.py                        # Shared feature engineering + weather fetch (train & app use same code)
├── requirements.txt
├── data/
│   ├── generate_historical_data.py # Synthetic historical dataset generator (swap for real data)
│   └── historical_accidents.csv    # (generated)
└── model_artifacts/                # (generated) best_model.pkl, scaler.pkl, comparison table, meta
```

## How the algorithm comparison works

`train_model.py`:
1. Splits historical data 80/20 (stratified by severity).
2. Trains each candidate algorithm on the training split.
3. Scores every algorithm on the test split using accuracy, weighted F1,
   weighted precision/recall, and multiclass ROC-AUC (one-vs-rest).
4. Saves the full leaderboard to `model_artifacts/model_comparison.csv`
   (also shown live in the Streamlit sidebar).
5. Saves whichever model has the **highest weighted F1** as
   `best_model.pkl` — that's the one `app.py` loads for live prediction.

This means the "which algorithm is best" decision is always made
empirically from your actual data, not hardcoded — if you swap in a
larger/real dataset, a different algorithm may win, and the app will
automatically pick it up next time you retrain.

## Notes on live location & weather

- Live GPS uses `streamlit_js_eval.get_geolocation()`, which asks the
  browser for permission (works on `localhost` and HTTPS deployments —
  browsers block geolocation on plain HTTP).
- Live weather uses [Open-Meteo](https://open-meteo.com/) — free, no
  API key, no rate-limit signup needed.
- If either fails (permission denied / offline), the app falls back to
  manual entry fields so it never breaks.
