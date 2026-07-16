# German Electricity Demand - Weekend Load Forecasting

Forecasting German electricity demand (Open Power System Data, country code
`DE`) from January 2015 to October 2020, then projecting the final two years with
five model families and comparing them against a seasonal-naive benchmark.

## Data lens

Everything is built on the **weekend-only weekly mean**: for each week the mean
hourly load over Saturday and Sunday hours only. Weekends form a distinct
reserve-planning regime - heavy industrial load is largely absent, so demand is
driven by residential and leisure use and sits several gigawatts below the plain
full-week mean. A public holiday landing on a weekday makes that weekday behave
like a weekend, so the weekend series is comparatively insensitive to holiday
timing, which makes it a clean, low-variance target for the seasonal models.

## What the notebook does

- **Leg 1** Load the OPSD 60-minute file, slice from 2015 in UTC, convert to
  Berlin time, bin to the weekend weekly and daily series, run EDA, an additive
  `seasonal_decompose(period=52)`, and the full ADF / ACF / PACF differencing
  battery.
- **Leg 2** Four benchmarks (mean, naive, seasonal naive, drift) over the
  104-week horizon.
- **Leg 3** SARIMA with the full AIC grid p in [0,6], d in [0,2], q in [0,6]
  (147 orders) run in parallel with `joblib`, residual diagnostics (ACF,
  histogram, Ljung-Box), a 95% forecast interval and RMSE / MASE.
- **Leg 4** SARIMAX reusing the chosen order plus Berlin temperature, HDD and
  CDD as exogenous regressors (a conditional forecast).
- **Leg 5** Feature-based ML with GradientBoosting (primary) and AdaBoost
  (secondary), a recursive 104-week rollout and a feature-importance plot.
- **Leg 6** An hourly LSTM, CPU-forced, with a sixteen-candidate sweep over
  units, dropout, learning rate and layer depth, a compiled recursive rollout,
  aggregated back to the weekend lens.
- **Leg 7** The six analysis questions, grounded in this run's numbers.
- **Leg 8** A consolidated MAE / RMSE / MASE / Bias table with MASE ratios, a
  master comparison figure and regime diagnostics.
- **Leg 9** Repository notes.

## How to run

1. Install the dependencies: `pip install -r requirements.txt`
2. Open `Karthik_24079547.ipynb` and Run All from a fresh kernel.

The OPSD load file (~130 MB) is downloaded automatically if it is not already
beside the notebook. Berlin daily temperature is read from
`berlin_daily_temperature.csv` if present, otherwise fetched from the Open-Meteo
archive API and cached; if the API is unreachable a clearly labelled synthetic
curve substitutes so the notebook still runs. All paths are relative and all
random seeds are fixed.

`FAST_RUN` is `False` by default so the full SARIMA grid runs as the brief
requires; set it to `True` only for quick iteration (it narrows p and q to
[0,3]). The full grid's wall time scales with the core count, from a couple of
minutes on a many-core machine to a couple of hours on two cores, and it prints
progress with an ETA.

## Outputs

Created at run time under `outputs/`:

- `figures/model_comparison_master.png` - all models against the actual
- `forecasts/weekly_forecasts.csv` - each model's 104-week path
- `metrics/model_scores.csv` - the consolidated score table
