# Forecasting — rolling-origin evaluation

## Why this example exists

My professional work includes production and cash forecasting. This independent companion demonstrates evaluation mechanics using generated daily production data. It is not the internal model, the cash-forecast trial, or evidence of their business results.

## Reproduce

`python projects/forecasting/demo.py`

Python 3.10+, standard library only. The script generates 180 daily observations from a trend, weekday effects, and seeded noise. It fits on the first 84 days and evaluates thirteen successive seven-day windows, each time fitting only on observations before that window. No test-set transformations or future values enter fitting.

## Baseline, model, and output

The seasonal-naive baseline repeats the previous week. The comparison model estimates a linear trend and weekday effects. Residual 10th/90th percentiles provide an illustrative range; it is not a calibrated prediction interval.

Current synthetic output: seasonal-naive MAE **3.8145**, trend-plus-weekday MAE **2.179**, across **91 held-out days**. MAE is in arbitrary production units, and WAPE divides total absolute error by total absolute observed volume. The data generator deliberately contains the structure this model fits, so this is a correctness demonstration rather than evidence of generalization to industrial data.

See [demo.py](demo.py) for implementation and [results.json](results.json) for the complete evaluation origins, metrics, and next-week forecast. Running the script prints the same JSON without writing files. A changed seed or data generator invalidates the saved snapshot.

## What to inspect

- `fit`: train-only trend, weekday effects, and residuals.
- `backtest`: rolling origins, seven-day horizon, seasonal baseline.
- `metrics`: MAE and WAPE on held-out predictions.
- Tests: temporal ordering, independence from future values, and ordered ranges.

## Limits and extensions

Daily rows must be complete and ordered; missing days need explicit imputation or a different baseline. This is a single series with no shutdown, weather, or shipment covariates. A cash forecast would need invoice-level availability timestamps, customer payment behavior, and a separate volume/value evaluation. Those sources are not simulated as if they were the real employer data.
