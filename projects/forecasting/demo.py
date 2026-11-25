"""Reproducible rolling-origin forecasting with synthetic daily production data."""
from datetime import date, timedelta
from statistics import mean
import json
import random


def sample_data(days=180):
    rng = random.Random(42)
    start = date(2024, 1, 1)
    weekly = [8, 5, 3, 2, 0, -10, -8]
    return [{'date': (start + timedelta(days=i)).isoformat(),
             'value': round(120 + 0.45*i + weekly[i % 7] + rng.uniform(-4, 4), 3)}
            for i in range(days)]


def fit(history):
    if len(history) < 14:
        raise ValueError('At least two complete weeks are required')
    ys = [r['value'] for r in history]
    xs = list(range(len(ys)))
    mx, my = mean(xs), mean(ys)
    slope = sum((x-mx)*(y-my) for x,y in zip(xs,ys))/sum((x-mx)**2 for x in xs)
    intercept = my - slope*mx
    seasonal = {}
    for weekday in range(7):
        seasonal[weekday] = mean(y-intercept-slope*i for i,y in enumerate(ys)
                                 if date.fromisoformat(history[i]['date']).weekday() == weekday)
    residuals = sorted(y-intercept-slope*i-seasonal[date.fromisoformat(history[i]['date']).weekday()]
                       for i,y in enumerate(ys))
    return {'intercept': intercept, 'slope': slope, 'weekday': seasonal,
            'residual_low': residuals[int(.1*(len(residuals)-1))],
            'residual_high': residuals[int(.9*(len(residuals)-1))], 'n': len(ys)}


def predict(model, future_dates):
    rows = []
    for offset, day in enumerate(future_dates):
        center = model['intercept'] + model['slope']*(model['n']+offset) + model['weekday'][date.fromisoformat(day).weekday()]
        rows.append({'date': day, 'forecast': center,
                     'low': center+model['residual_low'], 'high': center+model['residual_high']})
    return rows


def metrics(actual, predicted):
    if len(actual) != len(predicted):
        raise ValueError('Each observation requires exactly one prediction')
    errors = [abs(a-p) for a,p in zip(actual,predicted)]
    if not errors: raise ValueError('Evaluation must contain observations')
    return {'mae': round(mean(errors), 4),
            'wape': round(sum(errors)/sum(abs(a) for a in actual), 4) if any(actual) else None}


def backtest(rows, initial=84, horizon=7):
    if initial < 14 or horizon < 1 or len(rows) < initial+horizon:
        raise ValueError('Require two training weeks and at least one complete forecast window')
    dates = [date.fromisoformat(r['date']) for r in rows]
    if any(b-a != timedelta(days=1) for a,b in zip(dates, dates[1:])):
        raise ValueError('Daily observations must be complete, unique, and ordered')
    actual, seasonal, trend = [], [], []
    folds, predictions = [], []
    for origin in range(initial, len(rows)-horizon+1, horizon):
        history, future = rows[:origin], rows[origin:origin+horizon]
        model = fit(history)  # Every fold is fitted again, using past rows only.
        expected = predict(model, [r['date'] for r in future])
        actual.extend(r['value'] for r in future)
        baseline = [history[-7+i%7]['value'] for i in range(horizon)]
        seasonal.extend(baseline)
        trend.extend(r['forecast'] for r in expected)
        for lead,(observed,forecast,naive) in enumerate(zip(future,expected,baseline),1):
            predictions.append({'origin':history[-1]['date'], 'date':observed['date'],
                                'lead':lead, 'actual':observed['value'],
                                'seasonal_naive':round(naive,6),
                                'trend_plus_weekday':round(forecast['forecast'],6)})
        folds.append({'trained_through': history[-1]['date'], 'predict_from': future[0]['date'],
                      'predict_through': future[-1]['date']})
    return {'folds': folds, 'forecasted_days': len(actual),
            'seasonal_naive': metrics(actual, seasonal), 'trend_plus_weekday': metrics(actual, trend),
            'predictions': predictions}


def run():
    rows = sample_data()
    future = [(date.fromisoformat(rows[-1]['date'])+timedelta(days=i)).isoformat() for i in range(1,8)]
    return {'dataset': '180 synthetic daily production observations; arbitrary units; seed 42',
            'evaluation': backtest(rows),
            'next_seven_days': [{k:round(v,3) if isinstance(v,float) else v for k,v in row.items()}
                                for row in predict(fit(rows), future)]}


if __name__ == '__main__': print(json.dumps(run(), indent=2))
