"""As-of event features and future labels on synthetic operational records.

Run from any directory. No external packages or employer data are used.
"""
from datetime import date, timedelta
from statistics import mean, stdev
import json


def build_rows(records, events, horizon=7):
    rows = []
    for record in records:
        if record['operation_code'] not in {'code1', 'code2'}:
            continue
        cutoff = date.fromisoformat(record['record_date'])
        history = sorted((e for e in events if e['id'] == record['id']), key=lambda e: e['date'])
        past = [e for e in history if date.fromisoformat(e['date']) <= cutoff]
        future = [e for e in history if cutoff < date.fromisoformat(e['date']) <= cutoff + timedelta(days=horizon)]
        last = past[-1] if past else None
        starts = [e for e in past if e['kind'] == 'event']
        ends = [e for e in past if e['kind'] == 'resolve']
        # Future events affect only the label; all feature columns use past history.
        rows.append({**record, 'event_count': len(starts),
                     'last_resolve_date': ends[-1]['date'] if ends else None,
                     'currently_blocked': int(last is not None and last['kind'] == 'event'),
                     'target_event_flag': int(any(e['kind'] == 'event' for e in future))})
    return rows


def split_and_scale(rows, split_date, horizon=7):
    if horizon < 1:
        raise ValueError('Outcome horizon must be positive')
    train = [dict(r) for r in rows if r['record_date'] < split_date]
    validation = [dict(r) for r in rows if r['record_date'] >= split_date]
    if len(train) < 2:
        raise ValueError('At least two training rows are needed')
    # Purge training labels whose seven-day outcome window reaches validation.
    if any(date.fromisoformat(r['record_date']) + timedelta(days=horizon) >= date.fromisoformat(split_date) for r in train):
        raise ValueError('Training outcome windows overlap validation; add a temporal gap')
    center = mean(r['numeric_feature'] for r in train)
    scale = stdev(r['numeric_feature'] for r in train) or 1.0
    for row in train + validation:
        row['scaled_numeric_feature'] = (row['numeric_feature'] - center) / scale
    return train, validation, {'mean': center, 'scale': scale}


def sample_data():
    records = [
        {'id': 'A', 'record_date': '2024-01-01', 'operation_code': 'code1', 'numeric_feature': 10},
        {'id': 'B', 'record_date': '2024-01-02', 'operation_code': 'code2', 'numeric_feature': 20},
        {'id': 'C', 'record_date': '2024-01-15', 'operation_code': 'code1', 'numeric_feature': 100},
        {'id': 'D', 'record_date': '2024-01-16', 'operation_code': 'code2', 'numeric_feature': 200},
    ]
    events = [
        {'id': 'A', 'date': '2023-12-29', 'kind': 'event'},
        {'id': 'A', 'date': '2023-12-31', 'kind': 'resolve'},
        {'id': 'A', 'date': '2024-01-04', 'kind': 'event'},
        {'id': 'C', 'date': '2024-01-18', 'kind': 'event'},
    ]
    return records, events


if __name__ == '__main__':
    records, events = sample_data()
    train, validation, fitted = split_and_scale(build_rows(records, events), '2024-01-15')
    print(json.dumps({'data': 'synthetic', 'train_rows': len(train), 'validation_rows': len(validation),
                      'fitted_on_training_only': fitted, 'training': train, 'validation': validation}, indent=2))
