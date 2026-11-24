"""Three basic classifiers, without third-party ML libraries; synthetic data only."""
import json
import math
import random


def sigmoid(value):
    return 1 / (1 + math.exp(-max(-40, min(40, value))))


def fit_logistic(xs, ys, steps=1200, rate=0.2):
    weights = [0.0] * (len(xs[0]) + 1)
    for _ in range(steps):
        gradient = [0.0] * len(weights)
        for x, y in zip(xs, ys):
            features = [1.0] + list(x)
            error = sigmoid(sum(a*b for a,b in zip(weights, features))) - y
            gradient = [g + error*f for g,f in zip(gradient, features)]
        weights = [w - rate*g/len(xs) for w,g in zip(weights, gradient)]
    return weights


def logistic_predict(weights, xs):
    return [int(sigmoid(sum(a*b for a,b in zip(weights, [1.0]+list(x)))) >= 0.5) for x in xs]


def knn_predict(train_x, train_y, test_x, k=3):
    if not 1 <= k <= len(train_x):
        raise ValueError('k must be between 1 and the training size')
    result = []
    for x in test_x:
        nearest = sorted(range(len(train_x)), key=lambda i: sum((a-b)**2 for a,b in zip(x, train_x[i])))[:k]
        result.append(int(sum(train_y[i] for i in nearest) > k/2))
    return result


def fit_stump(xs, ys):
    # A depth-one decision tree: search feature thresholds and both directions.
    candidates = []
    for feature in range(len(xs[0])):
        values = sorted(set(x[feature] for x in xs))
        thresholds = [values[0]-1] + [(a+b)/2 for a,b in zip(values, values[1:])] + [values[-1]+1]
        for threshold in thresholds:
            for direction in (0, 1):
                errors = sum((int(x[feature] >= threshold) ^ direction) != y for x,y in zip(xs, ys))
                candidates.append((errors, feature, threshold, direction))
    return min(candidates)[1:]


def stump_predict(model, xs):
    feature, threshold, direction = model
    return [int(x[feature] >= threshold) ^ direction for x in xs]


def run():
    rng = random.Random(42)
    xs = [[rng.uniform(-2,2), rng.uniform(-2,2)] for _ in range(240)]
    ys = [int(x[0] + 0.4*x[1] > 0) for x in xs]
    train_x, test_x, train_y, test_y = xs[:180], xs[180:], ys[:180], ys[180:]
    predictions = {'knn': knn_predict(train_x, train_y, test_x),
                   'logistic_regression': logistic_predict(fit_logistic(train_x, train_y), test_x),
                   'decision_stump': stump_predict(fit_stump(train_x, train_y), test_x)}
    return {'data': 'synthetic; 180 training / 60 held-out records',
            'test_accuracy': {name: sum(a==b for a,b in zip(pred,test_y))/len(test_y) for name,pred in predictions.items()}}


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
