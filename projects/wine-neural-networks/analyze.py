"""Grouped holdout and training-only model selection on UCI red wine data."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from portfolio_support import ROOT, classification_metrics, provenance, save_or_print
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit, StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def prepare():
    path = ROOT/'data/public/winequality-red.csv'
    data = pd.read_csv(path, sep=';')
    x, y = data.drop(columns='quality'), (data['quality'] >= 6).astype(int)
    # Identical chemistry remains in one partition, even if sensory scores differ.
    groups = pd.util.hash_pandas_object(x, index=False).astype(str)
    train, test = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42).split(x, y, groups))
    return path, x, y, groups, train, test


def run():
    path,x,y,groups,train,test = prepare()
    baseline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1500, random_state=42))
    network = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(12,8),
                             max_iter=2000, tol=.001, n_iter_no_change=20,
                             random_state=42, solver='adam'))
    folds = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    search = GridSearchCV(network, {'mlpclassifier__alpha':[.1,1.,10.]},
                          scoring='balanced_accuracy', cv=folds, n_jobs=1, refit=True)
    search.fit(x.iloc[train], y.iloc[train], groups=groups.iloc[train])
    baseline.fit(x.iloc[train],y.iloc[train])
    models = {'logistic_regression':baseline, 'neural_network':search.best_estimator_}
    return {'dataset':provenance(path), 'target':'sensory quality >= 6',
            'input_features':list(x.columns), 'train_rows':len(train), 'test_rows':len(test),
            'group_overlap':len(set(groups.iloc[train])&set(groups.iloc[test])),
            'test_positive_rate':round(float(y.iloc[test].mean()),6),
            'selected_alpha':search.best_params_['mlpclassifier__alpha'],
            'training_cv_balanced_accuracy':round(float(search.best_score_),6),
            'fitted_network_iterations':int(search.best_estimator_[-1].n_iter_),
            'selection':'Three training-only grouped folds; fixed (12,8) hidden layers; threshold 0.5',
            'holdout':{name:classification_metrics(y.iloc[test],m.predict(x.iloc[test]),
                        m.predict_proba(x.iloc[test])[:,1]) for name,m in models.items()}}


if __name__ == '__main__': save_or_print(run(), Path(__file__).resolve().parent)
