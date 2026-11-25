"""Compare tabular baselines on unseen passenger groups, with fold-safe transforms."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from portfolio_support import ROOT, classification_metrics, provenance, save_or_print
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def prepare():
    path = ROOT/'data/legacy/train.csv'
    data = pd.read_csv(path)
    groups = data['PassengerId'].str.split('_').str[0]
    x = data.drop(columns=['Transported','PassengerId','Name']).copy()
    cabin = x.pop('Cabin').str.split('/', expand=True)
    x['CabinDeck'], x['CabinSide'] = cabin[0], cabin[2]
    for col in ['CryoSleep','VIP']:
        x[col] = x[col].map({True:'yes',False:'no'}).fillna('unknown')
    y = data['Transported'].astype(int)
    train,test = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42).split(x,y,groups))
    return path,x,y,groups,train,test


def run():
    path,x,y,groups,train,test = prepare()
    numeric = list(x.select_dtypes(include='number').columns)
    categorical = [c for c in x if c not in numeric]
    def preprocessing():
        return ColumnTransformer([
            ('numeric',make_pipeline(SimpleImputer(strategy='median'), StandardScaler()),numeric),
            ('category',make_pipeline(SimpleImputer(strategy='most_frequent'),
                OneHotEncoder(handle_unknown='ignore')),categorical)])
    models = {
        'logistic_regression':make_pipeline(preprocessing(), LogisticRegression(max_iter=1500, random_state=42)),
        'random_forest':make_pipeline(preprocessing(),RandomForestClassifier(n_estimators=150,
             max_depth=10,min_samples_leaf=3,n_jobs=1,random_state=42))}
    for model in models.values():model.fit(x.iloc[train],y.iloc[train])
    return {'dataset':provenance(path), 'train_rows':len(train), 'test_rows':len(test),
            'group_overlap':len(set(groups.iloc[train])&set(groups.iloc[test])),
            'input_features':list(x.columns), 'split':'20% of passenger groups held out, seed 42',
            'test_positive_rate':round(float(y.iloc[test].mean()),6),
            'selection':'Fixed baselines; no tuning against the holdout',
            'holdout':{name:classification_metrics(y.iloc[test],m.predict(x.iloc[test]),
                   m.predict_proba(x.iloc[test])[:,1]) for name,m in models.items()}}


if __name__ == '__main__':save_or_print(run(), Path(__file__).resolve().parent)
