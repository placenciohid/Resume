"""Leakage-aware classical baseline for app-review ratings; aggregate output only."""
from pathlib import Path
import re
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from portfolio_support import ROOT, classification_metrics, provenance, save_or_print
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline


def prepare():
    path = ROOT/'data/legacy/reviews.csv'
    data = pd.read_csv(path, usecols=['content','score','at'])
    raw_rows = len(data)
    data = data.dropna(subset=['content','score']).copy()
    data = data[data['score'].isin([1,2,4,5])].copy()
    data['text'] = data['content'].map(lambda s: re.sub(r'\s+',' ',str(s).lower()).strip())
    data = data[data['text'].str.len()>0].reset_index(drop=True)
    groups = data['text']
    y = (data['score']>=4).astype(int)
    train,test = next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42).split(data['text'],y,groups))
    return path,data,y,groups,train,test,raw_rows


def run():
    path,data,y,groups,train,test,raw_rows = prepare()
    model = make_pipeline(TfidfVectorizer(ngram_range=(1,2),min_df=3,max_features=20000,
                                         sublinear_tf=True),
                          LogisticRegression(max_iter=1500,random_state=42))
    model.fit(data['text'].iloc[train],y.iloc[train])
    return {'dataset':provenance(path), 'raw_rows':raw_rows, 'retained_rows':len(data),
            'target':'4-5 stars positive; 1-2 stars negative; neutral 3-star rows excluded',
            'train_rows':len(train), 'test_rows':len(test),
            'duplicate_text_groups_across_split':len(set(groups.iloc[train])&set(groups.iloc[test])),
            'test_positive_rate':round(float(y.iloc[test].mean()),6),
            'vocabulary_size':len(model[0].vocabulary_),
            'selection':'Fixed TF-IDF + logistic baseline; grouped normalized-text holdout; threshold 0.5',
            'holdout':classification_metrics(y.iloc[test],model.predict(data['text'].iloc[test]),
                                             model.predict_proba(data['text'].iloc[test])[:,1])}


if __name__ == '__main__':save_or_print(run(),Path(__file__).resolve().parent)
