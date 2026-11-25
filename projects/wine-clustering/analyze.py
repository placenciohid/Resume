"""Explore UCI red wine chemistry, keeping sensory quality out of clustering."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from portfolio_support import ROOT, provenance, save_or_print
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)
from sklearn.preprocessing import StandardScaler


def run():
    path = ROOT/'data/public/winequality-red.csv'
    data = pd.read_csv(path, sep=';')
    features = data.drop(columns='quality')
    scaled = StandardScaler().fit_transform(features)
    candidates = []
    for k in range(2, 11):
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(scaled)
        candidates.append({'k': k, 'silhouette': round(silhouette_score(scaled, labels), 6),
                           'davies_bouldin': round(davies_bouldin_score(scaled, labels), 6),
                           'calinski_harabasz': round(calinski_harabasz_score(scaled, labels), 6)})
    selected = max(candidates, key=lambda r:r['silhouette'])['k']
    labels = KMeans(n_clusters=selected, n_init=10, random_state=42).fit_predict(scaled)
    stability = [adjusted_rand_score(labels, KMeans(n_clusters=selected, n_init=10,
                 random_state=seed).fit_predict(scaled)) for seed in (7, 21, 84)]
    pca = PCA().fit(scaled)
    profiles = data.assign(cluster=labels).groupby('cluster').agg(
        rows=('quality','size'), alcohol=('alcohol','mean'),
        volatile_acidity=('volatile acidity','mean'),
        quality_for_posthoc_description=('quality','mean')).round(4).reset_index()
    return {'dataset': provenance(path), 'rows': len(data),
            'duplicate_rows': int(data.duplicated().sum()), 'features': list(features.columns),
            'quality_used_in_fit': False, 'selection': 'Largest in-sample silhouette across k=2..10',
            'candidates': candidates, 'selected_k': selected,
            'seed_stability_ari': [round(float(v),6) for v in stability],
            'pca_components_for_95pct_variance': int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), .95)+1),
            'first_two_pc_variance': round(float(sum(pca.explained_variance_ratio_[:2])),6),
            'cluster_profiles': profiles.to_dict(orient='records')}


if __name__ == '__main__': save_or_print(run(), Path(__file__).resolve().parent)
