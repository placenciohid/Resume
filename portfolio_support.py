"""Shared reporting contracts for the optional, real-data portfolio analyses."""
from pathlib import Path
import hashlib
import json
import platform

ROOT = Path(__file__).resolve().parent


def provenance(path):
    import numpy
    import pandas
    import sklearn
    return {'file': path.relative_to(ROOT).as_posix(),
            'sha256': hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
            'hash_basis': 'File bytes with CRLF normalized to LF',
            'python': platform.python_version(), 'numpy': numpy.__version__,
            'pandas': pandas.__version__, 'scikit_learn': sklearn.__version__}


def classification_metrics(y, predicted, probability):
    from sklearn.metrics import (accuracy_score, average_precision_score,
                                 balanced_accuracy_score, brier_score_loss,
                                 confusion_matrix, f1_score, roc_auc_score)
    return {key: round(float(value), 6) for key, value in {
        'accuracy': accuracy_score(y, predicted),
        'balanced_accuracy': balanced_accuracy_score(y, predicted),
        'f1': f1_score(y, predicted),
        'roc_auc': roc_auc_score(y, probability),
        'average_precision': average_precision_score(y, probability),
        'brier_score': brier_score_loss(y, probability)}.items()} | {
            'confusion_matrix': confusion_matrix(y, predicted, labels=[0, 1]).tolist()}


def save_or_print(result, folder):
    import sys
    text = json.dumps(result, indent=2, ensure_ascii=False) + '\n'
    if '--write' in sys.argv:
        (folder / 'analysis-results.json').write_text(text, encoding='utf-8')
    print(text)
