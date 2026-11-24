# Spaceship Titanic

## Question and approach

Predict a binary target from mixed numeric and categorical features. Feature preparation and supervised prediction in a Kaggle setting.

## Inspect the original work

**[Open the notebook](legacy/Spaceship%20Titanic%20-%20Kaggle%20Competition.ipynb)**. It contains 93 code cells, with the original analysis and saved outputs. Those outputs are historical experiment evidence, not a newly reproduced benchmark.

Main imports found in the notebook: `graphviz, io, keras.layers, keras.models, keras.wrappers.scikit_learn, matplotlib.pyplot, numpy, pandas, plotly.graph_objects, plotly.subplots, pydot, seaborn, shap, sklearn, sklearn.ensemble, sklearn.linear_model, sklearn.metrics, sklearn.model_selection, sklearn.neighbors, sklearn.preprocessing, sklearn.svm, sklearn.tree, tensorflow, tensorflow.keras, warnings, xgboost`.

### Code excerpt from the notebook

```python
# Embedding the name based on length and number of vowels
def name_embedding(name):
    name = name.split(' ')
    name = name[0] + name[1]
    length = len(name)
    vowels = sum(name.count(w) for w in 'aeiou')
    return length, vowels

# Create the new variables
train['NameLength'], train['NameVowels'] = zip(*train['Name'].apply(name_embedding))
test['NameLength'], test['NameVowels'] = zip(*test['Name'].apply(name_embedding))

# Dropping the Name column
train.drop('Name', axis=1, inplace=True)
test.drop('Name', axis=1, inplace=True)
```

## Data and reproducibility

The notebook is preserved as submitted in the original portfolio. It may refer to local files or external datasets not present here. Dependencies are not pinned and the saved results have not been rerun. Review data-loading cells before attempting execution. New reproducible examples are listed in the repository root README; this historical notebook is not presented as one of them.

[Additional original notebook](legacy/Spaceship%20Titanic%20-%20Kaggle.ipynb)
