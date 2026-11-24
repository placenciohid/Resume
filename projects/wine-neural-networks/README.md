# Neural networks for wine quality

## Question and approach

Model quality categories from measured wine characteristics. Neural-network training and evaluation in the original notebook.

## Inspect the original work

**[Open the notebook](legacy/Neural%20Networks%20Application%20-%20Wine%20Quality.ipynb)**. It contains 38 code cells, with the original analysis and saved outputs. Those outputs are historical experiment evidence, not a newly reproduced benchmark.

Main imports found in the notebook: `graphviz, io, keras.layers, keras.models, keras.utils.vis_utils, keras.wrappers.scikit_learn, matplotlib.pyplot, numpy, pandas, plotly.graph_objects, plotly.subplots, pydot, seaborn, shap, sklearn.metrics, sklearn.model_selection, sklearn.preprocessing`.

### Code excerpt from the notebook

```python
# Cross validate the binary classification model by tunnig the number of epochs and learning rate
def create_model():
    model = Sequential()
    model.add(Dense(12, input_dim=11, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = KerasClassifier(build_fn=create_model, verbose=0)
epochs = [10, 50, 100, 500]
batches = [5, 10, 20, 40, 60, 80, 100]
param_grid = dict(epochs=epochs, batch_size=batches)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(X_train, y_train)
```

## Data and reproducibility

The notebook is preserved as submitted in the original portfolio. It may refer to local files or external datasets not present here. Dependencies are not pinned and the saved results have not been rerun. Review data-loading cells before attempting execution. New reproducible examples are listed in the repository root README; this historical notebook is not presented as one of them.
