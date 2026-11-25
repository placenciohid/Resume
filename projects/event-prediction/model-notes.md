# Rare-event neural models: read the implementation behind the name

The research scripts explore representation learning and graph-based classification. Their input training files are not included. These notes explain the architecture that is actually present, identify the evaluation questions it raises, and connect it to the runnable [event-data contract](README.md).

## Autoencoder features and a Gumbel activation

The [GEV-NN script](legacy/GEV-NN-Model.py) combines three representations: a softmax-weighted copy of the input, an autoencoder's compressed representation, and two reconstruction summaries. A feed-forward classifier receives their concatenation.

The reconstruction summaries are mean squared reconstruction error and cosine similarity. The latter is named `cos_dist` in the script, but the formula calculates similarity, not `1 - similarity`. Preserving that distinction avoids reversing the interpretation of a diagnostic feature.

```python
euclid_dist = torch.mean((x - decoded) ** 2, dim=1, keepdim=True)
final_input = torch.cat((euclid_dist, cos_dist, selected_input, encoded), dim=1)
prediction = self.mlp(final_input)
```

The hidden layers use `exp(-exp(-x))`, a Gumbel CDF activation. The output layer uses a sigmoid. This is not a fitted generalized extreme-value distribution with a learned shape parameter. Likewise, softmax feature weights make a representation adaptive but do not establish causal feature importance or implement a full fuzzy inference system simply because of a class label.

The training objective adds one quarter of the autoencoder reconstruction loss to class-weighted binary cross-entropy. That encourages reconstruction and prediction jointly. An ablation would test whether the autoencoder, weighting branch, and asymmetric activation each improve performance over a smaller classifier. Their inclusion alone does not establish an advantage.

Before operational use, I would evaluate probabilities on forward-held-out observations, inspect calibration, and choose a decision threshold on validation data against an explicit review capacity. The rare-event precision–recall curve and the number of false alerts matter alongside ROC AUC. A reconstruction feature can detect unusual records without necessarily identifying the specific event the business cares about.

## A graph built from feature similarity

The [graph script](legacy/GNN-INCM_with_HSKDM.py) creates separate train, validation, and test graphs using nearest neighbors in standardized feature space. Nodes are records, and edges reflect similarity rather than a supplied customer, equipment, or transaction relationship.

```python
dist = torch.cdist(X_tensor, X_tensor)
_, indices = dist.topk(k + 1, largest=False)
edge_index = torch.stack([row.reshape(-1), indices[:, 1:].reshape(-1)])
```

The GNN stacks configurable convolution layers with batch normalization, ELU, dropout, and training-time edge dropout. Training uses focal loss, while three independently trained networks contribute probabilities that are averaged for evaluation. Although the historical class is named `HSKDM`, the implemented training loop uses independent focal-loss updates; it contains no teacher–student distillation loss.

That implementation raises two practical questions. First, should predictions be made for a batch of unseen records that can connect to one another, or for a single arriving record with access only to the historical graph? Separate test-graph construction assumes the former. Second, `torch.cdist` materializes all pairwise distances, so graph creation requires quadratic memory before nearest neighbors are selected. It is an inspectable prototype, not a scalable neighbor-search service.

The script applies SMOTE and undersampling within training. Because interpolation affects feature geometry, its effect on graph neighborhoods deserves an ablation against weighting alone. Graph similarity also needs a business rationale: nearby feature vectors need not share the same event-generating process.

## The evaluation boundary

The historical graph script chooses a decision threshold using test labels. That consumes the test set for selection. A revised benchmark should choose the threshold on validation data and evaluate it once on an untouched test period. Its random split also answers a different question from the temporal event-preparation reference.

I would begin with a time-aware linear or boosted-tree baseline, use an approximate-neighbor index for larger graphs, and compare precision at a fixed review workload. Those experiments would establish whether graph context earns its computational and operational cost.

The runnable [demo.py](demo.py), its saved [results](results.json), and the temporal-control tests provide a concrete starting point for that evaluation. They isolate a prerequisite that applies to either architecture: features must describe the past and labels must describe the specified future window.
