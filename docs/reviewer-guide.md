# A guide to reviewing the portfolio

## Start with the decision, then inspect the evidence

| If you want to assess… | Start here | Then inspect |
| --- | --- | --- |
| Applying data science in mining | [Forecasting case](../projects/forecasting/) | `demo.py`, per-origin predictions in `results.json`, and the linked planning discussion |
| Connecting process improvement and financial controls | [Collections case](../projects/commercial-reporting/) | Whole-document matching, decimal arithmetic, and duplicate/currency tests |
| Building a usable AI information product | [Knowledge search](../projects/knowledge-search/) | Source contract, retrieval stages, rank fusion, and no-evidence behavior |
| Evaluation judgment | [Wine model comparison](../projects/wine-neural-networks/) | Grouped folds, training-only pipelines, probability-based metrics, and the decision to favor the simpler baseline |
| Handling operational data | [Event prediction](../projects/event-prediction/) | Observation dates, outcome windows, current-state versus future-target logic |
| Mathematical modeling | [Optimization](../projects/optimization/) | Objective, feasibility, exact reference, and comparison against the same constraints |
| Research interpretation | [Grouped experiments](../projects/clustering-noncompliance/) | Why uncertainty changes and why the incremental comparison needs its own numerator |
| Systems integration | [Dynamic Zoom](../projects/dynamic-zoom/) | Freshness versus completeness, bounded buffering, frame accounting, and upstream collaboration |

## The useful file pattern

Each project starts with an article in `README.md`. Maintained portable examples use `demo.py` and `results.json`; maintained real-data analyses use `analyze.py` and `analysis-results.json`. `figures/` contains charts generated from those results. `legacy/` retains original notebooks and research scripts.

The public examples favor small, inspectable implementations. Where a more sophisticated production approach is discussed, the article identifies the boundary between the professional work and the public reference. That lets a reviewer assess both the implementation and the judgment behind its claims.
