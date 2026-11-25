# Darío Placencio
### Operational excellence · Data science · Mining & commercial services

I work with operations and commercial teams to improve how work is planned, understood, and carried out. My experience spans process improvement and digital delivery at BHP, data science in Rio Tinto's Borates Digital team, and operational excellence in Commercial Services.

This portfolio connects that experience with code a reviewer can inspect: forecasting evaluation, payment reconciliation, evidence retrieval, optimization, and machine learning. The project articles explain the business or analytical question, the implementation choices, the results, and what those results support.

**[Portfolio website](https://placenciohid.github.io/) · [Experience & education](docs/experience.md) · [LinkedIn](https://www.linkedin.com/in/darioplacencio/) · [Contact](mailto:darioplacenciohidalgo@gmail.com)**

## Professional work

### Forecasting and planning decisions

At Rio Tinto's Borates Digital team, I developed a **365-day production forecast with 21% lower error** against the prior baseline and rebuilt an annual planning model in Python, identifying **$746K in confirmed savings**. More recently, I completed a short-horizon cash-forecasting proof of concept in Commercial Services. The article separates those decisions and opens up an independently runnable forecasting evaluation.

**[Read the case study and inspect the code →](projects/forecasting/)**

### AI search for technical knowledge

I engineered a knowledge-search platform over **300,000+ documents**, adopted by **four teams**, combining document extraction with hybrid retrieval. The public companion shows how to preserve evidence through chunking, ranking, fusion, and source-linked results.

**[Read the case study and inspect the code →](projects/knowledge-search/)**

### Collections reporting and reconciliation

I delivered Atlantic Aluminium collections reporting, an **hourly receivables-aging feed**, and reconciliation views. The article explains the calculation choices behind a useful operational view: currency separation, reporting dates, document-level matching, and avoiding double-counted payment values.

**[Read the case study and inspect the code →](projects/commercial-reporting/)**

The professional outcomes above describe separate work experience. The public companions use generated or fictional inputs so the implementation can be reviewed without employer datasets or source code.

## Technical projects with reproducible evidence

| Project | Question worth reviewing | Evidence to inspect |
| --- | --- | --- |
| [Event data and rare-event modeling](projects/event-prediction/) | What was knowable at the moment of prediction? | As-of features, future-label windows, temporal gaps, train-only scaling, original neural scripts |
| [Capacity-constrained optimization](projects/optimization/) | Can two full schedules still have different value? | Exact optimum versus first fit, capacity checks, original Julia/Pyomo notebooks |
| [Machine learning from scratch](projects/ml-from-scratch/) | Why do three classifiers learn different boundaries? | Fitting logic, decision-region plots, shared held-out comparison |
| [App-review text analytics](projects/play-store-reviews/) | Does review language predict the rating? | Reproduced TF-IDF baseline on 12,495 source reviews; duplicate-text groups kept together |
| [Wine-quality prediction](projects/wine-neural-networks/) | Does a neural network earn its complexity? | Training-only grouped selection and a fixed holdout; comparison with logistic regression |
| [Chemistry-only wine clustering](projects/wine-clustering/) | Are visually different groups actually well separated? | Quality excluded from fitting; cluster separation, seed stability, and PCA variance |
| [Passenger classification](projects/spaceship-titanic/) | Does the comparison hold on entirely new travel groups? | Grouped holdout, fold-safe preparation, fixed linear and forest baselines |
| [Grouped experiments and non-compliance](projects/clustering-noncompliance/) | Is a pilot's apparent precision justified? | Store-level uncertainty and corrected incremental treatment arithmetic |
| [Dynamic Zoom contribution](projects/dynamic-zoom/) | Why can fast inference still produce lag? | Collaborative source link, frame-queue simulation, latency and dropped-frame accounting |
| [Restaurant review analysis](projects/yelp-reviews/) | What can word frequencies support? | Reviewed historical notebook, sampling and labeling decisions, visualization correction |

## Run something in under a minute

Eight compact references run on Python 3.10+ with no packages or accounts:

```bash
python projects/forecasting/demo.py
python projects/commercial-reporting/demo.py
python projects/knowledge-search/demo.py
python projects/optimization/demo.py
python projects/event-prediction/demo.py
python projects/ml-from-scratch/demo.py
python projects/clustering-noncompliance/demo.py
python projects/dynamic-zoom/demo.py
python -m unittest discover -s tests -v
```

For the four maintained real-data analyses, use Python 3.11+ and the optional environment:

```bash
python -m pip install -r requirements-analysis.txt
python projects/play-store-reviews/analyze.py
python projects/wine-clustering/analyze.py
python projects/wine-neural-networks/analyze.py
python projects/spaceship-titanic/analyze.py
```

Every analysis records its dataset hash, package versions, split definition, and metrics. `--write` explicitly refreshes its saved result. [The reviewer guide](docs/reviewer-guide.md) points to the most useful files; [the evidence notes](docs/evidence.md) distinguish professional outcomes, rerun analyses, and historical outputs.

## Experience beyond the model

At BHP I coordinated **15+ specialists** on an Asset Health initiative supporting predictive maintenance for mining-truck components. Earlier, I investigated Americas payroll issues and developed an investment case that secured **$5M+ for technology upgrades**. Those roles shaped how I approach technical work: understand the process, identify what a team needs to decide, and make the implementation usable in that context.

I hold a **master's in Machine Learning & Signal Processing** and a **master's in Business Analytics** from the University of Wisconsin–Madison, plus bachelor's degrees in Engineering Sciences and Industrial Civil Engineering. My research includes publications on augmented-reality training and creativity in business economics. [Full experience, education, and publications →](docs/experience.md)

## Repository notes

Original notebooks and research scripts remain in each project's `legacy/` folder. Maintained implementations sit beside them with current articles and saved results. [Earlier file locations](docs/earlier-file-locations.md) preserve the migration trail; [the archive index](archive/) locates additional historical material.

The repository is provided for portfolio review and retains its existing licensing status. Third-party datasets and referenced implementations retain their own terms; the [included UCI wine data](data/public/) is attributed under CC BY 4.0.
