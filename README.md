# Dario Placencio — Data Science & Operations Portfolio

Selected technical work in machine learning, data engineering, and optimization.
I work in Operational Excellence at Rio Tinto, with earlier experience in data science and process improvement at Rio Tinto and BHP.

**[Website](https://placenciohid.github.io/) · [Experience](https://placenciohid.github.io/aboutme.html) · [LinkedIn](https://www.linkedin.com/in/darioplacencio/)**

## Start here

| Project | Question | Runnable example | What to inspect |
| --- | --- | --- | --- |
| [Event prediction](projects/event-prediction/) | How do we turn event history into training data without using the future? | `python projects/event-prediction/demo.py` | As-of labels, event resolution, temporal split, training-only scaling |
| [Optimization](projects/optimization/) | Which procedures fit within a limited schedule? | `python projects/optimization/demo.py` | Objective, capacity constraints, exact solution and a baseline |
| [ML from scratch](projects/ml-from-scratch/) | How do basic classifiers learn a decision boundary? | `python projects/ml-from-scratch/demo.py` | kNN, a decision stump, logistic regression, held-out evaluation |
| [Forecast evaluation](projects/forecasting/) | How do we evaluate a forecast without looking ahead? | `python projects/forecasting/demo.py` | Rolling origins, baselines, forecast errors |
| [Evidence retrieval](projects/knowledge-search/) | How do we retrieve passages with source references? | `python projects/knowledge-search/demo.py` | Chunking, BM25, TF-IDF, fusion, reranking |
| [Receivables and reconciliation](projects/commercial-reporting/) | What makes a remittance fully matched? | `python projects/commercial-reporting/demo.py` | Aging, currency separation, whole-document counts |

## Additional project evidence

| Project | Repository evidence |
| --- | --- |
| [Dynamic Zoom contribution](projects/dynamic-zoom/) | Pipeline notes, integration sketch, and original upstream implementation |
| [Play Store reviews](projects/play-store-reviews/) | Original text-analysis notebook and technical notes |
| [Yelp reviews](projects/yelp-reviews/) | Original notebook and technical notes |
| [Wine clustering](projects/wine-clustering/) | Original unsupervised-learning notebook |
| [Wine neural networks](projects/wine-neural-networks/) | Original model notebook |
| [Clustering and non-compliance](projects/clustering-noncompliance/) | Original experimental analysis notebook |
| [Spaceship Titanic](projects/spaceship-titanic/) | Both original notebooks and reproduction limits |

## Run locally

Use Python 3.10 or later. Clone the repository, open a terminal in its root, and run any command above. No packages, private datasets, accounts, or GPU are required for these small demonstrations. Their synthetic inputs are deliberately simple; results demonstrate implementation behavior, not business performance.

Run checks with `python -m unittest discover -s tests -v`.

## Original work and limitations

The `legacy` folders retain the original notebooks and scripts. They may require external coursework data, Spark, Julia, Pyomo, a solver, or paths from the original environment. They are historical work, **not** the runnable examples above. The new demonstrations make a small part of each approach reproducible; they do not reproduce historical model results.

[Earlier notebooks](archive/) cover text analytics, clustering, neural networks, and Kaggle experiments. [Legacy datasets](data/legacy/) are retained for existing notebooks; their presence does not establish redistribution rights. The new examples use synthetic data instead.

## File migration

Older website links now point to these locations. Commit-specific links in Git history remain unchanged.

| Original root file | Current location |
| --- | --- |
| Building Machine Learning Models from Scratch - Decision Trees, kNN, and Logistic Regression.ipynb | [projects/ml-from-scratch/legacy/Building Machine Learning Models from Scratch - Decision Trees, kNN, and Logistic Regression.ipynb](projects/ml-from-scratch/legacy/Building%20Machine%20Learning%20Models%20from%20Scratch%20-%20Decision%20Trees%2C%20kNN%2C%20and%20Logistic%20Regression.ipynb) |
| Clustering and Non-Compliance.ipynb | [projects/clustering-noncompliance/legacy/Clustering and Non-Compliance.ipynb](projects/clustering-noncompliance/legacy/Clustering%20and%20Non-Compliance.ipynb) |
| Data_PipeLine_for_Event_Prediction.py | [projects/event-prediction/legacy/Data_PipeLine_for_Event_Prediction.py](projects/event-prediction/legacy/Data_PipeLine_for_Event_Prediction.py) |
| GEV-NN-Model.py | [projects/event-prediction/legacy/GEV-NN-Model.py](projects/event-prediction/legacy/GEV-NN-Model.py) |
| GNN-INCM_with_HSKDM.py | [projects/event-prediction/legacy/GNN-INCM_with_HSKDM.py](projects/event-prediction/legacy/GNN-INCM_with_HSKDM.py) |
| Google Play Store Review -  Text Analytics.ipynb | [projects/play-store-reviews/legacy/Google Play Store Review -  Text Analytics.ipynb](projects/play-store-reviews/legacy/Google%20Play%20Store%20Review%20-%20%20Text%20Analytics.ipynb) |
| Julia Optimization - Sudoki and Brain Puzzle.ipynb | [projects/optimization/legacy/Julia Optimization - Sudoki and Brain Puzzle.ipynb](projects/optimization/legacy/Julia%20Optimization%20-%20Sudoki%20and%20Brain%20Puzzle.ipynb) |
| Neural Networks Application - Wine Quality.ipynb | [projects/wine-neural-networks/legacy/Neural Networks Application - Wine Quality.ipynb](projects/wine-neural-networks/legacy/Neural%20Networks%20Application%20-%20Wine%20Quality.ipynb) |
| Pyomo Optimization - Operating Room Schedule.ipynb | [projects/optimization/legacy/Pyomo Optimization - Operating Room Schedule.ipynb](projects/optimization/legacy/Pyomo%20Optimization%20-%20Operating%20Room%20Schedule.ipynb) |
| Spaceship Titanic - Kaggle Competition.ipynb | [projects/spaceship-titanic/legacy/Spaceship Titanic - Kaggle Competition.ipynb](projects/spaceship-titanic/legacy/Spaceship%20Titanic%20-%20Kaggle%20Competition.ipynb) |
| Spaceship Titanic - Kaggle.ipynb | [projects/spaceship-titanic/legacy/Spaceship Titanic - Kaggle.ipynb](projects/spaceship-titanic/legacy/Spaceship%20Titanic%20-%20Kaggle.ipynb) |
| Text Analytics - Yelp Reviews.ipynb | [projects/yelp-reviews/legacy/Text Analytics - Yelp Reviews.ipynb](projects/yelp-reviews/legacy/Text%20Analytics%20-%20Yelp%20Reviews.ipynb) |
| Unsupervised Learning - Wine Quality.ipynb | [projects/wine-clustering/legacy/Unsupervised Learning - Wine Quality.ipynb](projects/wine-clustering/legacy/Unsupervised%20Learning%20-%20Wine%20Quality.ipynb) |
| reviews.csv | [data/legacy/reviews.csv](data/legacy/reviews.csv) |
| submission.csv | [data/legacy/submission.csv](data/legacy/submission.csv) |
| test.csv | [data/legacy/test.csv](data/legacy/test.csv) |
| train.csv | [data/legacy/train.csv](data/legacy/train.csv) |

## Use and attribution

The original repository did not grant a general open-source license. This update does not change that: the code is provided for portfolio review; contact the author before reuse. Third-party libraries, datasets, and referenced research retain their own terms. No employer source code or internal datasets are included in the runnable demonstrations.
