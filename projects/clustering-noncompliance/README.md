# Clustering and non-compliance

## Question and approach

Explore experimental observations, clustering, and compliance patterns. Exploratory clustering and interpretation of groups within the original experiment.

## Inspect the original work

**[Open the notebook](legacy/Clustering%20and%20Non-Compliance.ipynb)**. It contains 19 code cells, with the original analysis and saved outputs. Those outputs are historical experiment evidence, not a newly reproduced benchmark.

Main imports found in the notebook: `matplotlib.pyplot, numpy, pandas, seaborn, sklearn.linear_model, sklearn.metrics, sklearn.model_selection, statsmodels.api`.

### Code excerpt from the notebook

```python
# Import The Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
```

## Data and reproducibility

The notebook is preserved as submitted in the original portfolio. It may refer to local files or external datasets not present here. Dependencies are not pinned and the saved results have not been rerun. Review data-loading cells before attempting execution. New reproducible examples are listed in the repository root README; this historical notebook is not presented as one of them.
