# Unsupervised wine analysis

## Question and approach

Look for structure in measured wine characteristics without using quality as a training target. Exploratory clustering and visualization.

## Inspect the original work

**[Open the notebook](legacy/Unsupervised%20Learning%20-%20Wine%20Quality.ipynb)**. It contains 18 code cells, with the original analysis and saved outputs. Those outputs are historical experiment evidence, not a newly reproduced benchmark.

Main imports found in the notebook: `matplotlib.pyplot, numpy, pandas, seaborn, sklearn.cluster, sklearn.decomposition, sklearn.metrics, sklearn.preprocessing`.

### Code excerpt from the notebook

```python
# Import the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import adjusted_rand_score
```

## Data and reproducibility

The notebook is preserved as submitted in the original portfolio. It may refer to local files or external datasets not present here. Dependencies are not pinned and the saved results have not been rerun. Review data-loading cells before attempting execution. New reproducible examples are listed in the repository root README; this historical notebook is not presented as one of them.
