# App-store review text analytics

## Question and approach

Explore review text to identify recurring user concerns and sentiment. Text cleaning, term exploration, and interpretation of review patterns.

## Inspect the original work

**[Open the notebook](legacy/Google%20Play%20Store%20Review%20-%20%20Text%20Analytics.ipynb)**. It contains 68 code cells, with the original analysis and saved outputs. Those outputs are historical experiment evidence, not a newly reproduced benchmark.

Main imports found in the notebook: `datetime, en_core_web_sm, gensim, gensim.corpora, gensim.models, gensim.models.callbacks, gensim.models.coherencemodel, matplotlib.pyplot, nltk, nltk.corpus, nltk.stem, numpy, pandas, sklearn.decomposition, sklearn.feature_extraction.text, sklearn.metrics, sklearn.model_selection, spacy, spacy.lang.en, spacy.lang.en.stop_words, time, torch, torch.utils.data, transformers, warnings`.

### Code excerpt from the notebook

```python
# Import necessary libraries
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from gensim.models import TfidfModel, LsiModel
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.callbacks import PerplexityMetric


# Define stop words and lemmatizer
stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

# Define text preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Tokenize text
    tokens = nltk.word_tokenize(text)
    # Remove stop words and words with length < 3
    tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
```

## Data and reproducibility

The notebook is preserved as submitted in the original portfolio. It may refer to local files or external datasets not present here. Dependencies are not pinned and the saved results have not been rerun. Review data-loading cells before attempting execution. New reproducible examples are listed in the repository root README; this historical notebook is not presented as one of them.
