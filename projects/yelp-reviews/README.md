# Yelp review text analytics

## Question and approach

Explore restaurant review text and the signals associated with ratings. Text preprocessing and exploratory analysis of review language.

## Inspect the original work

**[Open the notebook](legacy/Text%20Analytics%20-%20Yelp%20Reviews.ipynb)**. It contains 20 code cells, with the original analysis and saved outputs. Those outputs are historical experiment evidence, not a newly reproduced benchmark.

Main imports found in the notebook: `collections, matplotlib.pyplot, nltk, nltk.corpus, numpy, pandas, seaborn, string, textblob, textblob.sentiments, warnings, wordcloud`.

### Code excerpt from the notebook

```python
# Create a function to extract the most common words from the text column
def common_words(df):
    words = []
    for i in df['review_text']:
        words.extend(i.split())
    words = [word.lower() for word in words if word.lower() not in stop]
    return Counter(words).most_common(200)
```

## Data and reproducibility

The notebook is preserved as submitted in the original portfolio. It may refer to local files or external datasets not present here. Dependencies are not pinned and the saved results have not been rerun. Review data-loading cells before attempting execution. New reproducible examples are listed in the repository root README; this historical notebook is not presented as one of them.
