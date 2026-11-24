# Evidence retrieval — chunking, hybrid search, and source links

## Professional context and public companion

My professional search work covered more than 300,000 documents and was adopted by four teams. This small independent reference uses six fictional documents. It illustrates source handling and retrieval decisions, not the employer implementation or its measured quality.

## Reproduce

`python projects/knowledge-search/demo.py`

Python 3.10+, no packages or services. Inspect [demo.py](demo.py) and [results.json](results.json).

## Retrieval stages

1. Tokenize text and split it into overlapping chunks, retaining document ID, chunk ID, title, and source reference.
2. BM25 ranks exact terms with inverse document frequency and length normalization.
3. A character-trigram TF-IDF index ranks text by cosine similarity. This is a lexical vector representation, **not** a semantic embedding model.
4. Reciprocal rank fusion combines both rankings; query-token coverage supplies a simple reranking signal.
5. Return the source excerpt and reference. There is no LLM answer generator in the demo. With no lexical evidence, return no answer.

An OCR adapter would produce the same text-plus-source contract from images/PDFs before chunking. The example begins with text and does not claim to run OCR. Production authorization must be enforced before retrieval, not after the model has read the document.

## Evaluation

Five hand-written queries check the expected top-ranked document. The current snapshot retrieves all five expected sources. The queries were designed with the tiny corpus, so this is a regression fixture, not an unbiased search-quality benchmark. Additional checks verify verbatim source evidence, empty queries, and no-evidence abstention.

## Tradeoffs

BM25 handles exact vocabulary well; trigram vectors tolerate surface variation but are not semantic understanding. Fusion avoids comparing incomparable raw scores. Word overlap is a cheap reranker and can miss paraphrases. A larger implementation would substitute embeddings and a learned reranker, preserve page-level citations, deduplicate versions, and evaluate relevance on held-out questions.
