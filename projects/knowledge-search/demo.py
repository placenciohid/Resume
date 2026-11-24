"""Evidence retrieval: BM25 + character TF-IDF, reciprocal rank fusion, reranking.

All documents are fictional. This is not a copy of the employer search platform.
"""
from collections import Counter
from math import log, sqrt
import json
import re


DOCUMENTS = [
    {'id':'maintenance','title':'Fictional maintenance guide','text':'Inspect pump seals before restarting equipment. Record vibration measurements and isolate the pump if abnormal vibration continues.'},
    {'id':'receivables','title':'Fictional receivables guide','text':'Review overdue invoices by customer and currency. Separate disputed invoices from payment matching exceptions before collections follow up.'},
    {'id':'search','title':'Fictional search design note','text':'Preserve document identifiers and page references when extracting text with OCR. Search results must link back to their source.'},
    {'id':'forecast','title':'Fictional forecasting note','text':'Evaluate a production forecast against a seasonal baseline. Fit all transformations on historical training observations before a validation period.'},
    {'id':'remittance','title':'Fictional remittance guide','text':'A remittance document can contain several invoice references. Mark the document fully cleared only when every invoice line is matched and the payment difference is within tolerance.'},
    {'id':'safety','title':'Fictional safety note','text':'Equipment isolation must be confirmed before maintenance begins. Consult the approved site safety procedure for actual work.'},
]


def tokens(text): return re.findall(r'[a-z0-9]+', text.lower())


def chunks(documents, size=36, overlap=8):
    if not 0 <= overlap < size: raise ValueError('Require 0 <= overlap < chunk size')
    result = []
    for doc in documents:
        words = doc['text'].split()
        for start in range(0, len(words), size-overlap):
            result.append({'chunk_id':f"{doc['id']}:{start}", 'document_id':doc['id'],
                           'title':doc['title'], 'text':' '.join(words[start:start+size]),
                           'source':f"synthetic/{doc['id']}.txt"})
            if start+size >= len(words): break
    return result


def bm25(query, corpus):
    words = [tokens(c['text']) for c in corpus]
    avg = sum(map(len,words))/len(words)
    scores = []
    for row in words:
        counts = Counter(row)
        score = 0.0
        for term in set(tokens(query)):
            df = sum(term in other for other in words)
            idf = log(1+(len(words)-df+.5)/(df+.5))
            tf = counts[term]
            score += idf*(tf*2.5)/(tf+1.5*(.25+.75*len(row)/avg))
        scores.append(score)
    return scores


def grams(text):
    text = ' '.join(tokens(text))
    return Counter(text[i:i+3] for i in range(max(0,len(text)-2)))


def vector_scores(query, corpus):
    counts = [grams(c['text']) for c in corpus]
    query_counts = grams(query)
    idf = {g:log((1+len(counts))/(1+sum(g in c for c in counts)))+1 for g in set().union(*counts)}
    q = {g:tf*idf.get(g,0) for g,tf in query_counts.items()}
    qnorm = sqrt(sum(v*v for v in q.values()))
    scores = []
    for c in counts:
        values = {g:tf*idf[g] for g,tf in c.items()}
        denom = qnorm*sqrt(sum(v*v for v in values.values()))
        scores.append(sum(q.get(g,0)*v for g,v in values.items())/denom if denom else 0)
    return scores


def search(query, corpus=None, limit=3):
    corpus = chunks(DOCUMENTS) if corpus is None else corpus
    if not tokens(query) or not corpus: return []
    lexical, vector = bm25(query, corpus), vector_scores(query, corpus)
    fused = Counter()
    for values in (lexical, vector):
        for rank,index in enumerate(sorted(range(len(corpus)), key=lambda i:values[i], reverse=True),1):
            if values[index] > 0: fused[index] += 1/(60+rank)
    candidates = []
    for index,score in fused.items():
        coverage = len(set(tokens(query)) & set(tokens(corpus[index]['text'])))/len(set(tokens(query)))
        # Require lexical evidence; char overlap alone is not a reliable answer.
        if coverage == 0: continue
        candidates.append({**corpus[index], 'fusion_score':round(score,6),
                           'query_coverage':round(coverage,4), 'score':score+0.1*coverage})
    return sorted(candidates,key=lambda c:c['score'],reverse=True)[:limit]


def run():
    cases = [('overdue customer invoices','receivables'), ('pump vibration','maintenance'),
             ('OCR source document','search'), ('forecast seasonal validation','forecast'),
             ('remittance fully cleared invoice','remittance')]
    results = [{'query':q, 'expected':expected, 'top_document':search(q)[0]['document_id']} for q,expected in cases]
    return {'dataset':'six fictional documents; five hand-written evaluation queries',
            'top1_hits':sum(r['expected']==r['top_document'] for r in results), 'queries':results,
            'example_evidence':search('remittance fully cleared invoice')}


if __name__ == '__main__': print(json.dumps(run(),indent=2))
