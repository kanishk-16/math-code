import math
from collections import Counter

# 1. Corpus & Tokenization
docs = [
    "deep learning and natural language processing",
    "retrieval augmented generation with vector search",
    "bm25 keyword search and natural language processing",
]

tokenized_docs = [doc.lower().split() for doc in docs]
N = len(tokenized_docs)
avgdl = sum(len(d) for d in tokenized_docs) / N
k1, b = 1.5, 0.75

# 2. Document Frequencies & IDF
doc_counts = Counter(word for d in tokenized_docs for word in set(d))
idf = {
    word: math.log((N - count + 0.5) / (count + 0.5) + 1.0)
    for word, count in doc_counts.items()
}


# 3. BM25 Scoring Function
def bm25_score(query: str):
    query_tokens = query.lower().split()
    scores = []

    for idx, doc in enumerate(tokenized_docs):
        score = 0.0
        doc_len = len(doc)
        tf = Counter(doc)

        for token in query_tokens:
            if token in tf:
                # Term Frequency with length normalization
                freq = tf[token]
                numerator = freq * (k1 + 1.0)
                denominator = freq + k1 * (1.0 - b + b * (doc_len / avgdl))
                score += idf.get(token, 0.0) * (numerator / denominator)

        scores.append((score, docs[idx]))

    return sorted(scores, reverse=True)


# 4. Example Run
query = "natural language bm25"
for score, doc in bm25_score(query):
    print(f"[{score:.4f}] {doc}")