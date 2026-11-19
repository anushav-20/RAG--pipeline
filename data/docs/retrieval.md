# Hybrid retrieval

Dense retrieval embeds the query and finds the nearest chunk vectors by cosine similarity. It captures meaning and paraphrase, but can miss exact identifiers such as error codes, product names, or rare acronyms.

BM25 is a sparse lexical ranking function. It scores documents by term frequency, inverse document frequency, and document length normalisation, with parameters k1 and b. BM25 excels at exact keyword matches but cannot match synonyms.

Hybrid search runs both retrievers and merges the results. Reciprocal Rank Fusion (RRF) scores each document as the sum of 1 divided by (k + rank) across the rankings, with k usually 60. RRF needs no score normalisation, which makes it robust because BM25 scores and cosine similarities are on different scales.

A reranker then reorders the top candidates. A cross-encoder reads the query and passage together and gives a much more accurate relevance score than a bi-encoder, at higher cost, so it is applied only to the top 20 to 50 candidates.
