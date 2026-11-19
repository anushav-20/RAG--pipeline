# Running RAG in production

Hallucination control: instruct the model to answer only from the supplied context, require inline citations, and return a fixed abstention phrase when the context lacks the answer. Treat retrieved text as untrusted data, because documents can contain prompt injection instructions.

Latency and cost: cache embeddings, use prompt caching for the stable system prompt, retrieve a small top k, and stream the answer to the user. Use approximate nearest neighbour indexes such as HNSW once the corpus exceeds a few hundred thousand chunks.

Freshness: store a content hash per document and re-embed only changed files. Keep metadata such as source, page number, and timestamp so answers can cite exactly where information came from. Add access control filters at retrieval time so users only see chunks they are permitted to read.

Observability: log the query, retrieved chunk ids, scores, and answer for every request so failures can be traced to retrieval or generation.
