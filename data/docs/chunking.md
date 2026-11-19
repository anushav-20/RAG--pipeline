# Chunking strategies

Chunking splits documents into passages small enough to embed and retrieve precisely. Fixed-size chunking cuts every N characters and is simple but can split sentences in half. Recursive chunking tries paragraph breaks first, then line breaks, then sentences, then words, so chunks follow the natural structure of the text.

Chunk overlap repeats the tail of one chunk at the start of the next. Overlap of 10 to 20 percent prevents an answer from being lost when it straddles a boundary. Typical chunk sizes are 200 to 1000 tokens. Smaller chunks give precise retrieval but lose context; larger chunks keep context but dilute the embedding and waste prompt tokens.

Semantic chunking places boundaries where the embedding similarity between adjacent sentences drops. Parent-document retrieval indexes small chunks but returns the larger parent section to the language model.
