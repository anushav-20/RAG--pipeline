# Evaluating RAG systems

Evaluate retrieval and generation separately. Retrieval metrics include Hit@k, which is the fraction of questions where a relevant passage appears in the top k results, and Mean Reciprocal Rank (MRR), which averages 1 divided by the rank of the first relevant passage. Recall@k and nDCG are also common.

Generation is judged on faithfulness (is every claim supported by the retrieved context), answer relevancy (does it address the question), and context precision. Frameworks such as RAGAS automate these with an LLM acting as judge.

A good evaluation set contains 50 to 200 real questions with labelled relevant sources, including questions that the corpus cannot answer, to test that the system abstains instead of hallucinating. Run the evaluation in continuous integration so that changes to chunk size, embedding model, or prompts cannot silently regress quality.
