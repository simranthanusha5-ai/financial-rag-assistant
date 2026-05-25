import numpy as np
from embeddings import model

def retrieve(query, index, chunks, top_k=3):
    query_embedding = model.encode([query])

    actual_top_k = min(top_k, len(chunks))

    distances, indices = index.search(
        np.array(query_embedding).astype("float32"),
        actual_top_k
    )

    retrieved_chunks = [chunks[i] for i in indices[0]]

    return retrieved_chunks