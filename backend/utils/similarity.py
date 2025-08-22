import numpy as np
from typing import Any, Dict, List


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    a_np = np.array(a)
    b_np = np.array(b)
    
    dot_product = np.dot(a_np, b_np)
    norm_a = np.linalg.norm(a_np)
    norm_b = np.linalg.norm(b_np)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return float(dot_product / (norm_a * norm_b))


def find_similar_documents(
    query_embedding: List[float],
    document_embeddings: List[Dict[str, Any]],
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Find the most similar documents to a query embedding.
    
    Args:
        query_embedding: The embedding vector of the query
        document_embeddings: List of dicts with 'embedding' and other metadata
        top_k: Number of top results to return
    
    Returns:
        List of documents sorted by similarity score (highest first)
    """
    similarities = []
    
    for doc in document_embeddings:
        if 'embedding' not in doc:
            continue
            
        similarity = cosine_similarity(query_embedding, doc['embedding'])
        doc_copy = doc.copy()
        doc_copy['similarity_score'] = similarity
        similarities.append(doc_copy)
    
    # Sort by similarity score (descending) and return top_k
    similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
    return similarities[:top_k]