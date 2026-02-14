from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .chunking import Chunk


@dataclass
class IndexBundle:
    index: faiss.Index
    chunk_ids: List[str]
    chunk_texts: List[str]
    chunk_doc_ids: List[str]
    embed_model_name: str


def build_faiss_index(
    chunks: List[Chunk],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64,
) -> IndexBundle:
    model = SentenceTransformer(model_name)

    texts = [c.text for c in chunks]
    chunk_ids = [c.chunk_id for c in chunks]
    chunk_doc_ids = [c.doc_id for c in chunks]

    embeddings: List[np.ndarray] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding chunks"):
        batch = texts[i : i + batch_size]
        emb = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        embeddings.append(np.asarray(emb, dtype=np.float32))

    X = np.vstack(embeddings)
    dim = X.shape[1]

    index = faiss.IndexFlatIP(dim)  # cosine since normalized
    index.add(X)

    return IndexBundle(
        index=index,
        chunk_ids=chunk_ids,
        chunk_texts=texts,
        chunk_doc_ids=chunk_doc_ids,
        embed_model_name=model_name,
    )


def retrieve(
    bundle: IndexBundle,
    query: str,
    top_k: int = 10,
    model: SentenceTransformer | None = None,
) -> List[int]:
    """
    Returns indices into bundle arrays for top_k results.
    """
    if model is None:
        model = SentenceTransformer(bundle.embed_model_name)
    q = model.encode([query], normalize_embeddings=True, show_progress_bar=False)
    q = np.asarray(q, dtype=np.float32)
    scores, idxs = bundle.index.search(q, top_k)
    return idxs[0].tolist()
