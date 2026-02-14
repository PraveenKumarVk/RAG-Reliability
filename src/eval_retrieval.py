from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np

from .embed_index import IndexBundle, retrieve
from .data import QA


@dataclass
class RetrievalMetrics:
    recall_at_k: float
    precision_at_k: float
    mrr: float
    n_queries: int


def _contains_answer(chunk_text: str, answer_text: str) -> bool:
    # weak-label match (string containment). Good enough for baseline.
    # Later we can improve with fuzzy matching.
    return answer_text.lower() in chunk_text.lower()


def evaluate_retrieval(
    bundle: IndexBundle,
    qas: List[QA],
    top_k: int = 10,
) -> RetrievalMetrics:
    hits = 0
    precisions: List[float] = []
    rr_sum = 0.0

    for qa in qas:
        idxs = retrieve(bundle, qa.question, top_k=top_k)
        retrieved_texts = [bundle.chunk_texts[i] for i in idxs if i >= 0]

        match_positions = []
        for rank, txt in enumerate(retrieved_texts, start=1):
            if _contains_answer(txt, qa.answer_text):
                match_positions.append(rank)

        if match_positions:
            hits += 1
            rr_sum += 1.0 / min(match_positions)
            # precision@k: fraction of retrieved chunks that contain answer
            precisions.append(len(match_positions) / float(top_k))
        else:
            precisions.append(0.0)

    recall_at_k = hits / float(len(qas)) if qas else 0.0
    precision_at_k = float(np.mean(precisions)) if precisions else 0.0
    mrr = rr_sum / float(len(qas)) if qas else 0.0

    return RetrievalMetrics(
        recall_at_k=recall_at_k,
        precision_at_k=precision_at_k,
        mrr=mrr,
        n_queries=len(qas),
    )
