from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from datasets import load_dataset


@dataclass
class Doc:
    doc_id: str
    text: str


@dataclass
class QA:
    doc_id: str
    question: str
    answer_text: str


def load_ledgar(max_docs: int | None = 200, max_qas: int | None = 800) -> Tuple[List[Doc], List[QA]]:
    """
    Loads LexGLUE LEDGAR (contract clause classification) and converts it into
    a weakly-supervised retrieval task.

    Retrieval task definition:
      - Each contract = a document (corpus item)
      - Each clause label becomes a "question" like: "Find clauses about: <LABEL>"
      - The clause text itself is the gold "answer_text" that should be retrieved

    This is safe, public, and gives you measurable Recall@k / MRR for chunking strategies.
    """
    ds = load_dataset("lex_glue", "ledgar")

    # Use validation for fast iteration
    split = ds["validation"]

    docs: Dict[str, str] = {}
    qas: List[QA] = []

    doc_count = 0
    qa_count = 0

    for i, row in enumerate(split):
        # LEDGAR rows typically contain: "text" (clause), "label" (category)
        clause_text = row.get("text", "")
        label = row.get("label", None)

        if not clause_text or label is None:
            continue

        # Create a pseudo-document by grouping clauses into batches (since LEDGAR is clause-level)
        # We’ll bundle every N clauses into a single "doc" so chunking has something real to do.
        # This avoids needing the original full contract documents.
        bundle_id = f"bundle_{i // 20}"  # 20 clauses per pseudo-doc

        docs.setdefault(bundle_id, "")
        docs[bundle_id] += ("\n\n" + clause_text.strip())

        question = f"Find clauses about: {label}"
        answer_text = clause_text.strip()

        qas.append(QA(doc_id=bundle_id, question=question, answer_text=answer_text))
        qa_count += 1

        if max_qas and qa_count >= max_qas:
            break

    # Cap docs
    doc_ids = list(docs.keys())
    if max_docs:
        doc_ids = doc_ids[:max_docs]

    docs_list = [Doc(doc_id=d, text=docs[d]) for d in doc_ids]
    doc_set = {d.doc_id for d in docs_list}
    qas = [qa for qa in qas if qa.doc_id in doc_set]

    return docs_list, qas
