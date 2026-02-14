from __future__ import annotations
import os
import pandas as pd

from .data import load_ledgar
from .chunking import chunk_fixed, chunk_sliding, chunk_paragraphs
from .embed_index import build_faiss_index
from .eval_retrieval import evaluate_retrieval


def run():
    print("START run()", flush=True)

    print("Loading CUAD...", flush=True)
    docs, qas = load_ledgar(max_docs=200, max_qas=800)
    print(f"Loaded docs={len(docs)} qas={len(qas)}", flush=True)

    strategies = [
        ("fixed", lambda txt, doc_id: chunk_fixed(txt, doc_id, chunk_size_chars=2000)),
        ("sliding", lambda txt, doc_id: chunk_sliding(txt, doc_id, window_chars=2000, overlap_chars=400)),
        ("paragraph", lambda txt, doc_id: chunk_paragraphs(txt, doc_id, min_len=200)),
    ]

    rows = []
    for name, fn in strategies:
        chunks = []
        for d in docs:
            chunks.extend(fn(d.text, d.doc_id))

        bundle = build_faiss_index(chunks)

        for k in (5, 10):
            m = evaluate_retrieval(bundle, qas, top_k=k)
            rows.append(
                {
                    "strategy": name,
                    "top_k": k,
                    "n_docs": len(docs),
                    "n_qas": m.n_queries,
                    "n_chunks": len(chunks),
                    "recall@k": m.recall_at_k,
                    "precision@k": m.precision_at_k,
                    "mrr": m.mrr,
                }
            )
            print(f"[{name}] k={k} recall={m.recall_at_k:.3f} precision={m.precision_at_k:.3f} mrr={m.mrr:.3f}")

    df = pd.DataFrame(rows).sort_values(["top_k", "strategy"])
    os.makedirs("results", exist_ok=True)
    out_path = "results/retrieval_baseline.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    run()
