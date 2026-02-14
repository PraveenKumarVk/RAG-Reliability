from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import re


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def chunk_fixed(text: str, doc_id: str, chunk_size_chars: int = 2000) -> List[Chunk]:
    """
    Simple character-based chunking for baseline (safe, fast).
    """
    t = text
    chunks: List[Chunk] = []
    i = 0
    idx = 0
    while i < len(t):
        piece = _normalize_ws(t[i : i + chunk_size_chars])
        if piece:
            chunks.append(Chunk(chunk_id=f"{doc_id}::fixed::{idx}", doc_id=doc_id, text=piece))
            idx += 1
        i += chunk_size_chars
    return chunks


def chunk_sliding(text: str, doc_id: str, window_chars: int = 2000, overlap_chars: int = 400) -> List[Chunk]:
    """
    Character sliding window chunking with overlap.
    """
    assert overlap_chars < window_chars
    t = text
    chunks: List[Chunk] = []
    step = window_chars - overlap_chars
    i = 0
    idx = 0
    while i < len(t):
        piece = _normalize_ws(t[i : i + window_chars])
        if piece:
            chunks.append(Chunk(chunk_id=f"{doc_id}::slide::{idx}", doc_id=doc_id, text=piece))
            idx += 1
        i += step
    return chunks


def chunk_paragraphs(text: str, doc_id: str, min_len: int = 200) -> List[Chunk]:
    """
    Split by blank lines, keep only reasonably sized paragraphs.
    """
    parts = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    chunks: List[Chunk] = []
    idx = 0
    for p in parts:
        p2 = _normalize_ws(p)
        if len(p2) >= min_len:
            chunks.append(Chunk(chunk_id=f"{doc_id}::para::{idx}", doc_id=doc_id, text=p2))
            idx += 1
    # fallback: if nothing passes min_len, return a fixed chunk
    if not chunks:
        return chunk_fixed(text, doc_id, chunk_size_chars=2000)
    return chunks
