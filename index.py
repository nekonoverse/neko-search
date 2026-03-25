"""In-memory inverted index with BM25 scoring."""

import logging
import math
from collections import defaultdict

logger = logging.getLogger(__name__)

# BM25 parameters
K1 = 1.2
B = 0.75


class InvertedIndex:
    """Subword-based inverted index with BM25 scoring."""

    def __init__(self):
        # subword -> {note_id: term_frequency}
        self.postings: dict[str, dict[str, int]] = defaultdict(dict)
        # note_id -> document length (token count)
        self.doc_lengths: dict[str, int] = {}
        self.total_docs: int = 0
        self.avg_doc_length: float = 0.0

    def _update_avg(self):
        if self.total_docs > 0:
            self.avg_doc_length = sum(self.doc_lengths.values()) / self.total_docs
        else:
            self.avg_doc_length = 0.0

    def add(self, note_id: str, tokens: list[str]):
        """Add or update a document in the index."""
        # Remove old postings if document exists
        if note_id in self.doc_lengths:
            self._remove_postings(note_id)

        # Build term frequencies
        tf: dict[str, int] = defaultdict(int)
        for token in tokens:
            tf[token] += 1

        # Add to postings
        for token, count in tf.items():
            self.postings[token][note_id] = count

        self.doc_lengths[note_id] = len(tokens)
        self.total_docs = len(self.doc_lengths)
        self._update_avg()

    def remove(self, note_id: str) -> bool:
        """Remove a document from the index."""
        if note_id not in self.doc_lengths:
            return False
        self._remove_postings(note_id)
        del self.doc_lengths[note_id]
        self.total_docs = len(self.doc_lengths)
        self._update_avg()
        return True

    def _remove_postings(self, note_id: str):
        """Remove all postings for a given document."""
        empty_tokens = []
        for token, docs in self.postings.items():
            if note_id in docs:
                del docs[note_id]
                if not docs:
                    empty_tokens.append(token)
        for token in empty_tokens:
            del self.postings[token]

    def search(self, query_tokens: list[str], limit: int = 20) -> list[tuple[str, float]]:
        """Search with BM25 scoring.

        Returns list of (note_id, score) sorted by score descending.
        """
        if not query_tokens or self.total_docs == 0:
            return []

        scores: dict[str, float] = defaultdict(float)
        avgdl = self.avg_doc_length or 1.0

        for token in query_tokens:
            posting = self.postings.get(token)
            if not posting:
                continue

            # IDF: log((N - df + 0.5) / (df + 0.5) + 1)
            df = len(posting)
            idf = math.log((self.total_docs - df + 0.5) / (df + 0.5) + 1.0)

            for note_id, tf in posting.items():
                dl = self.doc_lengths.get(note_id, 0)
                # BM25 TF component
                tf_norm = (tf * (K1 + 1)) / (tf + K1 * (1 - B + B * dl / avgdl))
                scores[note_id] += idf * tf_norm

        # Sort by score descending, take top N
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:limit]

    def rebuild(self, documents):
        """Rebuild the entire index from an iterable of documents.

        Each document should have 'note_id' and 'tokens' keys.
        """
        self.postings = defaultdict(dict)
        self.doc_lengths = {}
        count = 0
        for doc in documents:
            note_id = doc["note_id"]
            tokens = doc["tokens"]
            tf: dict[str, int] = defaultdict(int)
            for token in tokens:
                tf[token] += 1
            for token, c in tf.items():
                self.postings[token][note_id] = c
            self.doc_lengths[note_id] = len(tokens)
            count += 1

        self.total_docs = len(self.doc_lengths)
        self._update_avg()
        logger.info("Index rebuilt: %d documents, %d unique tokens", count, len(self.postings))
