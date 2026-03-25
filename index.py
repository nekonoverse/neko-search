"""In-memory inverted index with BM25 scoring and positional phrase search."""

import logging
import math
from collections import defaultdict

logger = logging.getLogger(__name__)

# BM25 parameters
K1 = 1.2
B = 0.75


class InvertedIndex:
    """Subword-based inverted index with BM25 scoring and phrase search."""

    def __init__(self):
        # subword -> {note_id: [position, ...]}
        self.postings: dict[str, dict[str, list[int]]] = defaultdict(dict)
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

        # Build position lists
        positions: dict[str, list[int]] = defaultdict(list)
        for i, token in enumerate(tokens):
            positions[token].append(i)

        # Add to postings
        for token, pos_list in positions.items():
            self.postings[token][note_id] = pos_list

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

    def search(
        self, query_tokens: list[str], limit: int = 20, phrase: bool = False
    ) -> list[tuple[str, float]]:
        """Search with BM25 scoring (AND mode).

        Only documents containing ALL query tokens are returned.
        If phrase=True, tokens must appear consecutively in document order.
        Returns list of (note_id, score) sorted by score descending.
        """
        if not query_tokens or self.total_docs == 0:
            return []

        # Deduplicate for AND intersection (preserve order for phrase check)
        unique_tokens = list(dict.fromkeys(query_tokens))

        # Collect postings for each token; if any token has no postings, no results
        token_postings = []
        for token in unique_tokens:
            posting = self.postings.get(token)
            if not posting:
                return []
            token_postings.append((token, posting))

        # AND: intersect document sets (start from smallest posting list)
        sorted_by_size = sorted(token_postings, key=lambda x: len(x[1]))
        candidates = set(sorted_by_size[0][1].keys())
        for _, posting in sorted_by_size[1:]:
            candidates &= posting.keys()
            if not candidates:
                return []

        # Phrase filter: check consecutive positions using original query order
        if phrase and len(query_tokens) > 1:
            phrase_candidates = set()
            for note_id in candidates:
                if self._check_phrase(note_id, query_tokens):
                    phrase_candidates.add(note_id)
            candidates = phrase_candidates
            if not candidates:
                return []

        # BM25 scoring on candidates only
        scores: dict[str, float] = {}
        avgdl = self.avg_doc_length or 1.0

        for note_id in candidates:
            score = 0.0
            dl = self.doc_lengths.get(note_id, 0)
            for token, posting in token_postings:
                df = len(posting)
                idf = math.log((self.total_docs - df + 0.5) / (df + 0.5) + 1.0)
                tf = len(posting[note_id])
                tf_norm = (tf * (K1 + 1)) / (tf + K1 * (1 - B + B * dl / avgdl))
                score += idf * tf_norm
            scores[note_id] = score

        # Sort by score descending, take top N
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:limit]

    def _check_phrase(self, note_id: str, query_tokens: list[str]) -> bool:
        """Check if query_tokens appear consecutively in the document."""
        # Get positions for the first token
        first_posting = self.postings.get(query_tokens[0])
        if not first_posting or note_id not in first_posting:
            return False

        # Build position sets for O(1) lookup
        pos_sets: list[set[int]] = []
        for token in query_tokens[1:]:
            posting = self.postings.get(token)
            if not posting or note_id not in posting:
                return False
            pos_sets.append(set(posting[note_id]))

        # Check if any start position leads to a consecutive sequence
        for start in first_posting[note_id]:
            if all((start + offset + 1) in ps for offset, ps in enumerate(pos_sets)):
                return True
        return False

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
            positions: dict[str, list[int]] = defaultdict(list)
            for i, token in enumerate(tokens):
                positions[token].append(i)
            for token, pos_list in positions.items():
                self.postings[token][note_id] = pos_list
            self.doc_lengths[note_id] = len(tokens)
            count += 1

        self.total_docs = len(self.doc_lengths)
        self._update_avg()
        logger.info("Index rebuilt: %d documents, %d unique tokens", count, len(self.postings))
