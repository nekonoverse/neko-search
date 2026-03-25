"""Prefix-based suggestion from inverted index vocabulary."""

import bisect
import logging

logger = logging.getLogger(__name__)


class SuggestIndex:
    """Sorted-list based prefix lookup for subword tokens."""

    def __init__(self):
        self._entries: list[tuple[str, int]] = []  # (token, df) sorted by token
        self._keys: list[str] = []  # token strings only, for bisect
        self._postings_ref: dict[str, dict[str, int]] | None = None
        self._dirty = False

    def rebuild(self, postings: dict[str, dict[str, int]]) -> None:
        """Rebuild from the inverted index postings."""
        entries = sorted(
            ((token, len(docs)) for token, docs in postings.items()),
            key=lambda x: x[0],
        )
        self._entries = entries
        self._keys = [e[0] for e in entries]
        self._dirty = False
        logger.info("SuggestIndex rebuilt: %d tokens", len(entries))

    def set_postings_ref(self, postings: dict[str, dict[str, int]]) -> None:
        """Set a reference to the live postings dict for lazy rebuilds."""
        self._postings_ref = postings
        self._dirty = True

    def mark_dirty(self) -> None:
        """Mark the index as needing a rebuild on next query."""
        self._dirty = True

    def _ensure_fresh(self) -> None:
        if self._dirty and self._postings_ref is not None:
            self.rebuild(self._postings_ref)

    def prefix_search(self, prefix: str, limit: int = 10) -> list[dict]:
        """Find tokens starting with prefix, ranked by document frequency.

        Returns list of {"token": str, "df": int}.
        """
        self._ensure_fresh()

        if not prefix or not self._keys:
            return []

        start = bisect.bisect_left(self._keys, prefix)
        candidates = []
        for i in range(start, len(self._keys)):
            token = self._keys[i]
            if not token.startswith(prefix):
                break
            candidates.append({"token": token, "df": self._entries[i][1]})

        candidates.sort(key=lambda s: s["df"], reverse=True)
        return candidates[:limit]
