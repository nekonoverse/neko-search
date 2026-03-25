"""Two-generation index management for zero-downtime vocabulary updates."""

import json
import logging
import os
import time
from dataclasses import asdict, dataclass

logger = logging.getLogger(__name__)


@dataclass
class GenerationMeta:
    version: int
    vocab_size: int
    doc_count: int
    created_at: float  # unix timestamp
    status: str  # "active", "building"


class GenerationManager:
    """Manages generation metadata and build state."""

    def __init__(self, data_dir: str):
        self._data_dir = data_dir
        self._meta_path = os.path.join(data_dir, "generation.json")
        self._current_version: int = 0
        self._building: bool = False
        self._build_error: str | None = None

    @property
    def current_version(self) -> int:
        return self._current_version

    @property
    def is_building(self) -> bool:
        return self._building

    @property
    def build_error(self) -> str | None:
        return self._build_error

    def load_meta(self) -> GenerationMeta | None:
        """Load generation metadata from disk."""
        if not os.path.exists(self._meta_path):
            return None
        with open(self._meta_path) as f:
            data = json.load(f)
        self._current_version = data.get("version", 0)
        return GenerationMeta(**data)

    def save_meta(self, meta: GenerationMeta) -> None:
        """Save generation metadata to disk."""
        with open(self._meta_path, "w") as f:
            json.dump(asdict(meta), f, indent=2)
        self._current_version = meta.version

    def start_build(self) -> bool:
        """Start a build. Returns False if already building."""
        if self._building:
            return False
        self._building = True
        self._build_error = None
        return True

    def finish_build(self, error: str | None = None) -> None:
        """Mark the build as complete."""
        self._building = False
        self._build_error = error

    def new_version(self) -> int:
        """Return the next version number."""
        return self._current_version + 1

    def create_active_meta(self, version: int, vocab_size: int, doc_count: int) -> GenerationMeta:
        """Create a new active generation metadata."""
        return GenerationMeta(
            version=version,
            vocab_size=vocab_size,
            doc_count=doc_count,
            created_at=time.time(),
            status="active",
        )
