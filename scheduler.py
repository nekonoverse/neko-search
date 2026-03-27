"""Automatic retraining scheduler for neko-search."""

import logging
import threading
import time

from generation import GenerationManager
from store import DocumentStore

logger = logging.getLogger(__name__)


class AutoTrainScheduler:
    """Periodically checks if retraining is needed and triggers it."""

    def __init__(
        self,
        gen_manager: GenerationManager,
        store: DocumentStore,
        train_fn,
        *,
        interval: int = 604800,
        min_new_docs: int = 1000,
    ):
        self._gen_manager = gen_manager
        self._store = store
        self._train_fn = train_fn
        self._interval = interval
        self._min_new_docs = min_new_docs
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._next_check_at: float = 0.0

    @property
    def next_check_at(self) -> float:
        return self._next_check_at

    def should_train(self) -> bool:
        """Check if retraining conditions are met."""
        if self._gen_manager.is_building:
            return False

        meta = self._gen_manager.load_meta()

        # No model trained yet — need initial training
        if meta is None:
            return self._store.doc_count >= self._min_new_docs

        # Check time elapsed since last training
        elapsed = time.time() - meta.created_at
        if elapsed < self._interval:
            return False

        # Check new document count
        new_docs = self._store.doc_count - meta.doc_count
        if new_docs < self._min_new_docs:
            return False

        return True

    def _run(self):
        """Background loop: sleep, check conditions, trigger training."""
        logger.info(
            "AutoTrainScheduler started (interval=%ds, min_new_docs=%d)",
            self._interval,
            self._min_new_docs,
        )
        while not self._stop_event.is_set():
            self._next_check_at = time.time() + self._interval
            if self._stop_event.wait(timeout=self._interval):
                break

            if self.should_train():
                logger.info("AutoTrainScheduler: conditions met, starting retraining")
                if self._gen_manager.start_build():
                    t = threading.Thread(
                        target=self._train_fn,
                        args=(8000,),
                        daemon=True,
                    )
                    t.start()

    def start(self):
        """Start the scheduler background thread."""
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the scheduler."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
