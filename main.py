"""neko-search: SentencePiece + BM25 search microservice for nekonoverse."""

import logging
import os
import shutil
import tempfile
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from generation import GenerationManager
from index import InvertedIndex
from store import DocumentStore
from suggest import SuggestIndex
from tokenizer import Tokenizer, preprocess

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

VERSION = "1"
DATA_DIR = os.environ.get("DATA_DIR", "/data")
MODEL_PATH = os.path.join(DATA_DIR, "sp.model")
DB_PATH = os.path.join(DATA_DIR, "documents.db")


@dataclass
class SearchState:
    """Holds all search-related state for atomic swapping."""
    tokenizer: Tokenizer
    index: InvertedIndex
    suggest: SuggestIndex


# Module-level state — single reference swap is atomic under CPython GIL
_state = SearchState(
    tokenizer=Tokenizer(),
    index=InvertedIndex(),
    suggest=SuggestIndex(),
)
store: DocumentStore | None = None
gen_manager = GenerationManager(DATA_DIR)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global store, _state
    os.makedirs(DATA_DIR, exist_ok=True)
    store = DocumentStore(DB_PATH)

    # Load generation metadata
    gen_manager.load_meta()

    # Load SentencePiece model if available
    if os.path.exists(MODEL_PATH):
        _state.tokenizer.load(MODEL_PATH)
        logger.info("Rebuilding index from store...")
        _state.index.rebuild(store.iter_all())
        _state.suggest.rebuild(_state.index.postings)
        _state.suggest.set_postings_ref(_state.index.postings)
        logger.info("Index ready: %d docs", _state.index.total_docs)
    else:
        logger.warning(
            "No SentencePiece model at %s — /index and /search require training first",
            MODEL_PATH,
        )

    yield

    if store:
        store.close()


app = FastAPI(title="neko-search", version=VERSION, lifespan=lifespan)


# --- Request/Response models ---

class IndexRequest(BaseModel):
    note_id: str
    text: str
    published: str | None = None


class BulkIndexRequest(BaseModel):
    notes: list[IndexRequest]


class SearchResponse(BaseModel):
    note_ids: list[str]
    total: int


class TrainRequest(BaseModel):
    vocab_size: int = Field(default=8000, ge=100, le=100000)


# --- Endpoints ---

@app.get("/health")
async def health():
    state = _state
    return {
        "status": "ok",
        "version": VERSION,
        "doc_count": state.index.total_docs,
        "vocab_size": state.tokenizer.vocab_size,
        "model_loaded": state.tokenizer.loaded,
        "generation": gen_manager.current_version,
        "building": gen_manager.is_building,
    }


@app.get("/version")
async def version():
    return {"version": VERSION}


@app.post("/index")
async def index_note(req: IndexRequest):
    state = _state
    if state.tokenizer.loaded:
        tokens = state.tokenizer.tokenize(req.text)
        if store:
            store.upsert(req.note_id, req.text, tokens, req.published)
        state.index.add(req.note_id, tokens)
        state.suggest.mark_dirty()
    else:
        if store:
            store.upsert(req.note_id, req.text, [], req.published)
    return {"ok": True, "tokens": len(tokens) if state.tokenizer.loaded else 0}


@app.post("/bulk-index")
async def bulk_index(req: BulkIndexRequest):
    state = _state
    model_loaded = state.tokenizer.loaded

    docs = []
    for note in req.notes:
        if model_loaded:
            tokens = state.tokenizer.tokenize(note.text)
        else:
            tokens = []
        docs.append({
            "note_id": note.note_id,
            "text": note.text,
            "tokens": tokens,
            "published": note.published,
        })
        if model_loaded:
            state.index.add(note.note_id, tokens)

    if store:
        store.bulk_upsert(docs)
    if model_loaded:
        state.suggest.mark_dirty()
    return {"ok": True, "indexed": len(docs)}


@app.delete("/index/{note_id}")
async def delete_note(note_id: str):
    state = _state
    state.index.remove(note_id)
    state.suggest.mark_dirty()
    if store:
        store.delete(note_id)
    return {"ok": True}


@app.get("/search", response_model=SearchResponse)
async def search(q: str, limit: int = 20):
    state = _state
    if not state.tokenizer.loaded:
        raise HTTPException(status_code=503, detail="SentencePiece model not loaded")

    limit = min(max(1, limit), 100)
    query_tokens = state.tokenizer.tokenize(q)
    if not query_tokens:
        return SearchResponse(note_ids=[], total=0)

    # No whitespace in cleaned query → phrase search (tokens must be consecutive)
    cleaned = preprocess(q)
    use_phrase = " " not in cleaned

    results = state.index.search(query_tokens, limit=limit, phrase=use_phrase)
    note_ids = [nid for nid, _score in results]
    return SearchResponse(note_ids=note_ids, total=len(note_ids))


@app.get("/suggest")
async def suggest(q: str, limit: int = 10):
    state = _state
    if not state.tokenizer.loaded:
        raise HTTPException(status_code=503, detail="SentencePiece model not loaded")

    limit = min(max(1, limit), 50)
    q = q.strip()
    if not q:
        return {"suggestions": [], "prefix": ""}

    cleaned = preprocess(q)
    if not cleaned:
        return {"suggestions": [], "prefix": ""}

    # Use raw text prefix instead of SentencePiece tokenization.
    # Japanese has no spaces, so the whole input becomes the prefix.
    # For spaced languages, use the last whitespace-separated segment.
    parts = cleaned.split()
    raw_prefix = parts[-1] if parts else cleaned
    # SentencePiece tokens start with ▁ (word boundary marker)
    prefix = "\u2581" + raw_prefix

    results = state.suggest.prefix_search(prefix, limit=limit)
    return {"suggestions": results, "prefix": prefix}


@app.post("/train")
async def train(req: TrainRequest = TrainRequest()):
    """Start background retraining. Returns immediately."""
    if not store or store.doc_count == 0:
        raise HTTPException(status_code=400, detail="No documents in store")

    if not gen_manager.start_build():
        raise HTTPException(status_code=409, detail="Build already in progress")

    t = threading.Thread(
        target=_background_train,
        args=(req.vocab_size,),
        daemon=True,
    )
    t.start()

    return {
        "ok": True,
        "status": "building",
        "current_version": gen_manager.current_version,
    }


@app.get("/train/status")
async def train_status():
    return {
        "building": gen_manager.is_building,
        "current_version": gen_manager.current_version,
        "error": gen_manager.build_error,
    }


def _background_train(vocab_size: int):
    """Background thread: train new model, retokenize, rebuild index, swap."""
    global _state
    new_version = gen_manager.new_version()

    try:
        logger.info("Generation %d: starting build (vocab_size=%d)", new_version, vocab_size)

        # 1. Export corpus
        texts = store.export_texts()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            for text in texts:
                cleaned = preprocess(text)
                if cleaned:
                    f.write(cleaned + "\n")
            corpus_path = f.name

        # 2. Train new model
        new_model_prefix = os.path.join(DATA_DIR, f"sp_gen{new_version}")
        try:
            Tokenizer.train(corpus_path, new_model_prefix, vocab_size=vocab_size)
        finally:
            os.unlink(corpus_path)

        new_model_path = f"{new_model_prefix}.model"

        # 3. Load new tokenizer
        new_tokenizer = Tokenizer(new_model_path)

        # 4. Re-tokenize all documents
        logger.info("Generation %d: re-tokenizing %d documents...", new_version, store.doc_count)
        retokenized = []
        for doc in store.iter_all():
            new_tokens = new_tokenizer.tokenize(doc["text"])
            retokenized.append({
                "note_id": doc["note_id"],
                "text": doc["text"],
                "tokens": new_tokens,
                "published": doc["published"],
            })

        # 5. Build new index + suggest
        new_index = InvertedIndex()
        new_index.rebuild(iter(retokenized))

        new_suggest = SuggestIndex()
        new_suggest.rebuild(new_index.postings)
        new_suggest.set_postings_ref(new_index.postings)

        # 6. Update store with new tokens
        store.bulk_upsert(retokenized)

        # 7. Atomic swap
        _state = SearchState(
            tokenizer=new_tokenizer,
            index=new_index,
            suggest=new_suggest,
        )

        # 8. Copy model to canonical path and clean up temp files
        shutil.copy2(new_model_path, MODEL_PATH)
        for ext in (".model", ".vocab"):
            p = f"{new_model_prefix}{ext}"
            if os.path.exists(p):
                os.unlink(p)

        # 9. Save generation metadata
        meta = gen_manager.create_active_meta(
            version=new_version,
            vocab_size=new_tokenizer.vocab_size,
            doc_count=new_index.total_docs,
        )
        gen_manager.save_meta(meta)

        logger.info(
            "Generation %d: build complete (vocab=%d, docs=%d)",
            new_version, meta.vocab_size, meta.doc_count,
        )
        gen_manager.finish_build()

    except Exception as e:
        logger.exception("Generation %d: build failed", new_version)
        gen_manager.finish_build(error=str(e))
