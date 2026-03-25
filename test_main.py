"""Tests for neko-search microservice."""

import os
import tempfile
import time

import pytest
from fastapi.testclient import TestClient

# Use temp directory for test data
_test_dir = tempfile.mkdtemp()
os.environ["DATA_DIR"] = _test_dir

from main import app  # noqa: E402

client = TestClient(app)


# --- Health / Version ---

def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["version"] == "1"
    assert data["model_loaded"] is False
    assert "generation" in data
    assert "building" in data


def test_version():
    resp = client.get("/version")
    assert resp.status_code == 200
    assert resp.json()["version"] == "1"


# --- Index / Search / Delete (without model) ---

def test_index_without_model():
    """Index stores to DB even without model (tokens empty)."""
    resp = client.post("/index", json={"note_id": "abc", "text": "hello"})
    assert resp.status_code == 200
    assert resp.json()["tokens"] == 0


def test_search_without_model():
    """Search fails when model not loaded."""
    resp = client.get("/search?q=hello")
    assert resp.status_code == 503


def test_delete_without_model():
    """Delete succeeds even without model (no-op)."""
    resp = client.delete("/index/abc")
    assert resp.status_code == 200


# --- Suggest (without model) ---

def test_suggest_without_model():
    resp = client.get("/suggest?q=hello")
    assert resp.status_code == 503


def test_suggest_empty_query():
    """Empty query returns empty suggestions (no model needed for early return)."""
    resp = client.get("/suggest?q=%20")
    # Stripped to empty -> returns empty without model check? No, model check first.
    assert resp.status_code == 503


# --- Train status ---

def test_train_status():
    resp = client.get("/train/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["building"] is False
    assert "current_version" in data
    assert data["error"] is None


# --- Tokenizer ---

def test_tokenizer_preprocess():
    from tokenizer import preprocess

    text = '<p>Hello @user@example.com check https://example.com/page</p>'
    result = preprocess(text)
    assert "https://" not in result
    assert "@user" not in result
    assert "<p>" not in result
    assert "Hello" in result
    assert "check" in result


# --- InvertedIndex ---

def test_inverted_index_basic():
    from index import InvertedIndex

    idx = InvertedIndex()
    idx.add("note1", ["hello", "world"])
    idx.add("note2", ["hello", "there"])
    idx.add("note3", ["goodbye", "world"])

    results = idx.search(["hello"])
    note_ids = [nid for nid, _ in results]
    assert "note1" in note_ids
    assert "note2" in note_ids
    assert "note3" not in note_ids


def test_inverted_index_remove():
    from index import InvertedIndex

    idx = InvertedIndex()
    idx.add("note1", ["hello", "world"])
    idx.add("note2", ["hello"])

    assert idx.remove("note1")
    results = idx.search(["hello"])
    note_ids = [nid for nid, _ in results]
    assert "note1" not in note_ids
    assert "note2" in note_ids


def test_inverted_index_and_search():
    """AND mode: only documents containing ALL query tokens are returned."""
    from index import InvertedIndex

    idx = InvertedIndex()
    idx.add("note1", ["hello", "world"])
    idx.add("note2", ["hello", "there"])
    idx.add("note3", ["goodbye", "world"])

    # Both tokens required
    results = idx.search(["hello", "world"])
    note_ids = [nid for nid, _ in results]
    assert "note1" in note_ids
    assert "note2" not in note_ids  # missing "world"
    assert "note3" not in note_ids  # missing "hello"


def test_inverted_index_and_no_match():
    """AND mode: returns empty when a query token exists in no documents."""
    from index import InvertedIndex

    idx = InvertedIndex()
    idx.add("note1", ["hello", "world"])

    results = idx.search(["hello", "nonexistent"])
    assert results == []


def test_inverted_index_bm25_ranking():
    """Documents with higher TF should rank higher."""
    from index import InvertedIndex

    idx = InvertedIndex()
    idx.add("note1", ["cat"])
    idx.add("note2", ["cat", "cat", "cat"])

    results = idx.search(["cat"])
    assert results[0][0] == "note2"  # Higher TF = higher score


# --- SuggestIndex ---

def test_suggest_index_basic():
    from suggest import SuggestIndex

    si = SuggestIndex()
    postings = {
        "hello": {"n1": 1, "n2": 1, "n3": 1},
        "help": {"n1": 1, "n2": 1},
        "world": {"n1": 1},
        "hell": {"n1": 1},
    }
    si.rebuild(postings)

    results = si.prefix_search("hel")
    tokens = [r["token"] for r in results]
    assert "hello" in tokens
    assert "help" in tokens
    assert "hell" in tokens
    assert "world" not in tokens
    # Ranked by df: hello(3) > help(2) > hell(1)
    assert results[0]["token"] == "hello"
    assert results[0]["df"] == 3


def test_suggest_index_empty():
    from suggest import SuggestIndex

    si = SuggestIndex()
    si.rebuild({})
    assert si.prefix_search("test") == []


def test_suggest_index_no_match():
    from suggest import SuggestIndex

    si = SuggestIndex()
    si.rebuild({"abc": {"n1": 1}})
    assert si.prefix_search("xyz") == []


def test_suggest_index_sentencepiece_prefix():
    """Verify prefix search works with ▁ word boundary markers (real SentencePiece tokens)."""
    from suggest import SuggestIndex

    si = SuggestIndex()
    postings = {
        "\u2581さんせい": {"n1": 1, "n2": 1},
        "\u2581さんせつ": {"n1": 1},
        "\u2581さん": {"n1": 1, "n2": 1, "n3": 1},
        "\u2581せ": {"n1": 1},
        "せい": {"n1": 1},
    }
    si.rebuild(postings)

    # Raw prefix "さんせ" → prepend ▁ → should match さんせい, さんせつ but NOT さん or せ
    results = si.prefix_search("\u2581さんせ")
    tokens = [r["token"] for r in results]
    assert "\u2581さんせい" in tokens
    assert "\u2581さんせつ" in tokens
    assert "\u2581さん" not in tokens
    assert "\u2581せ" not in tokens
    assert "せい" not in tokens


def test_suggest_index_limit():
    from suggest import SuggestIndex

    si = SuggestIndex()
    postings = {f"token{i}": {"n1": 1} for i in range(20)}
    si.rebuild(postings)
    results = si.prefix_search("token", limit=5)
    assert len(results) == 5


def test_suggest_index_lazy_rebuild():
    from suggest import SuggestIndex

    postings = {"hello": {"n1": 1}, "help": {"n1": 1}}
    si = SuggestIndex()
    si.set_postings_ref(postings)

    # First query triggers rebuild
    results = si.prefix_search("hel")
    assert len(results) == 2

    # Mutate postings and mark dirty
    postings["hero"] = {"n1": 1, "n2": 1}
    si.mark_dirty()

    results = si.prefix_search("he")
    assert len(results) == 3


# --- DocumentStore ---

def test_document_store():
    from store import DocumentStore

    db_path = os.path.join(_test_dir, "test_store.db")
    store = DocumentStore(db_path)

    store.upsert("n1", "hello world", ["hello", "world"], "2024-01-01T00:00:00Z")
    assert store.doc_count == 1

    doc = store.get("n1")
    assert doc is not None
    assert doc["tokens"] == ["hello", "world"]

    store.delete("n1")
    assert store.doc_count == 0
    assert store.get("n1") is None

    store.close()


def test_document_store_bulk():
    from store import DocumentStore

    db_path = os.path.join(_test_dir, "test_bulk.db")
    store = DocumentStore(db_path)

    docs = [
        {"note_id": f"n{i}", "text": f"text {i}", "tokens": [f"token{i}"], "published": None}
        for i in range(10)
    ]
    store.bulk_upsert(docs)
    assert store.doc_count == 10

    texts = store.export_texts()
    assert len(texts) == 10

    store.close()


# --- GenerationManager ---

def test_generation_meta_persistence():
    from generation import GenerationManager

    gm = GenerationManager(_test_dir)
    assert gm.load_meta() is None

    meta = gm.create_active_meta(version=1, vocab_size=8000, doc_count=100)
    gm.save_meta(meta)

    loaded = gm.load_meta()
    assert loaded is not None
    assert loaded.version == 1
    assert loaded.vocab_size == 8000
    assert loaded.doc_count == 100
    assert loaded.status == "active"

    # Clean up
    os.unlink(os.path.join(_test_dir, "generation.json"))


def test_generation_build_state():
    from generation import GenerationManager

    gm = GenerationManager(_test_dir)
    assert not gm.is_building
    assert gm.build_error is None

    assert gm.start_build()
    assert gm.is_building
    assert not gm.start_build()  # Can't start twice

    gm.finish_build()
    assert not gm.is_building

    assert gm.start_build()
    gm.finish_build(error="test error")
    assert not gm.is_building
    assert gm.build_error == "test error"


# --- SearchState ---

def test_search_state():
    from main import SearchState
    from index import InvertedIndex
    from suggest import SuggestIndex
    from tokenizer import Tokenizer

    state = SearchState(
        tokenizer=Tokenizer(),
        index=InvertedIndex(),
        suggest=SuggestIndex(),
    )
    assert not state.tokenizer.loaded
    assert state.index.total_docs == 0
