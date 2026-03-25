"""SentencePiece tokenizer wrapper."""

import logging
import os
import re

import sentencepiece as spm

logger = logging.getLogger(__name__)

# Pre-processing patterns
_URL_RE = re.compile(r"https?://\S+")
_MENTION_RE = re.compile(r"@[\w.-]+(?:@[\w.-]+)?")
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def preprocess(text: str) -> str:
    """Clean text for tokenization: remove URLs, mentions, HTML tags."""
    text = _HTML_TAG_RE.sub("", text)
    text = _URL_RE.sub("", text)
    text = _MENTION_RE.sub("", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


class Tokenizer:
    """SentencePiece Unigram LM tokenizer."""

    def __init__(self, model_path: str | None = None):
        self._sp: spm.SentencePieceProcessor | None = None
        if model_path and os.path.exists(model_path):
            self.load(model_path)

    @property
    def loaded(self) -> bool:
        return self._sp is not None

    @property
    def vocab_size(self) -> int:
        return self._sp.get_piece_size() if self._sp else 0

    def load(self, model_path: str) -> None:
        self._sp = spm.SentencePieceProcessor()
        self._sp.load(model_path)
        logger.info("Loaded SentencePiece model: %s (vocab=%d)", model_path, self.vocab_size)

    def tokenize(self, text: str) -> list[str]:
        """Preprocess and tokenize text into subword pieces."""
        if not self._sp:
            raise RuntimeError("SentencePiece model not loaded")
        cleaned = preprocess(text)
        if not cleaned:
            return []
        return self._sp.encode_as_pieces(cleaned)

    @staticmethod
    def train(
        corpus_path: str,
        model_prefix: str,
        vocab_size: int = 8000,
    ) -> str:
        """Train a SentencePiece Unigram LM model from a corpus file.

        Returns the path to the trained .model file.
        """
        spm.SentencePieceTrainer.train(
            input=corpus_path,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type="unigram",
            character_coverage=0.9995,
            normalization_rule_name="nfkc",
            pad_id=3,
        )
        model_path = f"{model_prefix}.model"
        logger.info("Trained SentencePiece model: %s (vocab=%d)", model_path, vocab_size)
        return model_path
