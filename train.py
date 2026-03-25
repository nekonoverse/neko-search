#!/usr/bin/env python3
"""Standalone SentencePiece training script.

Usage:
    python train.py --input corpus.txt --vocab-size 8000 --model-prefix /data/sp

The corpus file should contain one document per line (plain text, pre-cleaned).
"""

import argparse
import sys

from tokenizer import Tokenizer, preprocess


def main():
    parser = argparse.ArgumentParser(description="Train SentencePiece model for neko-search")
    parser.add_argument("--input", required=True, help="Path to corpus text file (one doc per line)")
    parser.add_argument("--vocab-size", type=int, default=8000, help="Vocabulary size (default: 8000)")
    parser.add_argument("--model-prefix", default="/data/sp", help="Output model path prefix")
    parser.add_argument("--show-samples", type=int, default=10,
                        help="Number of sample tokenizations to display")
    args = parser.parse_args()

    # Pre-process corpus
    print(f"Pre-processing corpus: {args.input}")
    cleaned_lines = []
    with open(args.input) as f:
        for line in f:
            cleaned = preprocess(line.strip())
            if cleaned:
                cleaned_lines.append(cleaned)

    if not cleaned_lines:
        print("Error: No valid lines in corpus after preprocessing", file=sys.stderr)
        sys.exit(1)

    print(f"  Lines: {len(cleaned_lines)}")

    # Write cleaned corpus to temp file
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
        for line in cleaned_lines:
            tmp.write(line + "\n")
        cleaned_path = tmp.name

    # Train
    print(f"Training SentencePiece (vocab_size={args.vocab_size})...")
    try:
        model_path = Tokenizer.train(cleaned_path, args.model_prefix, args.vocab_size)
    finally:
        import os
        os.unlink(cleaned_path)

    print(f"Model saved: {model_path}")

    # Show sample tokenizations
    if args.show_samples > 0:
        tok = Tokenizer(model_path)
        print(f"\nSample tokenizations (vocab={tok.vocab_size}):")
        import random
        samples = random.sample(cleaned_lines, min(args.show_samples, len(cleaned_lines)))
        for text in samples:
            pieces = tok.tokenize(text)
            print(f"  {text[:80]}")
            print(f"    -> {' | '.join(pieces[:30])}")
            print()


if __name__ == "__main__":
    main()
