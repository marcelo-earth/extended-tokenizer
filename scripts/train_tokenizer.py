#!/usr/bin/env python3
"""Train the BPE tokenizer on Wikipedia data."""

import argparse
import sys
from collections.abc import Iterator
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from extended_tokenizer.data.wikipedia import stream_wikipedia_texts
from extended_tokenizer.trainer import BPETrainer


def stream_text_files(input_path: Path) -> Iterator[str]:
    """Stream text from files in a directory."""
    if input_path.is_file():
        # Single file
        if input_path.suffix == ".bz2":
            # Wikipedia dump
            yield from stream_wikipedia_texts(str(input_path))
        else:
            # Plain text file
            with open(input_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield line
    else:
        # Directory of text files
        for txt_file in sorted(input_path.glob("**/*.txt")):
            with open(txt_file, encoding="utf-8") as f:
                text = f.read()
                if text.strip():
                    yield text


def main():
    parser = argparse.ArgumentParser(
        description="Train BPE tokenizer on text corpus"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Input directory with text files or Wikipedia dump (.xml.bz2)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="vocab/bpe_300k",
        help="Output directory for vocabulary files",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=300_000,
        help="Target vocabulary size (default: 300000)",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Minimum pair frequency for merging (default: 2)",
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=None,
        help="Maximum number of articles to process (for testing)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars",
    )

    args = parser.parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        sys.exit(1)

    print("=" * 60)
    print("BPE Tokenizer Training")
    print("=" * 60)
    print()
    print(f"Input: {input_path}")
    print(f"Output: {args.output}")
    print(f"Target vocab size: {args.vocab_size:,}")
    print(f"Min frequency: {args.min_frequency}")
    print()

    # Initialize trainer
    trainer = BPETrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )

    # Stream corpus
    if input_path.suffix == ".bz2":
        print("Detected Wikipedia dump file")
        corpus = stream_wikipedia_texts(
            str(input_path),
            max_articles=args.max_articles,
            show_progress=not args.no_progress,
        )
    else:
        print(f"Reading text files from {input_path}")
        corpus = stream_text_files(input_path)

    # Train
    print()
    print("Starting BPE training...")
    print("This may take several hours for large vocabularies.")
    print()

    try:
        vocab = trainer.train(corpus, show_progress=not args.no_progress)

        print()
        print("Training complete!")
        print(f"Final vocabulary size: {vocab.vocab_size:,}")
        print(f"Number of merges: {vocab.num_merges:,}")

        # Save vocabulary
        print()
        print(f"Saving vocabulary to {args.output}...")
        trainer.save(args.output)

        print()
        print("Done!")
        print()
        print("Next steps:")
        print("  1. Test the tokenizer:")
        print("     python -c \"from extended_tokenizer import ExtendedTokenizer; t = ExtendedTokenizer('vocab/bpe_300k'); print(t.encode('Hello!'))\"")
        print()
        print("  2. Run benchmarks:")
        print("     python scripts/benchmark.py --vocab-path vocab/bpe_300k")
        print()
        print("  3. Upload to HuggingFace Hub:")
        print("     huggingface-cli upload marcelo-earth/extended-tokenizer-300k vocab/bpe_300k")

    except KeyboardInterrupt:
        print("\nTraining cancelled.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during training: {e}")
        raise


if __name__ == "__main__":
    main()
