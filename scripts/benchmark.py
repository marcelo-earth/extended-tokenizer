#!/usr/bin/env python3
"""Benchmark ExtendedTokenizer against tiktoken."""

import argparse
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_benchmark_texts() -> list[str]:
    """Load sample texts for benchmarking."""
    texts = [
        # Short texts
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "Python is a programming language.",

        # Technical text
        """
        Machine learning is a subset of artificial intelligence (AI) that provides
        systems the ability to automatically learn and improve from experience without
        being explicitly programmed. Machine learning focuses on the development of
        computer programs that can access data and use it to learn for themselves.
        """,

        # Wikipedia-style text
        """
        The Python programming language was created by Guido van Rossum and first
        released in 1991. Python's design philosophy emphasizes code readability
        with its notable use of significant whitespace. Its language constructs and
        object-oriented approach aim to help programmers write clear, logical code
        for small and large-scale projects. Python is dynamically typed and
        garbage-collected. It supports multiple programming paradigms, including
        structured, object-oriented, and functional programming.
        """,

        # Code-like text
        """
        def fibonacci(n):
            if n <= 1:
                return n
            return fibonacci(n-1) + fibonacci(n-2)

        for i in range(10):
            print(f"fib({i}) = {fibonacci(i)}")
        """,

        # Unicode text
        "こんにちは世界 🌍 Привет мир مرحبا بالعالم",

        # Numbers and special characters
        "The meeting is at 3:30 PM on 2024-01-15. Contact: user@example.com",
    ]

    return texts


def benchmark_tokenizer(
    tokenizer,
    texts: list[str],
    name: str,
    iterations: int = 100,
) -> dict:
    """Benchmark a tokenizer.

    Returns dict with timing and token count statistics.
    """
    # Warmup
    for text in texts[:3]:
        tokenizer.encode(text)

    # Timing
    total_tokens = 0
    total_chars = 0

    start_time = time.perf_counter()

    for _ in range(iterations):
        for text in texts:
            tokens = tokenizer.encode(text)
            total_tokens += len(tokens)
            total_chars += len(text)

    end_time = time.perf_counter()
    elapsed = end_time - start_time

    # Roundtrip test
    roundtrip_errors = 0
    for text in texts:
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        if decoded != text:
            roundtrip_errors += 1

    return {
        "name": name,
        "total_time": elapsed,
        "iterations": iterations,
        "total_tokens": total_tokens,
        "total_chars": total_chars,
        "tokens_per_char": total_tokens / total_chars,
        "chars_per_token": total_chars / total_tokens,
        "throughput_chars": total_chars / elapsed,
        "roundtrip_errors": roundtrip_errors,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark ExtendedTokenizer against tiktoken"
    )
    parser.add_argument(
        "--vocab-path",
        type=str,
        default=None,
        help="Path to vocabulary (if not using HuggingFace)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of benchmark iterations",
    )
    parser.add_argument(
        "--no-tiktoken",
        action="store_true",
        help="Skip tiktoken comparison",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Tokenizer Benchmark")
    print("=" * 60)
    print()

    texts = load_benchmark_texts()
    print(f"Benchmark texts: {len(texts)}")
    print(f"Total characters: {sum(len(t) for t in texts):,}")
    print(f"Iterations: {args.iterations}")
    print()

    results = []

    # Benchmark ExtendedTokenizer
    print("Loading ExtendedTokenizer...")
    try:
        from extended_tokenizer import ExtendedTokenizer

        if args.vocab_path:
            tokenizer = ExtendedTokenizer(vocab_path=args.vocab_path)
        else:
            print("  (No vocab path specified, will try to load from HuggingFace)")
            tokenizer = ExtendedTokenizer.from_pretrained()

        print(f"  Vocab size: {tokenizer.vocab_size:,}")
        print()

        print("Benchmarking ExtendedTokenizer...")
        result = benchmark_tokenizer(
            tokenizer, texts, "ExtendedTokenizer", args.iterations
        )
        results.append(result)
        print(f"  Done: {result['throughput_chars']:.0f} chars/sec")

    except Exception as e:
        print(f"  Error loading ExtendedTokenizer: {e}")
        print("  Skipping ExtendedTokenizer benchmark")

    # Benchmark tiktoken
    if not args.no_tiktoken:
        print()
        print("Loading tiktoken (cl100k_base)...")
        try:
            import tiktoken

            enc = tiktoken.get_encoding("cl100k_base")
            print(f"  Vocab size: {enc.n_vocab:,}")
            print()

            # Create wrapper for consistent interface
            class TiktokenWrapper:
                def __init__(self, enc):
                    self.enc = enc

                def encode(self, text):
                    return self.enc.encode(text)

                def decode(self, tokens):
                    return self.enc.decode(tokens)

            print("Benchmarking tiktoken...")
            result = benchmark_tokenizer(
                TiktokenWrapper(enc), texts, "tiktoken (cl100k)", args.iterations
            )
            results.append(result)
            print(f"  Done: {result['throughput_chars']:.0f} chars/sec")

        except ImportError:
            print("  tiktoken not installed. Install with: pip install tiktoken")
        except Exception as e:
            print(f"  Error: {e}")

    # Print results
    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print()

    if not results:
        print("No tokenizers benchmarked successfully.")
        sys.exit(1)

    # Header
    print(f"{'Tokenizer':<25} {'Tokens/Char':<12} {'Chars/Token':<12} {'Chars/Sec':<15} {'Errors':<8}")
    print("-" * 72)

    for r in results:
        print(
            f"{r['name']:<25} "
            f"{r['tokens_per_char']:<12.4f} "
            f"{r['chars_per_token']:<12.2f} "
            f"{r['throughput_chars']:<15,.0f} "
            f"{r['roundtrip_errors']:<8}"
        )

    # Comparison
    if len(results) >= 2:
        print()
        print("-" * 72)
        ext_result = next((r for r in results if "Extended" in r["name"]), None)
        tik_result = next((r for r in results if "tiktoken" in r["name"]), None)

        if ext_result and tik_result:
            efficiency = tik_result["total_tokens"] / ext_result["total_tokens"]
            speed_ratio = ext_result["throughput_chars"] / tik_result["throughput_chars"]

            print()
            print("Comparison (ExtendedTokenizer vs tiktoken):")
            print(f"  Token efficiency: {efficiency:.2%}")
            print("    (< 100% = ExtendedTokenizer uses more tokens)")
            print("    (> 100% = ExtendedTokenizer uses fewer tokens)")
            print()
            print(f"  Speed ratio: {speed_ratio:.2f}x")
            print("    (> 1.0 = ExtendedTokenizer is faster)")


if __name__ == "__main__":
    main()
