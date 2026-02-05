"""Tests for the main tokenizer interface."""

import tempfile
from pathlib import Path

import pytest

from extended_tokenizer.tokenizer import ExtendedTokenizer
from extended_tokenizer.trainer import BPETrainer
from extended_tokenizer.vocab import SPECIAL_TOKENS


@pytest.fixture
def trained_tokenizer():
    """Create a small trained tokenizer for testing."""
    corpus = [
        "Hello world! ",
        "The quick brown fox jumps over the lazy dog. ",
        "Python is a programming language. ",
        "Machine learning is fun. ",
    ] * 50

    trainer = BPETrainer(vocab_size=300, min_frequency=2)
    trainer.train(corpus, show_progress=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "vocab"
        trainer.save(str(path))
        yield ExtendedTokenizer(vocab_path=path)


class TestExtendedTokenizer:
    """Tests for ExtendedTokenizer class."""

    def test_encode_empty(self, trained_tokenizer):
        """Test encoding empty string."""
        result = trained_tokenizer.encode("")
        assert result == []

    def test_encode_simple(self, trained_tokenizer):
        """Test encoding simple text."""
        result = trained_tokenizer.encode("Hello")
        assert len(result) > 0
        assert all(isinstance(t, int) for t in result)

    def test_decode_empty(self, trained_tokenizer):
        """Test decoding empty list."""
        result = trained_tokenizer.decode([])
        assert result == ""

    def test_roundtrip_ascii(self, trained_tokenizer):
        """Test encode/decode roundtrip with ASCII."""
        texts = [
            "Hello",
            "Hello world",
            "The quick brown fox",
            "Python",
        ]

        for text in texts:
            encoded = trained_tokenizer.encode(text)
            decoded = trained_tokenizer.decode(encoded)
            assert decoded == text, f"Roundtrip failed for: {text}"

    def test_roundtrip_unicode(self, trained_tokenizer):
        """Test encode/decode roundtrip with unicode."""
        texts = [
            "café",
            "日本語",
            "emoji 🎉",
            "Ümläut",
        ]

        for text in texts:
            encoded = trained_tokenizer.encode(text)
            decoded = trained_tokenizer.decode(encoded)
            assert decoded == text, f"Roundtrip failed for: {text}"

    def test_roundtrip_whitespace(self, trained_tokenizer):
        """Test roundtrip preserves whitespace."""
        texts = [
            "hello world",
            "hello  world",  # double space
            "hello\nworld",  # newline
            "hello\tworld",  # tab
        ]

        for text in texts:
            encoded = trained_tokenizer.encode(text)
            decoded = trained_tokenizer.decode(encoded)
            assert decoded == text, f"Roundtrip failed for: {repr(text)}"

    def test_special_tokens_with_flag(self, trained_tokenizer):
        """Test special token handling with add_special_tokens."""
        text = "Hello"

        without_special = trained_tokenizer.encode(text, add_special_tokens=False)
        with_special = trained_tokenizer.encode(text, add_special_tokens=True)

        # With special tokens should have 2 more tokens
        assert len(with_special) == len(without_special) + 2

        # First should be start token
        assert with_special[0] == SPECIAL_TOKENS["<|startoftext|>"]
        # Last should be end token
        assert with_special[-1] == SPECIAL_TOKENS["<|endoftext|>"]

    def test_decode_skips_special_by_default(self, trained_tokenizer):
        """Test that decode skips special tokens by default."""
        tokens = [
            SPECIAL_TOKENS["<|startoftext|>"],
            72,  # 'H'
            101,  # 'e'
            SPECIAL_TOKENS["<|endoftext|>"],
        ]

        decoded = trained_tokenizer.decode(tokens)
        assert "<|" not in decoded

    def test_decode_includes_special_when_flag_false(self, trained_tokenizer):
        """Test that decode includes special tokens when flag is False."""
        tokens = [
            SPECIAL_TOKENS["<|startoftext|>"],
            72,  # 'H'
            SPECIAL_TOKENS["<|endoftext|>"],
        ]

        decoded = trained_tokenizer.decode(tokens, skip_special_tokens=False)
        assert "<|startoftext|>" in decoded

    def test_vocab_size(self, trained_tokenizer):
        """Test vocab_size property."""
        # Should be around 300 (our training target) + special tokens
        assert trained_tokenizer.vocab_size > 256
        assert trained_tokenizer.vocab_size <= 310

    def test_special_token_ids(self, trained_tokenizer):
        """Test special token ID properties."""
        assert trained_tokenizer.eos_token_id == SPECIAL_TOKENS["<|endoftext|>"]
        assert trained_tokenizer.bos_token_id == SPECIAL_TOKENS["<|startoftext|>"]
        assert trained_tokenizer.pad_token_id == SPECIAL_TOKENS["<|padding|>"]

    def test_len(self, trained_tokenizer):
        """Test __len__ returns vocab_size."""
        assert len(trained_tokenizer) == trained_tokenizer.vocab_size

    def test_repr(self, trained_tokenizer):
        """Test string representation."""
        repr_str = repr(trained_tokenizer)
        assert "ExtendedTokenizer" in repr_str
        assert "vocab_size" in repr_str


class TestBatchEncoding:
    """Tests for batch encoding/decoding."""

    def test_encode_batch(self, trained_tokenizer):
        """Test batch encoding."""
        texts = ["Hello", "World", "Python"]
        results = trained_tokenizer.encode_batch(texts)

        assert len(results) == 3
        assert all(isinstance(r, list) for r in results)

        # Should match individual encoding
        for text, result in zip(texts, results):
            assert result == trained_tokenizer.encode(text)

    def test_encode_batch_parallel(self, trained_tokenizer):
        """Test parallel batch encoding."""
        texts = ["Hello world"] * 20
        results = trained_tokenizer.encode_batch(texts, num_workers=2)

        assert len(results) == 20
        # All should be identical
        assert all(r == results[0] for r in results)

    def test_decode_batch(self, trained_tokenizer):
        """Test batch decoding."""
        texts = ["Hello", "World", "Python"]
        encoded = trained_tokenizer.encode_batch(texts)
        decoded = trained_tokenizer.decode_batch(encoded)

        assert decoded == texts


class TestEdgeCases:
    """Tests for edge cases."""

    def test_very_long_text(self, trained_tokenizer):
        """Test encoding very long text."""
        text = "Hello world " * 1000
        encoded = trained_tokenizer.encode(text)
        decoded = trained_tokenizer.decode(encoded)
        assert decoded == text

    def test_single_character(self, trained_tokenizer):
        """Test single character encoding."""
        for char in "abcABC123!@#":
            encoded = trained_tokenizer.encode(char)
            decoded = trained_tokenizer.decode(encoded)
            assert decoded == char

    def test_bytes_boundary(self, trained_tokenizer):
        """Test handling of multi-byte UTF-8 sequences."""
        # 1 byte
        assert trained_tokenizer.decode(trained_tokenizer.encode("a")) == "a"
        # 2 bytes
        assert trained_tokenizer.decode(trained_tokenizer.encode("é")) == "é"
        # 3 bytes
        assert trained_tokenizer.decode(trained_tokenizer.encode("中")) == "中"
        # 4 bytes
        assert trained_tokenizer.decode(trained_tokenizer.encode("𝕳")) == "𝕳"

    def test_mixed_content(self, trained_tokenizer):
        """Test mixed content types."""
        text = "Hello 123 café 日本語 🎉"
        encoded = trained_tokenizer.encode(text)
        decoded = trained_tokenizer.decode(encoded)
        assert decoded == text
