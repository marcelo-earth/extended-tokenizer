"""Tests for vocabulary management."""

import json
import tempfile
from pathlib import Path

import pytest

from extended_tokenizer.vocab import (
    SPECIAL_TOKENS,
    Vocabulary,
    load_vocab,
    save_vocab,
)


class TestVocabulary:
    """Tests for Vocabulary class."""

    def test_init_creates_byte_vocab(self):
        """Test that initialization creates 256 byte tokens."""
        vocab = Vocabulary()
        assert len(vocab.token_to_bytes) == 256
        assert len(vocab.bytes_to_token) == 256

        # Check some byte mappings
        assert vocab.token_to_bytes[0] == b"\x00"
        assert vocab.token_to_bytes[65] == b"A"
        assert vocab.token_to_bytes[255] == b"\xff"

    def test_add_merge(self):
        """Test adding merge rules."""
        vocab = Vocabulary()

        # Add a merge: A + B -> 256
        vocab.add_merge((65, 66), 256)

        assert vocab.token_to_bytes[256] == b"AB"
        assert vocab.bytes_to_token[b"AB"] == 256
        assert len(vocab.merges) == 1
        assert vocab.merges[0] == ((65, 66), 256)

    def test_merge_priority(self):
        """Test merge priority tracking."""
        vocab = Vocabulary()

        vocab.add_merge((65, 66), 256)  # AB
        vocab.add_merge((256, 67), 257)  # ABC

        assert vocab.merge_priority[(65, 66)] == (0, 256)
        assert vocab.merge_priority[(256, 67)] == (1, 257)

    def test_vocab_size(self):
        """Test vocabulary size calculation."""
        vocab = Vocabulary()

        # 256 byte tokens + special tokens
        base_size = 256 + len(SPECIAL_TOKENS)
        assert vocab.vocab_size == base_size

        # Add some merges
        vocab.add_merge((65, 66), 256)
        vocab.add_merge((67, 68), 257)

        assert vocab.vocab_size == base_size + 2

    def test_get_token_bytes(self):
        """Test getting bytes for token IDs."""
        vocab = Vocabulary()

        assert vocab.get_token_bytes(65) == b"A"
        assert vocab.get_token_bytes(999999) is None

    def test_get_token_id(self):
        """Test getting token ID for byte sequence."""
        vocab = Vocabulary()

        assert vocab.get_token_id(b"A") == 65
        assert vocab.get_token_id(b"AB") is None

        vocab.add_merge((65, 66), 256)
        assert vocab.get_token_id(b"AB") == 256

    def test_to_dict_and_from_dict(self):
        """Test serialization round-trip."""
        vocab = Vocabulary()
        vocab.add_merge((65, 66), 256)
        vocab.add_merge((256, 67), 257)

        data = vocab.to_dict()

        # Verify structure
        assert "token_to_bytes" in data
        assert "merges" in data
        assert "special_tokens" in data
        assert "vocab_size" in data

        # Reconstruct
        vocab2 = Vocabulary.from_dict(data)

        assert vocab2.vocab_size == vocab.vocab_size
        assert len(vocab2.merges) == len(vocab.merges)
        assert vocab2.token_to_bytes[256] == vocab.token_to_bytes[256]


class TestSaveLoad:
    """Tests for save/load functions."""

    def test_save_and_load(self):
        """Test saving and loading vocabulary."""
        vocab = Vocabulary()
        vocab.add_merge((65, 66), 256)
        vocab.add_merge((67, 68), 257)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "vocab"

            # Save
            save_vocab(vocab, path)

            # Verify files created
            assert (path / "vocab.json").exists()
            assert (path / "merges.txt").exists()
            assert (path / "config.json").exists()

            # Load
            vocab2 = load_vocab(path)

            assert vocab2.vocab_size == vocab.vocab_size
            assert len(vocab2.merges) == len(vocab.merges)

    def test_load_nonexistent_raises(self):
        """Test loading from nonexistent path raises error."""
        with pytest.raises(FileNotFoundError):
            load_vocab("/nonexistent/path")

    def test_merges_txt_format(self):
        """Test that merges.txt has correct format."""
        vocab = Vocabulary()
        vocab.add_merge((65, 66), 256)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "vocab"
            save_vocab(vocab, path)

            with open(path / "merges.txt") as f:
                content = f.read()

            assert "65 66 -> 256" in content

    def test_config_json_content(self):
        """Test config.json has expected content."""
        vocab = Vocabulary()
        vocab.add_merge((65, 66), 256)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "vocab"
            save_vocab(vocab, path)

            with open(path / "config.json") as f:
                config = json.load(f)

            assert config["tokenizer_type"] == "BPE"
            assert config["num_merges"] == 1
            assert "special_tokens" in config
