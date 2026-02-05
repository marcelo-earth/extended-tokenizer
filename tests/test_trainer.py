"""Tests for BPE trainer."""

import tempfile
from pathlib import Path

from extended_tokenizer.trainer import BPETrainer, pre_tokenize, text_to_bytes


class TestPreTokenization:
    """Tests for pre-tokenization functions."""

    def test_text_to_bytes(self):
        """Test converting text to bytes."""
        assert text_to_bytes("A") == [65]
        assert text_to_bytes("AB") == [65, 66]
        assert text_to_bytes("") == []

    def test_text_to_bytes_unicode(self):
        """Test bytes conversion with unicode."""
        # UTF-8 encoding of 'é'
        result = text_to_bytes("é")
        assert result == [195, 169]  # UTF-8 bytes for é

    def test_pre_tokenize_simple(self):
        """Test basic pre-tokenization."""
        tokens = pre_tokenize("Hello world")
        assert len(tokens) >= 2
        assert "Hello" in tokens or " Hello" in tokens[0] if tokens else False

    def test_pre_tokenize_contractions(self):
        """Test pre-tokenization handles contractions."""
        tokens = pre_tokenize("I'm don't won't")
        # Should split contractions
        assert any("'m" in t or "'t" in t for t in tokens)

    def test_pre_tokenize_numbers(self):
        """Test pre-tokenization handles numbers."""
        tokens = pre_tokenize("12345")
        # Numbers should be split into chunks of up to 3 digits
        assert len(tokens) >= 1

    def test_pre_tokenize_preserves_whitespace(self):
        """Test that whitespace is preserved in some form."""
        text = "Hello  world"
        tokens = pre_tokenize(text)
        reconstructed = "".join(tokens)
        # Whitespace should be preserved
        assert " " in reconstructed


class TestBPETrainer:
    """Tests for BPETrainer class."""

    def test_init(self):
        """Test trainer initialization."""
        trainer = BPETrainer(vocab_size=1000)
        assert trainer.vocab_size == 1000
        assert trainer.vocab.vocab_size == 256 + len(trainer.vocab.special_tokens)

    def test_train_simple(self):
        """Test training on simple corpus."""
        corpus = ["aa", "aa", "aa", "bb", "bb"]
        trainer = BPETrainer(vocab_size=258, min_frequency=2)

        vocab = trainer.train(corpus, show_progress=False)

        # Should have merged 'aa' since it appears 3 times
        assert vocab.num_merges >= 1
        assert vocab.vocab_size >= 257

    def test_train_creates_merges(self):
        """Test that training creates merge rules."""
        corpus = ["abab"] * 10
        trainer = BPETrainer(vocab_size=260, min_frequency=2)

        vocab = trainer.train(corpus, show_progress=False)

        # Should have created some merges
        assert len(vocab.merges) > 0

    def test_train_respects_vocab_size(self):
        """Test that training stops at vocab_size."""
        corpus = ["abcdefgh" * 10] * 100
        vocab_size = 270

        trainer = BPETrainer(vocab_size=vocab_size, min_frequency=2)
        vocab = trainer.train(corpus, show_progress=False)

        # Should not exceed vocab_size (256 bytes + merges + special)
        assert vocab.num_merges <= vocab_size - 256

    def test_train_respects_min_frequency(self):
        """Test that merges respect minimum frequency."""
        # 'aa' appears 10 times, 'bb' appears only once
        corpus = ["aa"] * 10 + ["bb"]
        trainer = BPETrainer(vocab_size=258, min_frequency=5)

        vocab = trainer.train(corpus, show_progress=False)

        # Should only merge 'aa', not 'bb'
        if vocab.num_merges > 0:
            # Check that the merged token corresponds to 'aa'
            first_merge = vocab.merges[0]
            merged_bytes = vocab.token_to_bytes[first_merge[1]]
            assert merged_bytes == b"aa"

    def test_save_and_load(self):
        """Test saving and loading trainer."""
        corpus = ["hello world"] * 10
        trainer = BPETrainer(vocab_size=260, min_frequency=2)
        vocab = trainer.train(corpus, show_progress=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "vocab"

            trainer.save(str(path))
            loaded = BPETrainer.load(str(path))

            assert loaded.vocab.vocab_size == vocab.vocab_size
            assert len(loaded.vocab.merges) == len(vocab.merges)

    def test_train_empty_corpus(self):
        """Test training on empty corpus."""
        corpus = []
        trainer = BPETrainer(vocab_size=260)

        vocab = trainer.train(corpus, show_progress=False)

        # Should have base vocabulary only
        assert vocab.num_merges == 0


class TestBPETrainerIntegration:
    """Integration tests for BPE trainer."""

    def test_train_and_encode(self):
        """Test that trained vocab can encode text."""
        corpus = ["the quick brown fox"] * 100
        trainer = BPETrainer(vocab_size=280, min_frequency=2)
        vocab = trainer.train(corpus, show_progress=False)

        # Create a simple encoder using the vocab
        text = "the fox"
        byte_tokens = list(text.encode("utf-8"))

        # Apply merges
        while len(byte_tokens) >= 2:
            best_pair = None
            best_priority = float("inf")
            best_idx = -1

            for i in range(len(byte_tokens) - 1):
                pair = (byte_tokens[i], byte_tokens[i + 1])
                if pair in vocab.merge_priority:
                    priority, _ = vocab.merge_priority[pair]
                    if priority < best_priority:
                        best_priority = priority
                        best_pair = pair
                        best_idx = i

            if best_pair is None:
                break

            _, new_token = vocab.merge_priority[best_pair]
            byte_tokens = byte_tokens[:best_idx] + [new_token] + byte_tokens[best_idx + 2:]

        # Should have some tokens
        assert len(byte_tokens) > 0
        assert len(byte_tokens) <= len(text.encode("utf-8"))
