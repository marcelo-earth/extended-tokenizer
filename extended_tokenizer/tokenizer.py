"""Main tokenizer interface for ExtendedTokenizer."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import regex

from extended_tokenizer.vocab import (
    SPECIAL_TOKEN_IDS,
    SPECIAL_TOKENS,
    load_vocab,
)

# GPT-4 style pre-tokenization pattern
PRE_TOKENIZE_PATTERN = regex.compile(
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)

# Default HuggingFace Hub model ID
DEFAULT_MODEL_ID = "marcelo-earth/extended-tokenizer-300k"


class ExtendedTokenizer:
    """Main tokenizer class - 300K vocabulary BPE tokenizer.

    This tokenizer uses Byte-Pair Encoding (BPE) with a vocabulary of 300,000
    tokens trained on English Wikipedia. It's designed for production LLM training.

    Usage:
        # Load from HuggingFace Hub (recommended)
        tokenizer = ExtendedTokenizer.from_pretrained()

        # Or load from local path
        tokenizer = ExtendedTokenizer(vocab_path="/path/to/vocab")

        # Encode text
        tokens = tokenizer.encode("Hello world!")

        # Decode back to text
        text = tokenizer.decode(tokens)
    """

    def __init__(self, vocab_path: str | Path | None = None):
        """Initialize the tokenizer.

        Args:
            vocab_path: Path to vocabulary directory. If None, downloads from
                       HuggingFace Hub.
        """
        if vocab_path is None:
            vocab_path = self._download_from_hub(DEFAULT_MODEL_ID)

        self.vocab = load_vocab(vocab_path)
        self._vocab_path = vocab_path

        # Build encoding cache for common tokens
        self._token_cache: dict[str, list[int]] = {}
        self._cache_max_size = 100_000

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = DEFAULT_MODEL_ID,
        cache_dir: str | None = None,
    ) -> ExtendedTokenizer:
        """Load tokenizer from HuggingFace Hub.

        Args:
            model_id: HuggingFace Hub model identifier
            cache_dir: Optional directory to cache downloaded files

        Returns:
            Initialized ExtendedTokenizer
        """
        vocab_path = cls._download_from_hub(model_id, cache_dir)
        return cls(vocab_path=vocab_path)

    @staticmethod
    def _download_from_hub(
        model_id: str,
        cache_dir: str | None = None,
    ) -> Path:
        """Download vocabulary from HuggingFace Hub.

        Args:
            model_id: HuggingFace Hub model identifier
            cache_dir: Optional directory to cache downloaded files

        Returns:
            Path to downloaded vocabulary directory
        """
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise ImportError(
                "huggingface_hub is required to download from HuggingFace Hub. "
                "Install it with: pip install huggingface-hub"
            )

        # Download the model files
        local_dir = snapshot_download(
            repo_id=model_id,
            cache_dir=cache_dir,
            allow_patterns=["vocab.json", "merges.txt", "config.json"],
        )

        return Path(local_dir)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Encode text to token IDs.

        Args:
            text: Text string to encode
            add_special_tokens: Whether to add start/end tokens

        Returns:
            List of token IDs
        """
        if not text:
            return []

        tokens: list[int] = []

        if add_special_tokens:
            tokens.append(SPECIAL_TOKENS["<|startoftext|>"])

        # Pre-tokenize
        pre_tokens = PRE_TOKENIZE_PATTERN.findall(text)

        for pre_token in pre_tokens:
            # Check cache
            if pre_token in self._token_cache:
                tokens.extend(self._token_cache[pre_token])
                continue

            # Encode this pre-token
            encoded = self._encode_chunk(pre_token)
            tokens.extend(encoded)

            # Cache if not too large
            if len(self._token_cache) < self._cache_max_size:
                self._token_cache[pre_token] = encoded

        if add_special_tokens:
            tokens.append(SPECIAL_TOKENS["<|endoftext|>"])

        return tokens

    def _encode_chunk(self, text: str) -> list[int]:
        """Encode a single pre-tokenized chunk using BPE."""
        # Convert to bytes
        byte_seq = text.encode("utf-8")

        # Initialize with byte tokens
        tokens = list(byte_seq)

        if len(tokens) <= 1:
            return tokens

        # Apply merges iteratively
        while len(tokens) >= 2:
            # Find the highest priority merge
            best_pair = None
            best_priority = float("inf")
            best_idx = -1

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in self.vocab.merge_priority:
                    priority, _ = self.vocab.merge_priority[pair]
                    if priority < best_priority:
                        best_priority = priority
                        best_pair = pair
                        best_idx = i

            if best_pair is None:
                break

            # Apply the merge
            _, new_token = self.vocab.merge_priority[best_pair]
            tokens = (
                tokens[:best_idx]
                + [new_token]
                + tokens[best_idx + 2:]
            )

        return tokens

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text.

        Args:
            ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens in output

        Returns:
            Decoded text string
        """
        byte_chunks: list[bytes] = []

        for token_id in ids:
            # Check for special tokens
            if token_id in SPECIAL_TOKEN_IDS:
                if not skip_special_tokens:
                    byte_chunks.append(SPECIAL_TOKEN_IDS[token_id].encode("utf-8"))
                continue

            # Get bytes for this token
            token_bytes = self.vocab.get_token_bytes(token_id)
            if token_bytes is not None:
                byte_chunks.append(token_bytes)

        # Concatenate and decode
        full_bytes = b"".join(byte_chunks)

        # Handle potential UTF-8 decoding errors gracefully
        return full_bytes.decode("utf-8", errors="replace")

    def encode_batch(
        self,
        texts: list[str],
        add_special_tokens: bool = False,
        num_workers: int | None = None,
    ) -> list[list[int]]:
        """Batch encode multiple texts with optional parallelization.

        Args:
            texts: List of text strings to encode
            add_special_tokens: Whether to add start/end tokens
            num_workers: Number of parallel workers (None for sequential)

        Returns:
            List of token ID lists
        """
        if num_workers is None or num_workers <= 1 or len(texts) < 10:
            return [self.encode(t, add_special_tokens) for t in texts]

        # Parallel encoding
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(
                lambda t: self.encode(t, add_special_tokens),
                texts
            ))

        return results

    def decode_batch(
        self,
        ids_batch: list[list[int]],
        skip_special_tokens: bool = True,
    ) -> list[str]:
        """Batch decode multiple token sequences.

        Args:
            ids_batch: List of token ID lists
            skip_special_tokens: Whether to skip special tokens

        Returns:
            List of decoded text strings
        """
        return [self.decode(ids, skip_special_tokens) for ids in ids_batch]

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size."""
        return self.vocab.vocab_size

    @property
    def eos_token_id(self) -> int:
        """End of sequence token ID."""
        return SPECIAL_TOKENS["<|endoftext|>"]

    @property
    def bos_token_id(self) -> int:
        """Beginning of sequence token ID."""
        return SPECIAL_TOKENS["<|startoftext|>"]

    @property
    def pad_token_id(self) -> int:
        """Padding token ID."""
        return SPECIAL_TOKENS["<|padding|>"]

    def get_vocab(self) -> dict[str, int]:
        """Get vocabulary as a dict mapping token strings to IDs.

        Note: This is provided for compatibility but may not be meaningful
        for byte-level tokenizers.
        """
        result = {}
        for token_id, token_bytes in self.vocab.token_to_bytes.items():
            try:
                token_str = token_bytes.decode("utf-8")
                result[token_str] = token_id
            except UnicodeDecodeError:
                # Use hex representation for non-UTF8 bytes
                result[f"<0x{token_bytes.hex()}>"] = token_id

        # Add special tokens
        result.update(SPECIAL_TOKENS)

        return result

    def __len__(self) -> int:
        """Return vocabulary size."""
        return self.vocab_size

    def __repr__(self) -> str:
        """String representation."""
        return f"ExtendedTokenizer(vocab_size={self.vocab_size})"
