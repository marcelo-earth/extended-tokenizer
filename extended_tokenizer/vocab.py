"""Vocabulary management for the BPE tokenizer."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# Special tokens
SPECIAL_TOKENS = {
    "<|endoftext|>": 300001,
    "<|padding|>": 300002,
    "<|startoftext|>": 300003,
}

# Reverse mapping
SPECIAL_TOKEN_IDS = {v: k for k, v in SPECIAL_TOKENS.items()}


@dataclass
class Vocabulary:
    """Vocabulary for BPE tokenizer.

    Stores the mapping between token IDs and their byte sequences,
    as well as the merge rules learned during training.
    """

    # Token ID -> bytes mapping
    token_to_bytes: dict[int, bytes] = field(default_factory=dict)

    # Bytes -> token ID mapping (for encoding)
    bytes_to_token: dict[bytes, int] = field(default_factory=dict)

    # Merge rules: list of (pair_of_token_ids, new_token_id)
    merges: list[tuple[tuple[int, int], int]] = field(default_factory=list)

    # Merge priority lookup: pair -> (priority, new_token)
    merge_priority: dict[tuple[int, int], tuple[int, int]] = field(default_factory=dict)

    # Special tokens
    special_tokens: dict[str, int] = field(default_factory=lambda: SPECIAL_TOKENS.copy())

    def __post_init__(self) -> None:
        """Initialize base vocabulary if empty."""
        if not self.token_to_bytes:
            self._init_byte_vocab()

    def _init_byte_vocab(self) -> None:
        """Initialize vocabulary with 256 byte tokens."""
        for i in range(256):
            byte_val = bytes([i])
            self.token_to_bytes[i] = byte_val
            self.bytes_to_token[byte_val] = i

    def add_merge(self, pair: tuple[int, int], new_token_id: int) -> None:
        """Add a merge rule."""
        # Compute merged bytes
        merged_bytes = self.token_to_bytes[pair[0]] + self.token_to_bytes[pair[1]]

        # Store the mapping
        self.token_to_bytes[new_token_id] = merged_bytes
        self.bytes_to_token[merged_bytes] = new_token_id

        # Store the merge rule
        priority = len(self.merges)
        self.merges.append((pair, new_token_id))
        self.merge_priority[pair] = (priority, new_token_id)

    def get_token_bytes(self, token_id: int) -> bytes | None:
        """Get the bytes for a token ID."""
        if token_id in SPECIAL_TOKEN_IDS:
            return SPECIAL_TOKEN_IDS[token_id].encode("utf-8")
        return self.token_to_bytes.get(token_id)

    def get_token_id(self, byte_seq: bytes) -> int | None:
        """Get the token ID for a byte sequence."""
        return self.bytes_to_token.get(byte_seq)

    def get_special_token_id(self, token_str: str) -> int | None:
        """Get the ID for a special token."""
        return self.special_tokens.get(token_str)

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size including special tokens."""
        return len(self.token_to_bytes) + len(self.special_tokens)

    @property
    def num_merges(self) -> int:
        """Number of merge rules."""
        return len(self.merges)

    def to_dict(self) -> dict:
        """Convert vocabulary to a serializable dict."""
        return {
            "token_to_bytes": {
                str(k): list(v) for k, v in self.token_to_bytes.items()
            },
            "merges": [
                {"pair": list(pair), "new_token": new_token}
                for pair, new_token in self.merges
            ],
            "special_tokens": self.special_tokens,
            "vocab_size": self.vocab_size,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Vocabulary:
        """Create vocabulary from a dict."""
        vocab = cls()
        vocab.token_to_bytes = {}
        vocab.bytes_to_token = {}
        vocab.merges = []
        vocab.merge_priority = {}
        vocab.special_tokens = data.get("special_tokens", SPECIAL_TOKENS.copy())

        # Load token mappings
        for token_id_str, byte_list in data["token_to_bytes"].items():
            token_id = int(token_id_str)
            byte_val = bytes(byte_list)
            vocab.token_to_bytes[token_id] = byte_val
            vocab.bytes_to_token[byte_val] = token_id

        # Load merge rules
        for i, merge_data in enumerate(data["merges"]):
            pair = tuple(merge_data["pair"])
            new_token = merge_data["new_token"]
            vocab.merges.append((pair, new_token))
            vocab.merge_priority[pair] = (i, new_token)

        return vocab


def save_vocab(vocab: Vocabulary, path: str | Path) -> None:
    """Save vocabulary to disk.

    Creates two files:
    - vocab.json: Token ID to bytes mapping and metadata
    - merges.txt: BPE merge rules in text format
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Save vocab.json
    vocab_path = path / "vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab.to_dict(), f, indent=2)

    # Save merges.txt (human-readable format)
    merges_path = path / "merges.txt"
    with open(merges_path, "w", encoding="utf-8") as f:
        f.write("# BPE merge rules for extended-tokenizer\n")
        f.write(f"# Total merges: {len(vocab.merges)}\n")
        f.write("# Format: token1_id token2_id -> new_token_id\n\n")
        for pair, new_token in vocab.merges:
            f.write(f"{pair[0]} {pair[1]} -> {new_token}\n")

    # Save config.json for HuggingFace compatibility
    config_path = path / "config.json"
    config = {
        "tokenizer_type": "BPE",
        "vocab_size": vocab.vocab_size,
        "num_merges": vocab.num_merges,
        "special_tokens": vocab.special_tokens,
        "model_name": "extended-tokenizer-300k",
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def load_vocab(path: str | Path) -> Vocabulary:
    """Load vocabulary from disk."""
    path = Path(path)

    vocab_path = path / "vocab.json"
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")

    with open(vocab_path, encoding="utf-8") as f:
        data = json.load(f)

    return Vocabulary.from_dict(data)
