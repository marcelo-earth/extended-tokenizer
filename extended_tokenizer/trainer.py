"""BPE Training Algorithm for the ExtendedTokenizer."""

from __future__ import annotations

import heapq
import multiprocessing as mp
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field

import regex

from extended_tokenizer.utils.progress import get_progress_bar
from extended_tokenizer.vocab import Vocabulary, save_vocab

# GPT-4 style pre-tokenization pattern
# This splits text into meaningful chunks before BPE
PRE_TOKENIZE_PATTERN = regex.compile(
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)


def pre_tokenize(text: str) -> list[str]:
    """Split text into pre-tokens using GPT-4 style regex."""
    return PRE_TOKENIZE_PATTERN.findall(text)


def text_to_bytes(text: str) -> list[int]:
    """Convert text to a list of byte values."""
    return list(text.encode("utf-8"))


@dataclass
class PairStats:
    """Statistics for tracking pair frequencies during BPE training."""

    # pair -> frequency
    pair_freqs: dict[tuple[int, int], int] = field(default_factory=Counter)

    # pair -> set of word indices containing this pair
    pair_to_words: dict[tuple[int, int], set[int]] = field(
        default_factory=lambda: defaultdict(set)
    )

    # word_idx -> current token sequence
    word_tokens: dict[int, list[int]] = field(default_factory=dict)

    # word_idx -> frequency (how many times this word appears in corpus)
    word_freqs: dict[int, int] = field(default_factory=dict)


class BPETrainer:
    """Byte-Pair Encoding trainer.

    Trains a BPE tokenizer from scratch using the standard algorithm:
    1. Initialize vocab with 256 byte tokens
    2. Count pair frequencies across the corpus
    3. Merge the most frequent pair
    4. Repeat until vocab_size is reached
    """

    def __init__(
        self,
        vocab_size: int = 300_000,
        min_frequency: int = 2,
        num_workers: int | None = None,
        chunk_size: int = 10_000,
    ):
        """Initialize the BPE trainer.

        Args:
            vocab_size: Target vocabulary size (including 256 byte tokens)
            min_frequency: Minimum pair frequency to consider for merging
            num_workers: Number of parallel workers for pair counting
            chunk_size: Number of texts to process in each chunk
        """
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.num_workers = num_workers or max(1, mp.cpu_count() - 1)
        self.chunk_size = chunk_size

        # Initialize vocabulary with byte tokens
        self.vocab = Vocabulary()

        # Training state
        self._stats: PairStats | None = None
        self._next_token_id = 256  # Start after byte tokens

    def train(
        self,
        corpus: Iterable[str],
        show_progress: bool = True,
    ) -> Vocabulary:
        """Train BPE on the given corpus.

        Args:
            corpus: Iterable of text strings to train on
            show_progress: Whether to show progress bars

        Returns:
            Trained Vocabulary object
        """
        # Step 1: Pre-tokenize and count word frequencies
        word_freqs = self._count_words(corpus, show_progress)

        # Step 2: Initialize training state
        self._init_training_state(word_freqs)

        # Step 3: Iteratively merge pairs
        num_merges = self.vocab_size - 256  # Exclude byte tokens
        self._train_merges(num_merges, show_progress)

        return self.vocab

    def _count_words(
        self,
        corpus: Iterable[str],
        show_progress: bool,
    ) -> Counter:
        """Pre-tokenize corpus and count word frequencies."""
        word_freqs: Counter = Counter()

        progress = get_progress_bar(
            desc="Pre-tokenizing",
            disable=not show_progress,
        )

        for text in corpus:
            tokens = pre_tokenize(text)
            for token in tokens:
                # Convert to tuple of bytes for hashability
                byte_seq = tuple(text_to_bytes(token))
                if byte_seq:  # Skip empty
                    word_freqs[byte_seq] += 1
            progress.update(1)

        progress.close()
        return word_freqs

    def _init_training_state(self, word_freqs: Counter) -> None:
        """Initialize pair statistics from word frequencies."""
        self._stats = PairStats()

        for word_idx, (word_bytes, freq) in enumerate(word_freqs.items()):
            word_tokens = list(word_bytes)
            self._stats.word_tokens[word_idx] = word_tokens
            self._stats.word_freqs[word_idx] = freq

            # Count pairs in this word
            for i in range(len(word_tokens) - 1):
                pair = (word_tokens[i], word_tokens[i + 1])
                self._stats.pair_freqs[pair] += freq
                self._stats.pair_to_words[pair].add(word_idx)

    def _train_merges(self, num_merges: int, show_progress: bool) -> None:
        """Perform BPE merges."""
        # Build max-heap of pairs by frequency (negate for max-heap)
        heap = [(-freq, pair) for pair, freq in self._stats.pair_freqs.items()]
        heapq.heapify(heap)

        progress = get_progress_bar(
            total=num_merges,
            desc="Training BPE",
            disable=not show_progress,
        )

        merges_done = 0
        while merges_done < num_merges and heap:
            # Get the most frequent pair
            neg_freq, pair = heapq.heappop(heap)
            freq = -neg_freq

            # Verify frequency is still accurate (may have changed due to merges)
            actual_freq = self._stats.pair_freqs.get(pair, 0)
            if actual_freq != freq:
                if actual_freq >= self.min_frequency:
                    heapq.heappush(heap, (-actual_freq, pair))
                continue

            if freq < self.min_frequency:
                continue

            # Perform the merge
            new_token_id = self._next_token_id
            self._next_token_id += 1

            self.vocab.add_merge(pair, new_token_id)

            # Update words containing this pair
            affected_pairs = self._apply_merge(pair, new_token_id)

            # Add affected pairs back to heap
            for affected_pair in affected_pairs:
                pair_freq = self._stats.pair_freqs.get(affected_pair, 0)
                if pair_freq >= self.min_frequency:
                    heapq.heappush(heap, (-pair_freq, affected_pair))

            merges_done += 1
            progress.update(1)

            if merges_done % 10000 == 0:
                progress.set_postfix({"vocab": self.vocab.vocab_size})

        progress.close()

    def _apply_merge(
        self,
        pair: tuple[int, int],
        new_token: int,
    ) -> set[tuple[int, int]]:
        """Apply a merge to all words containing the pair.

        Returns set of pairs whose frequencies changed.
        """
        affected_pairs: set[tuple[int, int]] = set()

        word_indices = list(self._stats.pair_to_words.get(pair, set()))
        del self._stats.pair_freqs[pair]
        del self._stats.pair_to_words[pair]

        for word_idx in word_indices:
            tokens = self._stats.word_tokens[word_idx]
            freq = self._stats.word_freqs[word_idx]

            # Find and merge all occurrences of the pair
            new_tokens = []
            i = 0
            while i < len(tokens):
                if (
                    i < len(tokens) - 1
                    and tokens[i] == pair[0]
                    and tokens[i + 1] == pair[1]
                ):
                    # Merge this pair
                    new_tokens.append(new_token)

                    # Update adjacent pair frequencies
                    # Left neighbor
                    if new_tokens and len(new_tokens) >= 2:
                        left_pair = (new_tokens[-2], new_token)
                        self._stats.pair_freqs[left_pair] += freq
                        self._stats.pair_to_words[left_pair].add(word_idx)
                        affected_pairs.add(left_pair)

                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            # Update right neighbor pairs
            for j in range(len(new_tokens) - 1):
                if new_tokens[j] == new_token:
                    right_pair = (new_token, new_tokens[j + 1])
                    if right_pair not in self._stats.pair_freqs:
                        self._stats.pair_freqs[right_pair] = 0
                    self._stats.pair_freqs[right_pair] += freq
                    self._stats.pair_to_words[right_pair].add(word_idx)
                    affected_pairs.add(right_pair)

            # Decrease frequency of broken pairs
            for j in range(len(tokens) - 1):
                old_pair = (tokens[j], tokens[j + 1])
                if old_pair != pair and old_pair in self._stats.pair_freqs:
                    self._stats.pair_freqs[old_pair] -= freq
                    if self._stats.pair_freqs[old_pair] <= 0:
                        del self._stats.pair_freqs[old_pair]
                        self._stats.pair_to_words[old_pair].discard(word_idx)

            self._stats.word_tokens[word_idx] = new_tokens

        return affected_pairs

    def save(self, path: str) -> None:
        """Save the trained vocabulary to disk."""
        save_vocab(self.vocab, path)

    @classmethod
    def load(cls, path: str) -> BPETrainer:
        """Load a trained tokenizer from disk."""
        from extended_tokenizer.vocab import load_vocab

        trainer = cls()
        trainer.vocab = load_vocab(path)
        trainer._next_token_id = 256 + trainer.vocab.num_merges
        return trainer
