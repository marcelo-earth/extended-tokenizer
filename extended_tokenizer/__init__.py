"""
ExtendedTokenizer - A 300K vocabulary BPE tokenizer trained on Wikipedia.

Usage:
    from extended_tokenizer import ExtendedTokenizer

    tokenizer = ExtendedTokenizer.from_pretrained()
    tokens = tokenizer.encode("Hello world!")
    text = tokenizer.decode(tokens)
"""

from extended_tokenizer.tokenizer import ExtendedTokenizer
from extended_tokenizer.trainer import BPETrainer
from extended_tokenizer.vocab import Vocabulary, load_vocab, save_vocab

__version__ = "0.1.0"
__all__ = [
    "ExtendedTokenizer",
    "BPETrainer",
    "Vocabulary",
    "load_vocab",
    "save_vocab",
]
