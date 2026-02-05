# ExtendedTokenizer

A 300K vocabulary BPE tokenizer trained on Wikipedia for production LLM training.

[![CI](https://github.com/marcelo-earth/extended-tokenizer/actions/workflows/ci.yml/badge.svg)](https://github.com/marcelo-earth/extended-tokenizer/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/extended-tokenizer.svg)](https://badge.fury.io/py/extended-tokenizer)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **300,000 token vocabulary** - Larger vocabulary for better compression and fewer tokens
- **Trained on Wikipedia** - High-quality English text from Wikipedia dumps
- **GPT-4 style pre-tokenization** - Uses the same regex pattern for consistent tokenization
- **Fast encoding/decoding** - Optimized BPE implementation with caching
- **HuggingFace Hub integration** - Download pre-trained vocabulary automatically
- **Production ready** - Designed for LLM training pipelines

## Installation

```bash
pip install extended-tokenizer
```

## Quick Start

```python
from extended_tokenizer import ExtendedTokenizer

# Load pre-trained 300K tokenizer from HuggingFace Hub
tokenizer = ExtendedTokenizer.from_pretrained("marcelo-earth/extended-tokenizer-300k")

# Encode text to token IDs
tokens = tokenizer.encode("Hello, world!")
print(tokens)  # [15496, 11, 995, 0]

# Decode back to text
text = tokenizer.decode(tokens)
print(text)  # "Hello, world!"

# Batch encoding
texts = ["Hello", "World", "Python"]
token_batches = tokenizer.encode_batch(texts)
```

## Loading from Local Path

If you have a trained vocabulary locally:

```python
tokenizer = ExtendedTokenizer(vocab_path="/path/to/vocab")
```

## Training Your Own Tokenizer

### 1. Download Wikipedia

```bash
python scripts/download_wikipedia.py --output data/wikipedia/
```

This downloads the latest English Wikipedia dump (~20GB compressed).

### 2. Train the Tokenizer

```bash
python scripts/train_tokenizer.py \
    --input data/wikipedia/enwiki-latest-pages-articles.xml.bz2 \
    --vocab-size 300000 \
    --output vocab/bpe_300k
```

Training parameters:
- `--vocab-size`: Target vocabulary size (default: 300,000)
- `--min-frequency`: Minimum pair frequency for merging (default: 2)
- `--max-articles`: Limit articles for testing (optional)

### 3. Benchmark

Compare your tokenizer against tiktoken:

```bash
pip install tiktoken
python scripts/benchmark.py --vocab-path vocab/bpe_300k
```

## API Reference

### ExtendedTokenizer

```python
class ExtendedTokenizer:
    def __init__(self, vocab_path: str = None):
        """Initialize tokenizer. Downloads from HuggingFace if no path given."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Encode text to token IDs."""

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""

    def encode_batch(self, texts: list[str], num_workers: int = None) -> list[list[int]]:
        """Batch encode with optional parallelization."""

    @classmethod
    def from_pretrained(cls, model_id: str = "marcelo-earth/extended-tokenizer-300k"):
        """Load from HuggingFace Hub."""

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size (300,003 with special tokens)."""

    @property
    def eos_token_id(self) -> int:
        """End of sequence token ID."""

    @property
    def bos_token_id(self) -> int:
        """Beginning of sequence token ID."""

    @property
    def pad_token_id(self) -> int:
        """Padding token ID."""
```

### BPETrainer

```python
class BPETrainer:
    def __init__(self, vocab_size: int = 300_000, min_frequency: int = 2):
        """Initialize trainer."""

    def train(self, corpus: Iterable[str]) -> Vocabulary:
        """Train BPE on corpus."""

    def save(self, path: str):
        """Save trained vocabulary."""
```

## Special Tokens

| Token | ID |
|-------|-----|
| `<\|endoftext\|>` | 300001 |
| `<\|padding\|>` | 300002 |
| `<\|startoftext\|>` | 300003 |

## Vocabulary Structure

| Range | Count | Description |
|-------|-------|-------------|
| 0-255 | 256 | Raw byte tokens |
| 256-300000 | ~300K | BPE merge tokens |
| 300001-300003 | 3 | Special tokens |

## Development

```bash
# Clone the repo
git clone https://github.com/marcelo-earth/extended-tokenizer.git
cd extended-tokenizer

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run linting
ruff check extended_tokenizer tests
black --check extended_tokenizer tests

# Type checking
mypy extended_tokenizer
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use this tokenizer in your research, please cite:

```bibtex
@software{extended_tokenizer,
  author = {Marcelo},
  title = {ExtendedTokenizer: A 300K Vocabulary BPE Tokenizer},
  year = {2024},
  url = {https://github.com/marcelo-earth/extended-tokenizer}
}
```

## Acknowledgments

- Training data from [English Wikipedia](https://dumps.wikimedia.org/)
- Pre-tokenization pattern from [tiktoken](https://github.com/openai/tiktoken)
- Wikipedia parsing with [mwparserfromhell](https://github.com/earwig/mwparserfromhell)
