"""Text preprocessing and normalization."""

from __future__ import annotations

import html
import re
import unicodedata


class TextPreprocessor:
    """Text cleaning and normalization for tokenizer training."""

    def __init__(
        self,
        normalize_unicode: bool = True,
        unicode_form: str = "NFC",
        remove_control_chars: bool = True,
        normalize_whitespace: bool = True,
        max_consecutive_newlines: int = 2,
    ):
        """Initialize preprocessor.

        Args:
            normalize_unicode: Whether to normalize unicode
            unicode_form: Unicode normalization form (NFC, NFD, NFKC, NFKD)
            remove_control_chars: Whether to remove control characters
            normalize_whitespace: Whether to normalize whitespace
            max_consecutive_newlines: Max consecutive newlines to allow
        """
        self.normalize_unicode = normalize_unicode
        self.unicode_form = unicode_form
        self.remove_control_chars = remove_control_chars
        self.normalize_whitespace = normalize_whitespace
        self.max_consecutive_newlines = max_consecutive_newlines

        # Compile regex patterns
        self._control_char_pattern = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
        self._multi_space_pattern = re.compile(r"[ \t]+")
        self._multi_newline_pattern = re.compile(r"\n{" + str(max_consecutive_newlines + 1) + r",}")

    def preprocess(self, text: str) -> str:
        """Apply all preprocessing steps to text.

        Args:
            text: Input text

        Returns:
            Preprocessed text
        """
        if not text:
            return text

        # Decode HTML entities
        text = html.unescape(text)

        # Unicode normalization
        if self.normalize_unicode:
            text = unicodedata.normalize(self.unicode_form, text)

        # Remove control characters (except newline, tab)
        if self.remove_control_chars:
            text = self._control_char_pattern.sub("", text)

        # Normalize whitespace
        if self.normalize_whitespace:
            # Normalize spaces (but preserve newlines)
            lines = text.split("\n")
            lines = [self._multi_space_pattern.sub(" ", line).strip() for line in lines]
            text = "\n".join(lines)

            # Limit consecutive newlines
            text = self._multi_newline_pattern.sub(
                "\n" * self.max_consecutive_newlines, text
            )

        return text.strip()

    def preprocess_batch(self, texts: list[str]) -> list[str]:
        """Preprocess a batch of texts.

        Args:
            texts: List of input texts

        Returns:
            List of preprocessed texts
        """
        return [self.preprocess(text) for text in texts]


class WikipediaPreprocessor(TextPreprocessor):
    """Specialized preprocessor for Wikipedia text."""

    def __init__(self, **kwargs):
        """Initialize Wikipedia preprocessor."""
        super().__init__(**kwargs)

        # Additional patterns for Wikipedia
        self._url_pattern = re.compile(r"https?://\S+")
        self._email_pattern = re.compile(r"\S+@\S+\.\S+")
        self._citation_pattern = re.compile(r"\[(\d+|citation needed)\]", re.IGNORECASE)

    def preprocess(self, text: str) -> str:
        """Apply Wikipedia-specific preprocessing.

        Args:
            text: Input text

        Returns:
            Preprocessed text
        """
        # Remove URLs
        text = self._url_pattern.sub("", text)

        # Remove emails
        text = self._email_pattern.sub("", text)

        # Remove citation markers
        text = self._citation_pattern.sub("", text)

        # Apply base preprocessing
        return super().preprocess(text)


def normalize_text(
    text: str,
    unicode_form: str = "NFC",
    strip: bool = True,
) -> str:
    """Simple text normalization function.

    Args:
        text: Input text
        unicode_form: Unicode normalization form
        strip: Whether to strip whitespace

    Returns:
        Normalized text
    """
    text = unicodedata.normalize(unicode_form, text)
    if strip:
        text = text.strip()
    return text
