"""Wikipedia dump downloader and processor."""

from __future__ import annotations

import bz2
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from pathlib import Path
from xml.etree.ElementTree import iterparse

import mwparserfromhell
import requests
from tqdm import tqdm

from extended_tokenizer.data.preprocessor import TextPreprocessor

# Wikipedia dump URLs
DUMP_BASE_URL = "https://dumps.wikimedia.org/enwiki/latest"
DUMP_FILENAME = "enwiki-latest-pages-articles.xml.bz2"


class WikipediaDownloader:
    """Download Wikipedia dump files."""

    DUMP_URL = f"{DUMP_BASE_URL}/{DUMP_FILENAME}"

    def __init__(self, output_dir: str = "data/wikipedia"):
        """Initialize downloader.

        Args:
            output_dir: Directory to save downloaded files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download(
        self,
        url: str | None = None,
        show_progress: bool = True,
    ) -> Path:
        """Download Wikipedia dump.

        Args:
            url: Optional custom URL (defaults to latest dump)
            show_progress: Whether to show progress bar

        Returns:
            Path to downloaded file
        """
        url = url or self.DUMP_URL
        filename = url.split("/")[-1]
        output_path = self.output_dir / filename

        if output_path.exists():
            print(f"File already exists: {output_path}")
            return output_path

        print(f"Downloading {url}")
        print("This may take a while (~20GB)")

        # Stream download with progress
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        with open(output_path, "wb") as f:
            with tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                disable=not show_progress,
                desc="Downloading",
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))

        print(f"Downloaded to {output_path}")
        return output_path


class WikipediaExtractor:
    """Extract and process articles from Wikipedia dump."""

    # Namespaces to include (0 = main articles)
    INCLUDED_NAMESPACES = {0}

    # Minimum article length (characters)
    MIN_ARTICLE_LENGTH = 500

    def __init__(
        self,
        preprocessor: TextPreprocessor | None = None,
        min_length: int = MIN_ARTICLE_LENGTH,
    ):
        """Initialize extractor.

        Args:
            preprocessor: Text preprocessor instance
            min_length: Minimum article length to include
        """
        self.preprocessor = preprocessor or TextPreprocessor()
        self.min_length = min_length

    def extract_articles(
        self,
        dump_path: str,
        max_articles: int | None = None,
        show_progress: bool = True,
    ) -> Iterator[str]:
        """Stream articles from Wikipedia XML dump.

        Args:
            dump_path: Path to Wikipedia dump file (.xml.bz2)
            max_articles: Optional limit on number of articles
            show_progress: Whether to show progress bar

        Yields:
            Cleaned article text
        """
        dump_path = Path(dump_path)

        # Open compressed file
        if dump_path.suffix == ".bz2":
            file_handle = bz2.open(dump_path, "rt", encoding="utf-8")
        else:
            file_handle = open(dump_path, encoding="utf-8")

        article_count = 0

        try:
            # Use iterparse for memory-efficient parsing
            context = iterparse(file_handle, events=("end",))

            pbar = tqdm(
                desc="Extracting articles",
                disable=not show_progress,
                unit=" articles",
            )

            for event, elem in context:
                # Look for page elements
                if elem.tag.endswith("page"):
                    article = self._process_page(elem)

                    if article:
                        article_count += 1
                        pbar.update(1)
                        yield article

                        if max_articles and article_count >= max_articles:
                            break

                    # Clear element to free memory
                    elem.clear()

            pbar.close()

        finally:
            file_handle.close()

        print(f"Extracted {article_count} articles")

    def _process_page(self, page_elem: ET.Element) -> str | None:
        """Process a single Wikipedia page element.

        Args:
            page_elem: XML element for a page

        Returns:
            Cleaned article text or None if should be skipped
        """
        # Find namespace
        ns_elem = page_elem.find(".//{http://www.mediawiki.org/xml/export-0.10/}ns")
        if ns_elem is None:
            ns_elem = page_elem.find(".//ns")

        if ns_elem is not None:
            try:
                ns = int(ns_elem.text)
                if ns not in self.INCLUDED_NAMESPACES:
                    return None
            except (ValueError, TypeError):
                pass

        # Find title
        title_elem = page_elem.find(
            ".//{http://www.mediawiki.org/xml/export-0.10/}title"
        )
        if title_elem is None:
            title_elem = page_elem.find(".//title")

        # Skip disambiguation pages, lists, etc.
        if title_elem is not None and title_elem.text:
            title = title_elem.text.lower()
            if any(skip in title for skip in ["disambiguation", "list of", "index of"]):
                return None

        # Find text content
        text_elem = page_elem.find(
            ".//{http://www.mediawiki.org/xml/export-0.10/}text"
        )
        if text_elem is None:
            text_elem = page_elem.find(".//text")

        if text_elem is None or not text_elem.text:
            return None

        raw_text = text_elem.text

        # Skip redirects
        if raw_text.lower().startswith("#redirect"):
            return None

        # Parse wikitext and extract plain text
        try:
            clean_text = self._clean_wikitext(raw_text)
        except Exception:
            return None

        # Apply text preprocessing
        clean_text = self.preprocessor.preprocess(clean_text)

        # Filter by length
        if len(clean_text) < self.min_length:
            return None

        return clean_text

    def _clean_wikitext(self, wikitext: str) -> str:
        """Convert wikitext to plain text.

        Args:
            wikitext: Raw Wikipedia markup

        Returns:
            Plain text content
        """
        # Parse with mwparserfromhell
        parsed = mwparserfromhell.parse(wikitext)

        # Remove templates, tables, etc.
        for template in parsed.filter_templates():
            try:
                parsed.remove(template)
            except ValueError:
                pass

        # Remove references
        for tag in parsed.filter_tags():
            if tag.tag.lower() in ("ref", "references", "table", "gallery"):
                try:
                    parsed.remove(tag)
                except ValueError:
                    pass

        # Get plain text
        text = parsed.strip_code()

        # Additional cleaning
        # Remove file/image links
        text = re.sub(r"\[\[File:[^\]]+\]\]", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\[\[Image:[^\]]+\]\]", "", text, flags=re.IGNORECASE)

        # Remove category links
        text = re.sub(r"\[\[Category:[^\]]+\]\]", "", text, flags=re.IGNORECASE)

        # Clean up remaining wiki syntax
        text = re.sub(r"\[\[([^\]|]+\|)?([^\]]+)\]\]", r"\2", text)  # [[link|text]] -> text
        text = re.sub(r"'{2,}", "", text)  # Remove bold/italic markers
        text = re.sub(r"={2,}.*?={2,}", "", text)  # Remove section headers

        # Clean whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)

        return text.strip()


def stream_wikipedia_texts(
    dump_path: str,
    max_articles: int | None = None,
    min_length: int = 500,
    show_progress: bool = True,
) -> Iterator[str]:
    """Convenience function to stream Wikipedia articles.

    Args:
        dump_path: Path to Wikipedia dump file
        max_articles: Optional limit on articles
        min_length: Minimum article length
        show_progress: Whether to show progress

    Yields:
        Cleaned article text
    """
    extractor = WikipediaExtractor(min_length=min_length)
    yield from extractor.extract_articles(
        dump_path,
        max_articles=max_articles,
        show_progress=show_progress,
    )
