"""Gate the checked-in corpus caches against the current parser configuration.

A stale cache fails silently at runtime: startup re-parses the whole corpus instead of loading the
cache. That cost 660 s for the five filings before this gate existed. Hashing the PDFs and comparing
signatures costs well under a second, so the failure surfaces in the default suite instead of in a
cold start.
"""

from src.config import settings
from src.ingestion.corpus_cache import (
    _cache_is_valid,
    _fingerprints,
    _load_cache,
    _parser_signature,
)
from src.ingestion.pdf_parser import MultimodalFinancialParser

CACHED_CORPORA = [
    ("corpus_pdfs", "corpus_index.json"),
    ("domain_corpus_pdfs", "domain_corpus_pdfs_index.json"),
]


def test_checked_in_corpus_caches_match_the_current_parser() -> None:
    parser = MultimodalFinancialParser(
        chunk_size=settings.PARSER_CHUNK_SIZE,
        chunk_overlap=settings.PARSER_CHUNK_OVERLAP,
    )
    signature = _parser_signature(parser)
    data_dir = settings.BASE_DIR / "data"

    stale = [
        cache_name
        for corpus_name, cache_name in CACHED_CORPORA
        if not _cache_is_valid(
            _load_cache(data_dir / cache_name),
            signature,
            _fingerprints(sorted((data_dir / corpus_name).glob("*.pdf"))),
        )
    ]

    assert not stale, (
        f"stale checked-in corpus caches: {', '.join(stale)}. Regenerate them in the same change; "
        "a stale cache turns every cold start into a full corpus re-parse."
    )
