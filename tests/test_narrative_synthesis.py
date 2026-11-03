"""Pin narrative extraction and page selection for prose documents.

Financial answers come from table cells; prose answers come from sentences. Extraction wraps at
the column edge and hyphenates there, so a line is not a sentence, and a page that merely
discusses the topic can outrank the page that states it. These tests pin the reconstruction, the
exclusion of table markup, and the tie-break that keeps the retrieval ranking.
"""

from src.domain.documents import RetrievedDocument
from src.usecases.synthesis import LocalSynthesizer, narrative_sentences


def _chunk(page: int, text: str, score: float = 0.5) -> RetrievedDocument:
    return RetrievedDocument(
        chunk_id=f"narrative-{page}",
        doc_name="NIST_AI_RMF_1_0",
        page_number=page,
        text=text,
        contains_table=False,
        dense_score=score,
        bm25_score=score,
        rrf_score=score,
        rerank_score=score,
    )


def test_a_wrapped_hyphenated_heading_is_rejoined_into_one_sentence() -> None:
    text = (
        "The AI RMF Core provides outcomes and actions that enable dialogue, understanding, and\n"
        "activities to manage AI risks and responsibly develop trustworthy AI systems. As illus-\n"
        "trated in Figure 5, the Core is composed of four functions: GOVERN, MAP, MEASURE,\n"
        "and MANAGE."
    )

    sentences = narrative_sentences(text)

    assert (
        "As illustrated in Figure 5, the Core is composed of four functions: "
        "GOVERN, MAP, MEASURE, and MANAGE." in sentences
    )


def test_table_markup_is_not_folded_into_prose() -> None:
    text = (
        "Résumé · São Paulo 2026\n"
        "\n"
        "EXTRACTED TABLES (PAGE 1)\n"
        "|Résumé São Paulo 2026<br>·|Change<br>·|\n"
        "|---|---|\n"
        "|Revenue 42 million|12.5%|"
    )

    sentences = narrative_sentences(text)

    assert sentences == ["Résumé · São Paulo 2026"]


def test_equal_matches_keep_the_retrieval_ranking() -> None:
    """NIST p17 (ranked first) and p7 mention the characteristics equally often."""
    question = "What characteristics of trustworthy AI systems are listed in the AI RMF?"
    first = _chunk(
        17,
        "[DOCUMENT: NIST_AI_RMF_1_0 | PAGE: 17] Characteristics of trustworthy AI systems "
        "include: valid and reliable, safe, secure and resilient.",
        score=0.9,
    )
    second = _chunk(
        7,
        "[DOCUMENT: NIST_AI_RMF_1_0 | PAGE: 7] Next, AI risks and trustworthiness are analyzed, "
        "outlining the characteristics of trustworthy AI systems.",
        score=0.4,
    )

    grouped = LocalSynthesizer._narrative_lines(question, [first, second])

    chunk, _lines = grouped["NIST_AI_RMF_1_0"]
    assert chunk.page_number == 17


def test_a_better_sentence_still_beats_retrieval_order() -> None:
    """Retrieval order is a tie-break, not an override."""
    question = "What characteristics of trustworthy AI systems are listed in the AI RMF?"
    weak = _chunk(
        17,
        "[DOCUMENT: NIST_AI_RMF_1_0 | PAGE: 17] This section is about risk management.",
        score=0.9,
    )
    strong = _chunk(
        7,
        "[DOCUMENT: NIST_AI_RMF_1_0 | PAGE: 7] The characteristics of trustworthy AI systems "
        "listed in the AI RMF are valid and reliable, safe, secure and resilient, and fair.",
        score=0.4,
    )

    grouped = LocalSynthesizer._narrative_lines(question, [weak, strong])

    chunk, _lines = grouped["NIST_AI_RMF_1_0"]
    assert chunk.page_number == 7
