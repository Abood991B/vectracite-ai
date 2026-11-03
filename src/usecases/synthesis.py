"""Deterministic, citation-grounded answer synthesis.

This is application policy: it decides which evidence supports a question, how tables and sections are
formatted, and when an answer is out of scope. It runs entirely locally and depends only on domain
types and pure parsing helpers — no HTTP client, no settings, no provider.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Dict, Iterable, List

from src.domain.attributes import (
    FINANCIAL_METRIC_RE,
    asks_for_people_count,
    is_count_question,
    person_count_value,
)
from src.domain.documents import RetrievedDocument
from src.domain.financial_statements import is_statement_heading
from src.domain.sections import SECTION_HEADER_REGEX
from src.retrieval.table_ast import TableASTParser


QUERY_STOPWORDS = {
    "a",
    "about",
    "according",
    "an",
    "and",
    "are",
    "at",
    "by",
    "compared",
    "did",
    "does",
    "for",
    "from",
    "give",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "overview",
    "tell",
    "the",
    "their",
    "this",
    "to",
    "what",
    "was",
    "were",
    "which",
    "with",
}
QUERY_TERM_SYNONYMS: Dict[str, set[str]] = {
    "cashflow": {"cash", "flow"},
    "freecashflow": {"free", "cash", "flow"},
    "fcf": {"free", "cash", "flow"},
    "capex": {"capital", "expenditure"},
    "ppne": {"property", "plant", "equipment"},
    "operatingmargin": {"operating", "margin"},
    "netincome": {"net", "income"},
    "netsales": {"net", "sales"},
    "totalrevenue": {"total", "revenue"},
    "revenue": {"sales"},
    "sales": {"revenue"},
    # Casual verbs people use instead of "revenue" / "income".
    "make": {"revenue", "sales"},
    "made": {"revenue", "sales"},
    "makes": {"revenue", "sales"},
    "making": {"revenue", "sales"},
    "earn": {"revenue", "income"},
    "earned": {"revenue", "income"},
    "earns": {"revenue", "income"},
    "generate": {"revenue", "sales"},
    "generated": {"revenue", "sales"},
    "bring": {"revenue", "sales"},
    "brought": {"revenue", "sales"},
    "sell": {"sales", "revenue"},
    "sold": {"sales", "revenue"},
    "topline": {"revenue", "sales"},
    "bottomline": {"income", "profit"},
    # Headcount vocabulary: the gold sentence says "employed 29,239 people", not "employee count".
    "employee": {"employees", "employed", "people", "workforce", "headcount", "staff"},
    "employees": {"employed", "people", "workforce", "headcount", "staff"},
    "headcount": {"employees", "employed", "people", "workforce"},
    "workforce": {"employees", "employed", "people"},
}
SECTION_ALIASES: Dict[str, tuple[str, ...]] = {
    "summary": ("professional summary", "executive summary", "summary", "profile", "overview"),
    "skills": ("technical skills", "skills", "core competencies", "technologies"),
    "experience": ("work experience", "experience", "employment"),
    "projects": ("key projects", "projects"),
    "education": ("education", "academic"),
    "certifications": ("certifications", "certificates", "awards"),
}
TABLE_INTENT_RE = re.compile(
    r"\b(table|row|column|revenue|sales|income|profit|margin|capex|"
    r"capital expenditure|expense|cash flow|cashflow|assets?|liabilit\w*|"
    r"ratio|earnings|dividend|debt|turnover|growth|percent|percentage|"
    r"top\s+line|bottom\s+line|topline|bottomline|"
    r"fy\s?\d{2,4}|\d{4}|%)\b",
    flags=re.IGNORECASE,
)
# A row that starts with a revenue or income statement line is stronger evidence than an expense
# or segment table that merely mentions the same word ("Sales and marketing"). The family is chosen
# by what the question asks for, so "net income" rows cannot satisfy a "net sales" question.
REVENUE_STATEMENT_RE = re.compile(
    r"^(net sales|total net sales|total revenues?|net revenues?)\b",
    flags=re.IGNORECASE,
)
INCOME_STATEMENT_RE = re.compile(
    r"^(gross profit|operating income|net income|operating profit|net earnings)\b",
    flags=re.IGNORECASE,
)
# A period is a column selector, not a metric. Counting a segment table's "2022" header as
# query-term overlap promotes a segment row over the statement total the question asks for. The
# same applies to a date inside a row label: "Balance at December 31, 2019 (Total)" matches the
# date, not the metric.
PERIOD_COLUMN_RE = re.compile(r"^(?:19|20)\d{2}$")
PERIOD_TERM_RE = re.compile(
    r"^(?:(?:19|20)\d{2}|january|february|march|april|may|june|july|august|september|"
    r"october|november|december|\d{1,2})$",
    flags=re.IGNORECASE,
)
# The consolidated total is the reported company figure; segment and expense rows repeat the
# same metric words without being it.
TOTAL_ROW_RE = re.compile(r"^(?:total|consolidated|combined)\b", flags=re.IGNORECASE)
REVENUE_FAMILY_RE = re.compile(r"\b(sales|revenues?|turnover)\b", flags=re.IGNORECASE)
INCOME_FAMILY_RE = re.compile(r"\b(income|profit|earnings)\b", flags=re.IGNORECASE)
# "Total cost of sales" contains the word "sales" without being revenue, so it cannot take the
# consolidated-total bonus on a revenue question.
COST_ROW_RE = re.compile(r"\bcost of\b", flags=re.IGNORECASE)
NUMBER_TOKEN_RE = re.compile(r"\d[\d,.]*\d")


def wants_table_evidence(question: str) -> bool:
    """True when table synthesis should run for this question.

    A count question that names no statement metric ("employee count in FY2022") is a narrative
    fact question; TABLE_INTENT_RE would otherwise match the bare year and answer it from a table.
    """
    if asks_for_people_count(question):
        return bool(FINANCIAL_METRIC_RE.search(question))
    return bool(TABLE_INTENT_RE.search(question))


# A line that states the asked-for quantity outranks one that merely repeats the attribute word,
# so the bonus is larger than any realistic term-overlap count on a single line.
COUNT_STATEMENT_BONUS = 10
META_QUESTION_RE = re.compile(
    r"\b(what model|which model|who are you|what are you|how do you work|"
    r"how does (?:this|it) work|what llm|privacy|is my data private|"
    r"local execution)\b",
    flags=re.IGNORECASE,
)
DOCUMENTS_QUESTION_RE = re.compile(
    r"\b(what documents|which documents|what files|which files).*\b"
    r"(loaded|indexed|available)\b",
    flags=re.IGNORECASE,
)
META_RESPONSE = (
    "VectraCite-AI retrieves document passages using dense and lexical search, "
    "reranks them, and produces a cited answer with numeric checks. "
    "The default synthesizer is local. When a remote LLM or OCR service is enabled, "
    "document content is sent to that configured service."
)


CANONICAL_FINANCIAL_TERMS = frozenset(
    {
        "revenue",
        "sales",
        "income",
        "profit",
        "margin",
        "debt",
        "cash",
        "flow",
        "assets",
        "liabilities",
        "expenses",
        "employees",
    }
)


def expand_query(question: str) -> str:
    """Append canonical financial terms so casual wording still retrieves the right evidence.

    Only applied when a casual term is recognised, so a question already using the domain
    vocabulary passes through unchanged.
    """
    folded = question.casefold().replace("top line", "topline").replace(
        "bottom line", "bottomline"
    )
    additions: set[str] = set()
    for word in re.findall(r"[\w]+", folded, flags=re.UNICODE):
        if word in CANONICAL_FINANCIAL_TERMS:
            continue
        additions.update(QUERY_TERM_SYNONYMS.get(word, set()))
        fiscal_year = re.fullmatch(r"fy(\d{2,4})", word)
        if fiscal_year:
            digits = fiscal_year.group(1)
            additions.add(digits if len(digits) == 4 else f"20{digits}")
    if not additions:
        return question
    return f"{question} {' '.join(sorted(additions))}"


COMPARISON_RE = re.compile(
    r"\b(compare|comparison|between|across|versus|vs\.?|both|each)\b",
    flags=re.IGNORECASE,
)


def is_comparison_question(question: str) -> bool:
    """True when the question asks about more than one document or entity."""
    return bool(COMPARISON_RE.search(question))


def mentioned_documents(question: str, indexed_docs: list[str]) -> list[str]:
    """Return indexed documents named in the question, preferring a matching filing year."""
    folded = question.casefold()
    matches = [
        doc_name
        for doc_name in indexed_docs
        if re.search(
            rf"\b{re.escape(doc_name.split('_', maxsplit=1)[0].casefold())}\b",
            folded,
        )
    ]
    years = set(re.findall(r"\b(?:19|20)\d{2}\b", folded))
    if not years or not matches:
        return matches
    dated = [doc_name for doc_name in matches if any(year in doc_name for year in years)]
    entities = {doc_name.split("_", maxsplit=1)[0] for doc_name in matches}
    if len(entities) == 1:
        return dated or matches
    # Drop a filing only when its entity has a year-matching filing; keep entities that do not.
    dated_entities = {doc_name.split("_", maxsplit=1)[0] for doc_name in dated}
    return [
        doc_name
        for doc_name in matches
        if doc_name in dated
        or doc_name.split("_", maxsplit=1)[0] not in dated_entities
    ]


def query_terms(question: str) -> set[str]:
    """Return the normalized terms used for table and narrative matching."""
    cleaned = re.sub(r"['\u2019]s\b", "", question.lower())
    cleaned = cleaned.replace("top line", "topline").replace(
        "bottom line", "bottomline"
    )
    words = re.findall(r"[\w]+", cleaned, flags=re.UNICODE)
    terms: set[str] = set()
    for word in words:
        if len(word) > 2 and word not in QUERY_STOPWORDS:
            terms.add(word)
        parts = re.findall(r"[a-z]+|\d+", word)
        if len(parts) > 1:
            terms.update(
                part
                for part in parts
                if len(part) > 2 and part not in QUERY_STOPWORDS
            )
        terms.update(QUERY_TERM_SYNONYMS.get(word, set()))
        fiscal_year = re.fullmatch(r"fy(\d{2,4})", word)
        if fiscal_year:
            digits = fiscal_year.group(1)
            terms.add(digits if len(digits) == 4 else f"20{digits}")
    return terms


def narrative_sentences(text: str) -> List[str]:
    """Rebuild sentences from extracted page text.

    Extraction wraps at the column edge and hyphenates there, so a line is not a sentence:
    "The AI RMF Core provides ... and" / "activities to manage ... As illus-" / "trated in
    Figure 5 ..." is one sentence. Scoring lines answers with fragments, so lines are rejoined
    first: a trailing hyphen closes up, anything else joins with a space, and the result is split
    at sentence punctuation.

    Table markup is dropped before joining. It carries no prose, and joining it to the paragraph
    above produces one giant "sentence" that later filters discard, taking the prose with it.
    """
    sentences: List[str] = []
    buffer: List[str] = []

    def flush() -> None:
        if not buffer:
            return
        block = " ".join(buffer)
        sentences.extend(
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", block)
            if sentence.strip()
        )
        buffer.clear()

    for raw_line in text.splitlines():
        line = " ".join(raw_line.split()).strip()
        if not line:
            flush()
            continue
        if line.startswith("|") or line.startswith("EXTRACTED TABLES"):
            continue
        if buffer and buffer[-1].endswith("-"):
            buffer[-1] = buffer[-1][:-1] + line
        else:
            buffer.append(line)
    flush()
    return sentences


class LocalSynthesizer:
    """Build the deterministic answer a local deployment serves without a provider."""

    def answer(
        self,
        question: str,
        chunks: List[RetrievedDocument],
        indexed_docs: List[str] | None = None,
    ) -> str:
        """Select evidence, then prefer sections, then tables, then narrative."""
        answer_chunks = self._select_answer_chunks(question, chunks, indexed_docs)
        if is_comparison_question(question):
            comparison = self._comparison_answer(question, chunks, indexed_docs)
            if comparison:
                return comparison
        section_answer = self._format_sections(question, answer_chunks)
        if section_answer:
            return section_answer

        table_answer = self._table_answer(question, answer_chunks)
        if table_answer:
            return table_answer

        narrative_answer = self._general_answer(question, answer_chunks)
        return narrative_answer or self.out_of_scope(question)

    def out_of_scope(self, question: str) -> str:
        """Return the standard message for a question the evidence cannot answer."""
        return self._out_of_scope_message(question)

    def build_evidence(self, question: str, chunks: List[RetrievedDocument]) -> str:
        """Render the evidence block a remote provider receives."""
        return self._build_evidence(question, chunks)
    @classmethod
    def _is_cross_document_request(
        cls, question: str, indexed_docs: List[str] | None = None
    ) -> bool:
        if not re.search(
            r"\b(compare|comparison|between|across|versus|vs\.?|both|each)\b",
            question,
            flags=re.IGNORECASE,
        ):
            return False

        question_folded = question.casefold()
        question_compact = re.sub(r"[^a-z0-9]", "", question_folded)
        docs = list(indexed_docs or [])
        exact_documents = {
            doc_name
            for doc_name in docs
            if re.search(rf"\b{re.escape(doc_name.casefold())}\b", question_folded)
        }
        explicit_documents = set(exact_documents)
        if not exact_documents:
            for doc_name in docs:
                alias = doc_name.split("_", maxsplit=1)[0].casefold()
                compact = re.sub(r"[^a-z0-9]", "", alias)
                if (len(alias) >= 2 and re.search(rf"\b{re.escape(alias)}\b", question_folded)) or (
                    compact and compact in question_compact
                ):
                    explicit_documents.add(doc_name)
        return len(explicit_documents) >= 2 or bool(
            re.search(r"\b(documents?|files?|reports?|filings?|pdfs?)\b", question, re.I)
        )
    @classmethod
    def _select_answer_chunks(
        cls,
        question: str,
        chunks: List[RetrievedDocument],
        indexed_docs: List[str] | None = None,
    ) -> List[RetrievedDocument]:
        """Keep normal answers on the primary hit; admit only qualified comparisons."""
        if not chunks:
            return []
        primary_doc = chunks[0].doc_name
        selected = [chunk for chunk in chunks if chunk.doc_name == primary_doc]
        if not cls._is_cross_document_request(question, indexed_docs):
            return selected

        terms = query_terms(question)
        for chunk in chunks:
            if chunk.doc_name == primary_doc or not terms:
                continue
            chunk_terms = set(
                re.findall(r"[\w]+", chunk.text.casefold(), flags=re.UNICODE)
            )
            coverage = len(terms & chunk_terms) / len(terms)
            score = float(chunk.rerank_score or 0.0)
            if score >= 0.60 and coverage >= 0.50:
                selected.append(chunk)
        return selected
    @staticmethod
    def _out_of_scope_message(question: str) -> str:
        return (
            f"No matching information for '{question}' was found in the selected PDF documents. "
            "Try asking about a topic covered in your active PDFs or upload a relevant PDF file "
            "in the sidebar."
        )
    @staticmethod
    def _clean_lines(chunk: RetrievedDocument) -> List[str]:
        return [
            line.strip()
            for line in chunk.text.splitlines()
            if line.strip() and not line.strip().startswith("[DOCUMENT:")
        ]
    @staticmethod
    def _canonical_section(line: str) -> str | None:
        normalized = re.sub(r"[^a-z0-9]+", " ", line.casefold()).strip()
        if not normalized or not SECTION_HEADER_REGEX.fullmatch(line):
            return None
        for section, aliases in SECTION_ALIASES.items():
            if normalized in aliases:
                return section
        return None
    @classmethod
    def _requested_sections(cls, question: str) -> set[str]:
        folded = question.casefold()
        requested = {
            section
            for section, aliases in SECTION_ALIASES.items()
            if any(alias in folded for alias in aliases)
        }
        return {"summary"} if "overview" in folded and not requested else requested
    @classmethod
    def _ordered_chunks(cls, chunks: Iterable[RetrievedDocument]) -> List[RetrievedDocument]:
        return sorted(
            chunks,
            key=lambda chunk: (
                chunk.doc_name,
                chunk.page_number,
            ),
        )
    @classmethod
    def _extract_sections(
        cls,
        chunks: List[RetrievedDocument],
        requested: set[str],
    ) -> Dict[str, List[tuple[str, int, List[str]]]]:
        sections: Dict[str, List[tuple[str, int, List[str]]]] = defaultdict(list)
        for doc_name in sorted({chunk.doc_name for chunk in chunks}):
            doc_chunks = cls._ordered_chunks(chunk for chunk in chunks if chunk.doc_name == doc_name)
            active_section: str | None = None
            active_page = 1
            active_lines: List[str] = []

            def flush_section() -> None:
                nonlocal active_lines
                if active_section in requested and active_lines:
                    sections[active_section].append((doc_name, active_page, list(active_lines)))
                active_lines = []

            for chunk in doc_chunks:
                for line in cls._clean_lines(chunk):
                    section = cls._canonical_section(line)
                    if section is not None:
                        flush_section()
                        active_section = section
                        active_page = chunk.page_number
                        active_lines = [line]
                    elif active_section in requested:
                        active_lines.append(line)
            flush_section()
        return sections
    @classmethod
    def _format_sections(
        cls,
        question: str,
        chunks: List[RetrievedDocument],
    ) -> str | None:
        requested = cls._requested_sections(question)
        if not requested:
            return None
        sections = cls._extract_sections(chunks, requested)
        if not sections:
            return None

        parts: List[str] = []
        for section in ("summary", "skills", "experience", "projects", "education", "certifications"):
            for doc_name, page_number, lines in sections.get(section, []):
                parts.append(f"[{doc_name}, Page {page_number}]\n" + "\n".join(lines))
        return "\n\n".join(parts)

    def _comparison_answer(
        self,
        question: str,
        chunks: List[RetrievedDocument],
        indexed_docs: List[str] | None = None,
    ) -> str:
        """Answer a cross-document question with one cited block per mentioned document."""
        grouped: Dict[str, List[RetrievedDocument]] = {}
        for chunk in self._ordered_chunks(chunks):
            grouped.setdefault(chunk.doc_name, []).append(chunk)

        targets = mentioned_documents(question, indexed_docs or list(grouped))
        if targets:
            grouped = {name: grouped[name] for name in targets if name in grouped}
        missing = [name for name in targets if name not in grouped]
        if len(grouped) + len(missing) < 2:
            return ""

        blocks: List[str] = []
        for doc_chunks in grouped.values():
            table = self._table_answer(question, doc_chunks)
            if table:
                blocks.append("- " + table.replace("\n", "\n  "))
        for doc_name in missing:
            blocks.append(f"- [{doc_name}]: no matching evidence in the indexed corpus.")
        if len(blocks) < 2:
            return ""
        return "Comparison across documents:\n" + "\n".join(blocks)

    @classmethod
    def _statement_line_answer(
        cls,
        terms: set[str],
        chunks: List[RetrievedDocument],
        statement_pages: set[tuple[str, int]],
    ) -> tuple[str, int] | None:
        """The reporting line on a consolidated statement page that matches the question.

        Statement pages often extract as plain text with no markdown table to parse, so the
        metric is matched against the line itself rather than a cell row key. Any statement
        metric works - gross profit, research and development, operating cash flow, total
        assets - because the match is on the question's own terms, not a fixed family list.
        """
        best: tuple[tuple[int, int, int], RetrievedDocument, str] | None = None
        for chunk in chunks:
            if (chunk.doc_name, chunk.page_number) not in statement_pages:
                continue
            entity_terms = set(
                re.findall(r"[a-z0-9]+", chunk.doc_name.split("_", maxsplit=1)[0].lower())
            )
            metric_terms = {
                term
                for term in terms - entity_terms
                if term not in QUERY_STOPWORDS
                and len(term) >= 3
                and not PERIOD_TERM_RE.match(term)
            }
            if not metric_terms:
                continue
            for raw_line in chunk.text.splitlines():
                line = " ".join(raw_line.split())
                # Markdown rows are the table path's job; a header row names the metric and the
                # periods but states no figure, so only real values make a line a candidate.
                if not line or line.startswith("|"):
                    continue
                # "Cost of sales" shares a synonym with revenue without being revenue. Only a
                # question that asks for cost may be answered by a cost row.
                if COST_ROW_RE.search(line) and "cost" not in metric_terms:
                    continue
                values = [
                    token
                    for token in NUMBER_TOKEN_RE.findall(line)
                    if not PERIOD_COLUMN_RE.match(token)
                ]
                if not values:
                    continue
                overlap = len(
                    metric_terms & set(re.findall(r"[a-z0-9]+", line.lower()))
                )
                if overlap == 0:
                    continue
                # A statement row is the asked-for label followed by its periods' values, so
                # penalise words the question did not ask for: "Total current assets" is shorter
                # than "Total assets $ 19,845 $ 17,890" but adds a metric word the question
                # never used.
                label_terms = {
                    token
                    for token in re.findall(r"[a-z]+", line.lower())
                    if token not in QUERY_STOPWORDS
                }
                score = (
                    overlap,
                    -len(label_terms - metric_terms),
                    -len(line.split()),
                    -chunk.page_number,
                )
                if best is None or score > best[0]:
                    best = (score, chunk, line)
        if best is None:
            return None
        (overlap, _, _, _), chunk, line = best
        return (
            f"From [{chunk.doc_name}, Page {chunk.page_number}]:\n- {line}",
            overlap,
        )

    @classmethod
    def _table_answer(
        cls, question: str, chunks: List[RetrievedDocument]
    ) -> str | None:

        if not wants_table_evidence(question):
            return None
        terms = query_terms(question)
        folded_question = question.casefold()
        wants_revenue = bool(
            re.search(
                r"\b(sales|revenues?|turnover|top ?line|topline)\b", folded_question
            )
        )
        wants_income = bool(
            re.search(r"\b(income|profit|earnings|margin|eps)\b", folded_question)
        )
        ordered = sorted(
            chunks,
            key=lambda chunk: (
                not chunk.contains_table,
                -(float(chunk.rerank_score or chunk.rrf_score or 0.0)),
            ),
        )
        best_score: tuple[float, float, float, float, float, float] | None = None
        best_chunk: RetrievedDocument | None = None
        best_cells = []
        statement_pages = {
            (chunk.doc_name, chunk.page_number)
            for chunk in ordered
            if chunk.chunk_index == 0 and is_statement_heading(chunk.text)
        }
        for chunk in ordered:
            matched_cells = TableASTParser.match_query_cells(
                f"{question} {' '.join(sorted(terms))}",
                chunk.text,
                doc_name=chunk.doc_name,
                page_number=chunk.page_number,
            )
            if not matched_cells:
                continue
            entity_terms = set(
                re.findall(r"[a-z0-9]+", chunk.doc_name.split("_", maxsplit=1)[0].lower())
            )
            metric_terms = {
                term
                for term in terms - entity_terms
                if term not in QUERY_STOPWORDS
                and len(term) >= 3
                and not PERIOD_TERM_RE.match(term)
            }
            # A cost row is not the metric, even when the synonym map ties them together, so
            # drop it before scoring and before rendering.
            if "cost" not in metric_terms:
                matched_cells = [
                    cell
                    for cell in matched_cells
                    if not COST_ROW_RE.search(cell.row_key)
                ]
                if not matched_cells:
                    continue
            best_cell_overlap = 0
            statement_rows: set[str] = set()
            has_total = False
            for cell in matched_cells:
                row_terms = set(re.findall(r"[a-z0-9]+", cell.row_key.lower()))
                header_terms = {
                    term
                    for term in re.findall(r"[a-z0-9]+", cell.col_header.lower())
                    if not PERIOD_COLUMN_RE.match(term)
                }
                best_cell_overlap = max(
                    best_cell_overlap,
                    len(metric_terms & (row_terms | header_terms)),
                )
                if (wants_revenue and REVENUE_STATEMENT_RE.search(cell.row_key)) or (
                    wants_income and INCOME_STATEMENT_RE.search(cell.row_key)
                ):
                    statement_rows.add(cell.row_key.casefold())
                if TOTAL_ROW_RE.match(cell.row_key.strip()) and not COST_ROW_RE.search(
                    cell.row_key
                ) and (
                    (wants_revenue and REVENUE_FAMILY_RE.search(cell.row_key))
                    or (wants_income and INCOME_FAMILY_RE.search(cell.row_key))
                ):
                    has_total = True
            if metric_terms and best_cell_overlap == 0:
                continue
            # Score the best single cell, not the union of all cells: an expense table with a
            # "Sales and marketing" row and a "Percentage of total revenue" row must not outrank
            # an income statement's "Total revenue" row. Overlap comes first so a segment table
            # whose row covers more query terms still wins, then a consolidated total row (the
            # reported figure, not a segment), then a statement line over a table that merely
            # mentions the metric, then a page that is itself a primary statement over a note or
            # MD&A table that restates it. Ties fall back to the parsed table, the retrieval
            # score, and finally the earlier page, so the choice does not depend on evidence order.
            score = (
                float(best_cell_overlap),
                1.0 if has_total else 0.0,
                float(len(statement_rows)),
                1.0
                if (chunk.doc_name, chunk.page_number) in statement_pages
                else 0.0,
                float(chunk.contains_table),
                float(chunk.rerank_score or chunk.rrf_score or 0.0),
                -float(chunk.page_number),
            )
            if best_score is None or score > best_score:
                best_score, best_chunk, best_cells = score, chunk, matched_cells
        # A note or an MD&A table restates the figure; the company's own statement is the source.
        # When the parsed-table winner is not a statement page, answer from the statement line.
        # When it is a statement page, a line that matches the asked-for metric more closely still
        # wins: "Total assets" beats the statement of changes in equity's "Balance ... (Total)".
        statement_line = cls._statement_line_answer(terms, ordered, statement_pages)
        if statement_line:
            line_answer, line_overlap = statement_line
            table_overlap = best_score[0] if best_score else -1.0
            if (
                best_chunk is None
                or (best_chunk.doc_name, best_chunk.page_number) not in statement_pages
                or line_overlap > table_overlap
            ):
                return line_answer
        if best_chunk is None:
            return None
        cell_lines = [
            f"- {cell.row_key} ({cell.col_header}): {cell.value}"
            for cell in best_cells[:8]
        ]
        delta_lines = [
            f"- Period change - {delta['row_key']} ({delta['comparison']}): "
            f"{delta['absolute_change']} ({delta['pct_change']})"
            for delta in TableASTParser.compute_row_deltas(best_cells)[:4]
        ]
        answer = (
            f"From [{best_chunk.doc_name}, Page {best_chunk.page_number}]:\n"
            + "\n".join(cell_lines)
        )
        if delta_lines:
            answer += "\n\nCalculated comparisons:\n" + "\n".join(delta_lines)
        return answer
    @classmethod
    def _narrative_lines(
        cls,
        question: str,
        chunks: List[RetrievedDocument],
    ) -> Dict[str, tuple[RetrievedDocument, List[str]]]:
        terms = query_terms(question)
        count_question = is_count_question(question)
        # The "[DOCUMENT: ...]" header belongs to the chunk, not to the first sentence: left in
        # place it makes that sentence start with "[" and get dropped as metadata.
        candidates = [
            (
                chunk,
                narrative_sentences(
                    re.sub(r"^\[DOCUMENT:[^\]]*\]\s*", "", chunk.text.strip())
                ),
            )
            # Keep the caller's order: it is the retrieval ranking, and it breaks ties between
            # pages that match the question equally well.
            for chunk in chunks
        ]
        weights = cls._term_weights(terms, candidates)
        grouped: Dict[str, tuple[RetrievedDocument, List[str]]] = {}
        scores: Dict[str, float] = {}
        for chunk, sentences in candidates:
            scored_lines: List[tuple[float, int, str]] = []
            for line_index, line in enumerate(sentences):
                if not line or line.startswith(("[", "---", "|")):
                    continue
                raw_tokens = re.findall(r"[a-z0-9]+", line.lower())
                if len(raw_tokens) < 3 and not re.search(r"\d", line):
                    continue
                if line.isupper() and not re.search(r"\d", line):
                    continue
                line_terms = set(raw_tokens)
                overlap = sum(
                    weights.get(term, 0.0) for term in terms & line_terms
                )
                # A count question is answered by the sentence that states the quantity, not by the
                # most verbose page that repeats the attribute: stock-plan boilerplate mentions
                # employees more often than the human-capital sentence stating the number.
                if overlap and count_question and person_count_value(line):
                    overlap += COUNT_STATEMENT_BONUS
                if overlap:
                    scored_lines.append((overlap, line_index, line))
            if not scored_lines:
                continue
            selected = sorted(scored_lines, key=lambda item: item[0], reverse=True)[:5]
            selected.sort(key=lambda item: item[1])
            # Rank a page by its best sentence, not by the sum over its sentences: summing rewards
            # a long discussion page over the syllabus line that actually answers the question.
            chunk_score = max(item[0] for item in selected)
            if chunk.doc_name not in grouped or chunk_score > scores[chunk.doc_name]:
                grouped[chunk.doc_name] = (chunk, [item[2] for item in selected])
                scores[chunk.doc_name] = chunk_score
        return grouped

    @staticmethod
    def _term_weights(
        terms: set[str],
        candidates: List[tuple[RetrievedDocument, List[str]]],
    ) -> Dict[str, float]:
        """Weight query terms by how rare they are in the candidate text.

        Every page of a court opinion mentions "public", so counting it as heavily as "CADA" or
        "accommodations" hands the citation to the page that repeats the common word. Weighting by
        local inverse sentence frequency favours the sentence that uses the words this question is
        actually made of.
        """
        containing: Dict[str, int] = {term: 0 for term in terms}
        for _, sentences in candidates:
            seen: set[str] = set()
            for sentence in sentences:
                seen.update(re.findall(r"[a-z0-9]+", sentence.lower()))
            for term in terms & seen:
                containing[term] += 1
        documents = len(candidates) or 1
        return {
            term: 1.0 + math.log(documents / max(count, 1))
            for term, count in containing.items()
        }
    @classmethod
    def _general_answer(cls, question: str, chunks: List[RetrievedDocument]) -> str:
        grouped = cls._narrative_lines(question, chunks)
        if len(grouped) > 1:
            parts = []
            for chunk, lines in grouped.values():
                parts.append(
                    f"- [{chunk.doc_name}, Page {chunk.page_number}]\n  " + "\n  ".join(lines[:2])
                )
            return "Findings by document:\n" + "\n".join(parts)
        if grouped:
            chunk, lines = next(iter(grouped.values()))
            return f"According to [{chunk.doc_name}, Page {chunk.page_number}]:\n" + "\n".join(lines[:6])

        top = chunks[0]
        top_score = float(top.rerank_score if top.rerank_score is not None else top.rrf_score)
        if top_score >= 0.30:
            cleaned = re.sub(r"^\[DOCUMENT:[^\]]+\]\s*", "", top.text.strip())
            excerpt = " ".join(cleaned.split())[:350]
            return f"Based on [{top.doc_name}, Page {top.page_number}], {excerpt}"
        return ""
    @staticmethod
    def _build_evidence(question: str, chunks: List[RetrievedDocument]) -> str:
        """Build a bounded, citation-labelled evidence packet for a remote model."""
        blocks: List[str] = []
        remaining = 24_000
        for chunk in chunks:
            header = f"[{chunk.doc_name}, Page {chunk.page_number}]"
            body = chunk.text.strip()
            block = f"{header}\n{body}"
            if len(block) > remaining:
                block = block[:remaining]
            if not block:
                continue
            blocks.append(block)
            remaining -= len(block)
            if remaining <= 0:
                break
        return "QUESTION:\n{}\n\nEVIDENCE:\n{}".format(
            question, "\n\n".join(blocks)
        )
