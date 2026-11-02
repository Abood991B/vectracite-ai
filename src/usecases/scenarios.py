"""Workbench scenario cards built from benchmark definitions and the indexed catalog.

Pure policy over data supplied by the caller: no settings, no retriever instance, no HTTP.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

from src.retrieval.table_ast import TableASTParser
from src.usecases.ports import RetrieverPort


def build_scenarios(
    *,
    definitions: Sequence[Mapping[str, Any]],
    retriever: RetrieverPort,
    live_cases: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Build scenario cards from benchmark JSON plus live table schemas."""
    indexed_docs = {document["doc_name"] for document in retriever.list_documents()}
    scenarios: List[Dict[str, Any]] = []
    covered_docs: set[str] = set()

    for definition in definitions:
        doc_name = str(definition["doc_name"])
        if doc_name not in indexed_docs:
            continue
        scenarios.append(
            {
                "id": f"benchmark:{doc_name}:{definition['page_number']}",
                "doc_name": doc_name,
                "page_number": int(definition["page_number"]),
                "question": str(definition["question"]),
                "source": "benchmark",
            }
        )
        covered_docs.add(doc_name)

    for case in live_cases:
        doc_name = str(case["doc_name"])
        if doc_name not in indexed_docs or doc_name in covered_docs:
            continue
        scenarios.append(
            {
                "id": f"live:{doc_name}:{case['page_number']}",
                "doc_name": doc_name,
                "page_number": int(case["page_number"]),
                "question": str(case["question"]),
                "source": "live_table_schema",
            }
        )
        covered_docs.add(doc_name)

    for doc_name in sorted(indexed_docs - covered_docs):
        chunks = retriever.get_document_chunks(doc_name)
        table_chunk = next((chunk for chunk in chunks if chunk["contains_table"]), None)
        if table_chunk:
            cells = TableASTParser.extract_cells(
                table_chunk["text"], doc_name, int(table_chunk["page_number"])
            )
            if cells:
                first_cell = cells[0]
                question = (
                    f"In {doc_name}, what was {first_cell.row_key} in "
                    f"{first_cell.col_header}?"
                )
            else:
                question = f"What does the indexed table in {doc_name} show?"
            page_number = int(table_chunk["page_number"])
        else:
            question = f"Summarize the indexed evidence in {doc_name}."
            page_number = int(chunks[0]["page_number"]) if chunks else 1

        scenarios.append(
            {
                "id": f"indexed:{doc_name}:{page_number}",
                "doc_name": doc_name,
                "page_number": page_number,
                "question": question,
                "source": "indexed_document",
            }
        )

    return scenarios
