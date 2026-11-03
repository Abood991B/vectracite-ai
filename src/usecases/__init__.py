"""Application use cases that orchestrate domain objects without framework or vendor details.

Import submodules directly (``from src.usecases.answer_pipeline import ...``). This package
intentionally re-exports nothing: eager imports here create a cycle because adapters import the
query-normalization helpers while the use-case modules import those adapters' ports.
"""
