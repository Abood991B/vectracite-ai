import logging
import math
import re
from typing import List

from src.config import settings
from src.retrieval.hybrid_retriever import RetrievedDocument
from src.retrieval.table_ast import TableASTParser

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Adaptive SCAR™ (Speculative Cell-Coordinate AST Reranker):
      1. Computes 2D Table-AST Schema Alignment over (RowKey, ColHeader) coordinates.
      2. Adaptive Margin Gate: When the top candidate achieves a structural AST margin
         advantage (delta_ast >= 0.35 and ast_align >= 0.50) over all competing chunks,
         gates neural Cross-Encoder verification to 1 micro-compressed pass (<45 tokens),
         cutting CPU reranking latency while preserving table precision.
      3. Fuses calibrated Cross-Encoder probability with 2D AST alignment and RRF prior.
    """

    def __init__(
        self,
        model_name: str = settings.RERANKER_MODEL,
        use_lightweight_scorer: bool = False,
        speculative_top_m: int = 2,
        margin_threshold: float = 0.35,
    ) -> None:
        self.model_name = model_name
        self.use_lightweight_scorer = use_lightweight_scorer
        self.speculative_top_m = speculative_top_m
        self.margin_threshold = margin_threshold
        self._cross_encoder = None

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-max(min(x, 20.0), -20.0)))

    def rerank(
        self, query: str, candidates: List[RetrievedDocument], top_n: int = 5
    ) -> List[RetrievedDocument]:
        """Executes Adaptive SCAR speculative reranking over Hybrid RRF candidates."""
        if not candidates:
            return []

        max_rrf = max((c.rrf_score for c in candidates), default=1.0) or 1.0
        ast_scores = [TableASTParser.compute_ast_alignment_score(query, c.text) for c in candidates]

        preliminary = []
        for doc, ast_align in zip(candidates, ast_scores):
            rrf_norm = doc.rrf_score / max_rrf
            fast_prior = (0.65 * ast_align) + (0.35 * rrf_norm)
            preliminary.append((fast_prior, ast_align, rrf_norm, doc))

        preliminary.sort(key=lambda item: item[0], reverse=True)

        if not self.use_lightweight_scorer:
            try:
                if self._cross_encoder is None:
                    from sentence_transformers import CrossEncoder

                    self._cross_encoder = CrossEncoder(self.model_name, max_length=32)

                # Adaptive Margin Gate: if candidate #1 has a strong 2D Table-AST lock,
                # evaluate only 1 neural pass instead of speculative_top_m passes.
                top_ast = preliminary[0][1]
                second_ast = preliminary[1][1] if len(preliminary) > 1 else 0.0
                ast_margin = top_ast - second_ast
                active_m = (
                    1
                    if top_ast >= 0.50 and ast_margin >= self.margin_threshold
                    else min(self.speculative_top_m, len(preliminary))
                )

                top_slice = preliminary[:active_m]
                compressed_pairs = [
                    (query, TableASTParser.compress_for_cross_encoder(query, item[3].text))
                    for item in top_slice
                ]
                raw_logits = self._cross_encoder.predict(compressed_pairs)

                for idx, (fast_prior, ast_align, rrf_norm, doc) in enumerate(preliminary):
                    if idx < active_m:
                        logit_val = (
                            float(raw_logits[idx])
                            if hasattr(raw_logits, "__len__")
                            else float(raw_logits)
                        )
                        ce_prob = self._sigmoid(logit_val)
                        doc.rerank_score = round(
                            (0.40 * ce_prob) + (0.40 * ast_align) + (0.20 * rrf_norm), 6
                        )
                    else:
                        doc.rerank_score = round(fast_prior * 0.70, 6)

                ranked = sorted(
                    [item[3] for item in preliminary],
                    key=lambda d: d.rerank_score or -999.0,
                    reverse=True,
                )
                return ranked[:top_n]
            except Exception as exc:
                logger.warning("Falling back to deterministic AST+lexical scorer: %s", exc)
                self.use_lightweight_scorer = True

        q_tokens = set(re.findall(r"[a-zA-Z0-9_$\.%]+", query.lower())) - TableASTParser.STOPWORDS
        for _fast_prior, ast_align, rrf_norm, doc in preliminary:
            d_tokens = (
                set(re.findall(r"[a-zA-Z0-9_$\.%]+", doc.text.lower())) - TableASTParser.STOPWORDS
            )
            overlap = len(q_tokens & d_tokens) / max(len(q_tokens), 1)
            doc.rerank_score = round(
                (0.35 * overlap) + (0.45 * ast_align) + (0.20 * rrf_norm), 6
            )

        ranked = sorted(
            [item[3] for item in preliminary],
            key=lambda d: d.rerank_score or 0.0,
            reverse=True,
        )
        return ranked[:top_n]
