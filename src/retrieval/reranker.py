import math
import re
from typing import Any, List
from sentence_transformers import CrossEncoder

from src.config import settings
from src.retrieval.hybrid_retriever import RetrievedDocument
from src.retrieval.table_ast import TableASTParser, named_entities

_SHARED_CROSS_ENCODER: CrossEncoder | None = None
_SHARED_LIGHTWEIGHT_CROSS_ENCODER: "LightweightCrossEncoder | None" = None


class LightweightCrossEncoder:
    """Small deterministic scorer used only when tests opt into lightweight mode."""

    @staticmethod
    def _tokens(value: str) -> set[str]:
        cleaned = re.sub(r"['’]s\b", "", value.lower())
        return {
            token
            for token in re.findall(r"[\w$\.%]+", cleaned, flags=re.UNICODE)
            if len(token) >= 2 or token.isdigit()
        }

    def predict(self, pairs: List[tuple[str, str]], show_progress_bar: bool = False) -> List[float]:
        del show_progress_bar
        scores: List[float] = []
        for query, text in pairs:
            query_tokens = self._tokens(query)
            text_tokens = self._tokens(text)
            overlap = len(query_tokens & text_tokens) / max(len(query_tokens), 1)
            table_signal = 0.8 if "|" in text else 0.0
            scores.append(round((overlap * 8.0) + table_signal - 2.0, 6))
        return scores


def get_shared_cross_encoder(
    model_name: str = settings.RERANKER_MODEL,
    use_lightweight: bool | None = None,
) -> Any:
    """Load either the production BGE reranker or the isolated CI substitute."""
    global _SHARED_CROSS_ENCODER, _SHARED_LIGHTWEIGHT_CROSS_ENCODER
    lightweight = settings.USE_LIGHTWEIGHT_MODELS if use_lightweight is None else use_lightweight
    if lightweight:
        if _SHARED_LIGHTWEIGHT_CROSS_ENCODER is None:
            _SHARED_LIGHTWEIGHT_CROSS_ENCODER = LightweightCrossEncoder()
        return _SHARED_LIGHTWEIGHT_CROSS_ENCODER
    if _SHARED_CROSS_ENCODER is None:
        _SHARED_CROSS_ENCODER = CrossEncoder(model_name)
    return _SHARED_CROSS_ENCODER


class CrossEncoderReranker:
    """
    SCAR™ (Speculative Cell-Coordinate AST Reranker).

    The reranker computes 2D Table-AST alignment, then applies a configurable
    Cross-Encoder gate. The default always_top2 gate scores the two strongest
    fast-prior candidates; adaptive and top-1/all gates remain available for
    ablation. Final scores fuse calibrated Cross-Encoder probability with AST
    alignment and the RRF prior.
    """

    def __init__(
        self,
        model_name: str = settings.RERANKER_MODEL,
        speculative_top_m: int = 2,
        margin_threshold: float = 0.35,
        use_lightweight_scorer: bool | None = None,
        force_cross_encoder: bool = False,
        cross_encoder_gate: str = settings.RERANKER_GATE,
        confidence_threshold: float = settings.RERANKER_CONFIDENCE_THRESHOLD,
    ) -> None:
        if cross_encoder_gate not in {
            "adaptive",
            "adaptive_top2",
            "always_top1",
            "always_top2",
            "always_all",
            "top1_then_top2",
        }:
            raise ValueError(
                "cross_encoder_gate must be adaptive, adaptive_top2, "
                "always_top1, always_top2, always_all, or top1_then_top2"
            )
        self.confidence_threshold = confidence_threshold
        self.model_name = model_name
        self.speculative_top_m = speculative_top_m
        self.margin_threshold = margin_threshold
        self.force_cross_encoder = force_cross_encoder or cross_encoder_gate == "always_all"
        self.cross_encoder_gate = (
            "always_all" if self.force_cross_encoder else cross_encoder_gate
        )
        self.use_lightweight_scorer = (
            settings.USE_LIGHTWEIGHT_MODELS
            if use_lightweight_scorer is None
            else use_lightweight_scorer
        )

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
        query_entities = named_entities(query, {c.doc_name for c in candidates})
        ast_scores = [
            TableASTParser.compute_ast_alignment_score(query, c.text, query_entities)
            for c in candidates
        ]
        ordered_ast = sorted(ast_scores, reverse=True)
        top_ast = ordered_ast[0] if ordered_ast else 0.0
        second_ast = ordered_ast[1] if len(ordered_ast) > 1 else 0.0
        if (
            self.cross_encoder_gate == "adaptive"
            and (top_ast < 0.50 or (top_ast - second_ast) < self.margin_threshold)
        ):
            for doc in candidates:
                doc.rerank_score = doc.rrf_score
            return candidates[:top_n]

        preliminary = []
        for doc, ast_align in zip(candidates, ast_scores):
            rrf_norm = doc.rrf_score / max_rrf
            fast_prior = (0.65 * ast_align) + (0.35 * rrf_norm)
            preliminary.append((fast_prior, ast_align, rrf_norm, doc))

        preliminary.sort(key=lambda item: item[0], reverse=True)

        cross_encoder = get_shared_cross_encoder(
            self.model_name,
            use_lightweight=self.use_lightweight_scorer,
        )

        # Adaptive Margin Gate: if candidate #1 has a strong 2D Table-AST lock,
        # evaluate only 1 neural pass instead of speculative_top_m passes.
        top_ast = preliminary[0][1]
        second_ast = preliminary[1][1] if len(preliminary) > 1 else 0.0
        ast_margin = top_ast - second_ast
        if self.cross_encoder_gate == "top1_then_top2":
            top_pair = (
                query,
                TableASTParser.compress_for_cross_encoder(
                    query, preliminary[0][3].text
                ),
            )
            raw_logits = list(
                cross_encoder.predict([top_pair], show_progress_bar=False)
            )
            active_m = 1
            if (
                len(preliminary) > 1
                and self._sigmoid(float(raw_logits[0]))
                < self.confidence_threshold
            ):
                second_pair = (
                    query,
                    TableASTParser.compress_for_cross_encoder(
                        query, preliminary[1][3].text
                    ),
                )
                raw_logits.extend(
                    cross_encoder.predict([second_pair], show_progress_bar=False)
                )
                active_m = 2
        else:
            if self.cross_encoder_gate == "always_all":
                active_m = len(preliminary)
            elif self.cross_encoder_gate == "always_top1":
                active_m = 1
            elif self.cross_encoder_gate in {"always_top2", "adaptive_top2"}:
                active_m = (
                    1
                    if (
                        self.cross_encoder_gate == "adaptive_top2"
                        and top_ast >= 0.50
                        and ast_margin >= self.margin_threshold
                    )
                    else min(self.speculative_top_m, len(preliminary))
                )
            else:
                active_m = (
                    1
                    if top_ast >= 0.50 and ast_margin >= self.margin_threshold
                    else min(self.speculative_top_m, len(preliminary))
                )
            top_slice = preliminary[:active_m]
            compressed_pairs = [
                (
                    query,
                    TableASTParser.compress_for_cross_encoder(query, item[3].text),
                )
                for item in top_slice
            ]
            raw_logits = list(
                cross_encoder.predict(compressed_pairs, show_progress_bar=False)
            )

        for idx, (fast_prior, ast_align, rrf_norm, doc) in enumerate(preliminary):
            if idx < active_m:
                logit_val = float(raw_logits[idx])
                ce_prob = self._sigmoid(logit_val)
                doc.rerank_score = round(
                    (0.40 * ce_prob) + (0.40 * ast_align) + (0.20 * rrf_norm), 6
                )
            else:
                doc.rerank_score = round(fast_prior * 0.70, 6)

        ranked = sorted(
            [item[3] for item in preliminary],
            key=lambda d: d.rerank_score or 0.0,
            reverse=True,
        )
        return ranked[:top_n]
