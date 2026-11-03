"""KFP v2 / Vertex AI pipeline for corpus ingestion, indexing, and evaluation.

The pipeline uses the project image so every component executes the same parser,
retriever, and evaluation code as the API. Compile with:

    pip install -r requirements-mlops.txt
    python k8s/vertex_ai_pipeline.py --compile --output vectracite_pipeline.yaml
"""

import argparse
from typing import Dict

try:
    from kfp import compiler, dsl
except ModuleNotFoundError:  # Local API/CI installs do not need the MLOps extra.
    compiler = None
    dsl = None

PIPELINE_IMAGE = "ghcr.io/abood991b/vectracite-ai:latest"
PIPELINE_NAME = "vectracite-financebench-rag-mlops-pipeline"


def get_vertex_pipeline_specification() -> Dict[str, str]:
    """Return the pipeline contract without requiring the optional KFP SDK."""
    return {
        "pipeline_name": PIPELINE_NAME,
        "sdk": "kfp>=2.7.0 / google-cloud-aiplatform",
        "stages": (
            "ingest_corpus -> build_qdrant_index -> evaluate_and_gate"
        ),
        "quality_gate": "recall_at_3 >= 0.90",
        "compile": "python k8s/vertex_ai_pipeline.py --compile --output pipeline.yaml",
    }


if dsl is not None:

    @dsl.container_component
    def ingest_corpus(corpus_dir: str, chunks: dsl.Output[dsl.Artifact]):
        """Parse the corpus and emit a portable JSON chunk artifact."""
        return dsl.ContainerSpec(
            image=PIPELINE_IMAGE,
            command=["python", "scripts/pipeline_ingest.py"],
            args=["--corpus-dir", corpus_dir, "--output", chunks.path],
        )


    @dsl.container_component
    def build_qdrant_index(
        chunks: dsl.Input[dsl.Artifact],
        qdrant_host: str,
        qdrant_port: str,
        collection: str,
        report: dsl.Output[dsl.Artifact],
    ):
        """Rebuild the shared Qdrant index from the ingested chunk artifact."""
        return dsl.ContainerSpec(
            image=PIPELINE_IMAGE,
            command=["python", "scripts/pipeline_index.py"],
            args=[
                "--chunks",
                chunks.path,
                "--qdrant-host",
                qdrant_host,
                "--qdrant-port",
                qdrant_port,
                "--collection",
                collection,
                "--output",
                report.path,
            ],
        )


    @dsl.container_component
    def evaluate_and_gate(
        qdrant_host: str,
        qdrant_port: str,
        collection: str,
        min_recall_at_3: float,
        report: dsl.Output[dsl.Artifact],
    ):
        """Run the retrieval ablation and fail the pipeline below the quality gate."""
        return dsl.ContainerSpec(
            image=PIPELINE_IMAGE,
            command=["python", "scripts/pipeline_evaluate.py"],
            args=[
                "--qdrant-host",
                qdrant_host,
                "--qdrant-port",
                qdrant_port,
                "--collection",
                collection,
                "--min-recall-at-3",
                str(min_recall_at_3),
                "--output",
                report.path,
            ],
        )


    @dsl.pipeline(
        name=PIPELINE_NAME,
        description="Ingest filings, rebuild Qdrant, and gate on retrieval recall.",
    )
    def vectracite_pipeline(
        corpus_dir: str = "/app/data/corpus_pdfs",
        qdrant_host: str = "qdrant-service",
        qdrant_port: str = "6333",
        collection: str = "financebench_hybrid_v1",
        min_recall_at_3: float = 0.90,
    ):
        ingest_task = ingest_corpus(corpus_dir=corpus_dir)
        index_task = build_qdrant_index(
            chunks=ingest_task.outputs["chunks"],
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port,
            collection=collection,
        )
        evaluate_task = evaluate_and_gate(
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port,
            collection=collection,
            min_recall_at_3=min_recall_at_3,
        )
        evaluate_task.after(index_task)


def compile_pipeline(output_path: str = "vectracite_pipeline.yaml") -> str:
    """Compile the KFP v2 pipeline; raises when the optional MLOps extra is absent."""
    if compiler is None or dsl is None:
        raise RuntimeError(
            "KFP is not installed. Run: pip install -r requirements-mlops.txt"
        )
    compiler.Compiler().compile(
        pipeline_func=vectracite_pipeline,
        package_path=output_path,
    )
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--output", default="vectracite_pipeline.yaml")
    args = parser.parse_args()
    if args.compile:
        print(f"Compiled pipeline to {compile_pipeline(args.output)}")
    else:
        print("Vertex AI / Kubeflow Pipeline Specification:")
        print(get_vertex_pipeline_specification())
