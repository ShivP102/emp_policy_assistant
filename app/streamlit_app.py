from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import Settings, get_settings
from src.generation.answer_generator import AnswerGenerator, AnswerResult
from src.ingestion.pipeline import IngestionPipeline
from src.retrieval.pipeline import RetrievalPipeline
from src.utils.ingest_state import IngestState


def _load_openai_key(settings: Settings) -> Settings:
    if settings.openai_api_key:
        return settings

    try:
        secret_key = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        secret_key = ""

    if secret_key:
        return settings.model_copy(update={"openai_api_key": secret_key})
    return settings


@st.cache_resource(show_spinner="Loading models and vector store...")
def load_pipelines() -> tuple[IngestionPipeline, RetrievalPipeline]:
    settings = _load_openai_key(get_settings())
    ingest_pipeline = IngestionPipeline(settings)
    retrieval_pipeline = RetrievalPipeline(
        settings,
        ingest_pipeline.embedder,
        ingest_pipeline.store,
    )
    return ingest_pipeline, retrieval_pipeline


@st.cache_resource(show_spinner="Loading answer generator...")
def load_answer_generator(settings: Settings) -> AnswerGenerator:
    return AnswerGenerator(settings)


def ensure_index(ingest_pipeline: IngestionPipeline) -> None:
    if ingest_pipeline.is_stale():
        with st.spinner("Building policy index from PDFs..."):
            result = ingest_pipeline.run(force=False)
        if result.skipped:
            st.sidebar.info("Policy index is up to date.")
        else:
            st.sidebar.success(
                f"Indexed {result.chunks_stored} chunks from {result.files_processed} PDFs."
            )
    else:
        st.sidebar.info("Policy index is up to date.")


def render_sources(result: AnswerResult) -> None:
    if not result.sources:
        return

    with st.expander("Sources"):
        for index, source in enumerate(result.sources, start=1):
            st.markdown(
                f"**{index}. {source['source']}** (Page {source['page']})  \n"
                f"Rerank score: {source.get('rerank_score', 'N/A')}"
            )
            st.caption(source["excerpt"] + ("..." if len(source["excerpt"]) >= 300 else ""))


def main() -> None:
    st.set_page_config(
        page_title="Employee Policy Assistant",
        page_icon="📋",
        layout="wide",
    )
    st.title("Employee Policy Assistant")
    st.caption("Ask questions about company policies. Answers are grounded in indexed PDFs.")

    ingest_pipeline, retrieval_pipeline = load_pipelines()
    settings = ingest_pipeline.settings

    if not settings.openai_api_key:
        st.error(
            "OPENAI_API_KEY is not configured. Set it in `.env` or "
            "`.streamlit/secrets.toml` before asking questions."
        )

    with st.sidebar:
        st.header("Index Status")
        ensure_index(ingest_pipeline)

        chunk_count = ingest_pipeline.store.get_collection_count()
        st.metric("Indexed chunks", chunk_count)

        manifest = IngestState(settings).load_manifest()
        if manifest:
            st.caption(f"Last ingested: {manifest.get('ingested_at', 'Unknown')}")

        if st.button("Rebuild index", use_container_width=True):
            with st.spinner("Rebuilding index..."):
                result = ingest_pipeline.run(force=True)
            st.success(f"Rebuilt index with {result.chunks_stored} chunks.")
            st.rerun()

        st.divider()
        st.subheader("Models")
        st.text(f"Embeddings: {settings.embedding_model.split('/')[-1]}")
        st.text(f"Reranker: {settings.cross_encoder_model.split('/')[-1]}")
        st.text(f"LLM: {settings.openai_model}")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                render_sources(AnswerResult(answer="", sources=message["sources"]))

    prompt = st.chat_input("Ask a question about company policies...")
    if not prompt:
        return

    if not settings.openai_api_key:
        st.warning("Configure OPENAI_API_KEY to enable chat responses.")
        return

    answer_generator = load_answer_generator(settings)

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching policies..."):
            documents = retrieval_pipeline.retrieve(prompt)
            result = answer_generator.generate(prompt, documents)

        st.markdown(result.answer)
        render_sources(result)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result.answer,
            "sources": result.sources,
        }
    )


if __name__ == "__main__":
    main()
