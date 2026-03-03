import hashlib
import os
from pathlib import Path
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import get_settings
from app.retrieval.embedder import LocalEmbedder
from app.retrieval.vector_store import ChromaVectorStore


def iter_markdown_files(data_dir: Path) -> List[Path]:
    return [p for p in data_dir.rglob("*.md") if p.is_file()]


def compute_id(text: str, source: str) -> str:
    """Deterministic ID for a chunk based on content and source."""
    h = hashlib.sha256()
    h.update(source.encode("utf-8"))
    h.update(b"||")
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def ingest() -> None:
    settings = get_settings()

    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "backend" / "app" / "data"
    if not data_dir.exists():
        raise RuntimeError(f"Data directory does not exist: {data_dir}")

    md_files = iter_markdown_files(data_dir)
    if not md_files:
        raise RuntimeError(f"No Markdown files found in {data_dir}")

    print(f"Found {len(md_files)} markdown files under {data_dir}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )

    embedder = LocalEmbedder(model_name=settings.embedding_model_name)
    vector_store = ChromaVectorStore(
        persist_directory=settings.chroma_db_dir,
        collection_name=settings.chroma_collection,
    )

    all_ids: List[str] = []
    all_texts: List[str] = []
    all_embeddings: List[List[float]] = []
    all_metadatas: List[dict] = []

    for md_path in md_files:
        source = os.path.relpath(md_path, data_dir)
        raw_text = md_path.read_text(encoding="utf-8")

        chunks = splitter.split_text(raw_text)
        if not chunks:
            continue

        print(f"Chunking {source}: {len(chunks)} chunks")

        embeddings = embedder.embed_texts(chunks)

        for text, embedding in zip(chunks, embeddings):
            doc_id = compute_id(text, source)
            all_ids.append(doc_id)
            all_texts.append(text)
            all_embeddings.append(embedding)
            all_metadatas.append(
                {
                    "source": source,
                }
            )

    if not all_ids:
        print("No chunks to ingest.")
        return

    print(f"Ingesting {len(all_ids)} chunks into ChromaDB...")
    vector_store.add_documents(
        ids=all_ids,
        texts=all_texts,
        embeddings=all_embeddings,
        metadatas=all_metadatas,
    )

    print("Ingestion complete.")


if __name__ == "__main__":
    ingest()

