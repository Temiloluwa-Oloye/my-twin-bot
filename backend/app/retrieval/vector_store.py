from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import chromadb
from chromadb import PersistentClient


@dataclass
class RetrievedDocument:
    id: str
    text: str
    metadata: Dict[str, Any]


class ChromaVectorStore:
    """Thin wrapper around ChromaDB for storing and querying document embeddings."""

    def __init__(
        self,
        persist_directory: str,
        collection_name: str = "twin_knowledge_base",
    ) -> None:
        self._client: PersistentClient = chromadb.PersistentClient(path=persist_directory)
        self._collection = self._client.get_or_create_collection(name=collection_name)

    @property
    def collection(self):
        return self._collection

    def add_documents(
        self,
        ids: Sequence[str],
        texts: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        self._collection.add(
            ids=list(ids),
            documents=list(texts),
            embeddings=list(embeddings),
            metadatas=list(metadatas) if metadatas is not None else None,
        )

    def query_by_embedding(
        self,
        query_embedding: Iterable[float],
        top_k: int = 3,
    ) -> List[RetrievedDocument]:
        result = self._collection.query(
            query_embeddings=[list(query_embedding)],
            n_results=top_k,
        )

        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0] or [{} for _ in ids]

        retrieved: List[RetrievedDocument] = []
        for doc_id, text, metadata in zip(ids, docs, metadatas):
            retrieved.append(
                RetrievedDocument(
                    id=str(doc_id),
                    text=text,
                    metadata=metadata or {},
                )
            )
        return retrieved

