from typing import Iterable, List

from sentence_transformers import SentenceTransformer


class LocalEmbedder:
    """HuggingFace sentence-transformers embedder running locally."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self._model = SentenceTransformer(model_name)

    def embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        return [embedding.tolist() for embedding in self._model.encode(list(texts), convert_to_numpy=True)]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_texts([text])[0]

