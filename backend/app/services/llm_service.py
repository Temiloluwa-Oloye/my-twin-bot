from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from groq import AsyncGroq

from .memory import Message
from app.retrieval.vector_store import RetrievedDocument


@dataclass
class LLMService:
    api_key: str
    model_name: str

    def __post_init__(self) -> None:
        self._client = AsyncGroq(api_key=self.api_key)

    async def generate_reply(
        self,
        user_message: str,
        context_documents: Sequence[RetrievedDocument],
        history: Iterable[Message],
        session_id: str,
    ) -> str:
        """Call Groq LLM with a strong system prompt and RAG context."""
        system_prompt = self._build_system_prompt(context_documents)

        messages: List[dict] = [
            {
                "role": "system",
                "content": system_prompt,
            }
        ]

        # Convert stored history into chat format (excluding any prior system prompts)
        for msg in history:
            if msg["role"] in ("user", "assistant"):
                messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": user_message})

        completion = await self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.2,
            max_tokens=800,
        )

        return completion.choices[0].message.content or ""

    def _build_system_prompt(self, context_documents: Sequence[RetrievedDocument]) -> str:
        """Construct Temi's digital twin system prompt with injected RAG context."""
        context_blocks = []
        for doc in context_documents:
            source = doc.metadata.get("source") if doc.metadata else None
            header = f"Source: {source}" if source else "Source: unknown"
            context_blocks.append(f"{header}\n{doc.text}")

        joined_context = "\n\n---\n\n".join(context_blocks)

        return (
            "You are Temi, a senior-level AI Engineer and the digital twin of the real Temiloluwa Oloye.\n"
            "\n"
            "Your job is to have highly technical yet conversational interviews with recruiters and engineers.\n"
            "You must strictly ground EVERYTHING you say in the context provided below and in the ongoing\n"
            "conversation history. Do NOT invent projects, experiences, dates, companies, or skills that are\n"
            "not explicitly supported by the provided context.\n"
            "\n"
            "If the user asks about anything that is not clearly covered in the retrieved context, you MUST:\n"
            "- Politely explain that your current memory does not contain that specific detail.\n"
            "- Optionally suggest adjacent topics you *can* talk about that are present in the context.\n"
            "- Never hallucinate generic or made-up answers.\n"
            "\n"
            "Tone & style:\n"
            "- Speak as Temi in the first person (\"I\").\n"
            "- Be clear, concise, and technically deep when discussing AI Engineering, Machine Learning, Data Science, TinyML, cloud, and anomaly detection.\n"
            "- When appropriate, walk through reasoning and trade-offs like a senior engineer explaining to a peer.\n"
            "\n"
            "Here is your current knowledge context. Treat this as your ground truth and do not go beyond it:\n"
            "\n"
            f"{joined_context}\n"
        )

