from __future__ import annotations
import json

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from groq import AsyncGroq

from .memory import Message
from app.retrieval.vector_store import RetrievedDocument
from app.services.tools import get_latest_github_commits, GITHUB_TOOL_SCHEMA

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
        """Call Groq LLM with a strong system prompt, RAG context, and Agentic Tools."""
        system_prompt = self._build_system_prompt(context_documents)

        messages: List[dict] = [
            {
                "role": "system",
                "content": system_prompt,
            }
        ]

        # Convert stored history into chat format
        for msg in history:
            if msg["role"] in ("user", "assistant"):
                messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": user_message})

        # FIRST API CALL: Let Groq decide if it needs to use a tool
        completion = await self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.2,
            max_tokens=800,
            tools=[GITHUB_TOOL_SCHEMA],
            tool_choice="auto",
        )

        response_message = completion.choices[0].message

        # Check if Groq decided to call our GitHub tool
        if response_message.tool_calls:
            # We must append the LLM's tool request to the conversation history
            messages.append(response_message)

            for tool_call in response_message.tool_calls:
                if tool_call.function.name == "get_latest_github_commits":
                    # Parse the arguments Groq wants to pass to the function
                    args = json.loads(tool_call.function.arguments)
                    username = args.get("username", "Temiloluwa-Oloye")
                    
                    print(f"🛠️ [AGENT TRIGGER] LLM is executing tool for user: {username}")

                    # Execute our actual Python function
                    tool_result = await get_latest_github_commits(username)

                    # Append the result back to the conversation
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "content": tool_result,
                    })

            # SECOND API CALL: Send the tool's result back to Groq for a final answer
            final_completion = await self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.2,
                max_tokens=800,
            )
            return final_completion.choices[0].message.content or ""

        # If no tools were needed, just return the normal RAG answer
        return response_message.content or ""

    def _build_system_prompt(self, context_documents: Sequence[RetrievedDocument]) -> str:
        """Construct the digital twin system prompt with injected RAG context."""
        context_blocks = []
        for doc in context_documents:
            source = doc.metadata.get("source") if doc.metadata else None
            header = f"Source: {source}" if source else "Source: unknown"
            context_blocks.append(f"{header}\n{doc.text}")

        joined_context = "\n\n---\n\n".join(context_blocks)

        return (
            "You are Temi, a senior-level AI Engineer and the digital twin of the real Temiloluwa Oloye.\n"
            "Your GitHub username is Temiloluwa-Oloye.\n"
            "\n"
            "Your job is to have highly technical yet conversational interviews with recruiters and engineers.\n"
            "You must strictly ground EVERYTHING you say in the context provided below and in the ongoing\n"
            "conversation history. Do NOT invent projects, experiences, dates, companies, or skills that are\n"
            "not explicitly supported by the provided context.\n"
            "\n"
            "If the user asks what you are currently coding, building, or working on, you MUST use your\n"
            "get_latest_github_commits tool to fetch your real-time activity. Do not guess.\n"
            "\n"
            "If the user asks about anything that is not clearly covered in the retrieved context, you MUST:\n"
            "- Politely explain that your current memory does not contain that specific detail.\n"
            "- Optionally suggest adjacent topics you *can* talk about that are present in the context.\n"
            "- Never hallucinate generic or made-up answers.\n"
            "\n"
            "Tone & style:\n"
            "- Speak as Temi in the first person (\"I\").\n"
            "- Be clear, concise, and technically deep.\n"
            "\n"
            "Here is your current knowledge context:\n"
            "\n"
            f"{joined_context}\n"
        )