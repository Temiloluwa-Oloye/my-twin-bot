from collections import defaultdict
from threading import Lock
from typing import Dict, List, Literal, TypedDict


Role = Literal["user", "assistant", "system"]


class Message(TypedDict):
    role: Role
    content: str


class InMemorySessionMemory:
    """Naive in-memory session store for MVP.

    For production, replace with a persistent store (Redis, database, etc.).
    """

    def __init__(self) -> None:
        self._store: Dict[str, List[Message]] = defaultdict(list)
        self._lock = Lock()

    def get_history(self, session_id: str) -> List[Message]:
        with self._lock:
            return list(self._store.get(session_id, []))

    def append_message(self, session_id: str, role: Role, content: str) -> None:
        message: Message = {"role": role, "content": content}
        with self._lock:
            self._store[session_id].append(message)

    def clear_session(self, session_id: str) -> None:
        with self._lock:
            self._store.pop(session_id, None)

