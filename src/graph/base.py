from abc import ABC, abstractmethod
from typing import Any


class GraphStore(ABC):
    """
    Strategy interface for graph database backends.
    All federation and pipeline code depends only on this interface.
    """

    @abstractmethod
    def connect(self) -> None:
        """Open or initialize the backend connection."""
        ...

    @abstractmethod
    def create_schema(self) -> None:
        """Idempotently create node/edge tables or graph structures."""
        ...

    @abstractmethod
    def ingest(self, nodes: list[dict], edges: list[dict]) -> None:
        """Bulk-load prepared node and edge records."""
        ...

    @abstractmethod
    def retrieve_context(self, account_id: str, limit: int, mode: str = "flat") -> str:
        """
        Return a formatted natural-language string describing account activity,
        ready to be embedded in an LLM prompt.

        mode="flat" - raw transaction list, up to limit rows (no RAG augmentation)
        mode="rag"  - flat transaction list plus a locally-detected pattern
                      section (pass-through, structuring, cycle, rapid-burst,
                      fan-out). The flat rows give the model raw evidence;
                      the pattern section adds named AML signals the model
                      can latch onto during training and inference.
                      Bank-scoped by design: cross-bank chains are outside
                      the node's privacy boundary.
        """
        ...

    def structural_signals(self, account_id: str) -> list[str]:
        """
        Return the names of AML patterns detected for this account, or [].
        Used by training to build rationale-augmented targets so the adapter
        learns to map topology signals to the verdict. Default is no-op so
        stub backends (e.g. Neo4j) can be instantiated without implementing it.
        """
        return []

    @abstractmethod
    def query(self, query_str: str, params: dict[str, Any]) -> list[list[Any]]:
        """
        Execute a raw backend query.
        For Kuzu/Neo4j: query_str is Cypher.
        For NetworkX: query_str is ignored; params must contain account_id and depth.
        Returns rows as list[list] for uniform downstream handling.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Release any held resources."""
        ...

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.close()
