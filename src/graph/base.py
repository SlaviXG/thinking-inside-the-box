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

        mode="flat"  - raw transaction list, up to limit rows
        mode="graph" - bank-scoped topology stats: velocity, fan-out/fan-in,
                       intra-bank chains, cross-bank exposure summary.
                       Strictly limited to what one bank node can observe.
        """
        ...

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
