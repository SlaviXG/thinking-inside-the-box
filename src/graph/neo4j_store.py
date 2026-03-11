from typing import Any

from src.graph.base import GraphStore

_NOT_IMPLEMENTED_MSG = (
    "Neo4jGraphStore is a Phase 4+ stub for real distributed deployment. "
    "Use KuzuGraphStore (embedded) or NetworkXGraphStore (in-memory) for simulation."
)


class Neo4jGraphStore(GraphStore):
    """
    Stub implementation for a Neo4j-backed graph store.
    Intended for real multi-node deployment beyond the dissertation simulation.
    All methods raise NotImplementedError until implemented.
    """

    def __init__(self, uri: str, user: str, password: str) -> None:
        self._uri = uri
        self._user = user
        self._password = password

    def connect(self) -> None:
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    def create_schema(self) -> None:
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    def ingest(self, nodes: list[dict], edges: list[dict]) -> None:
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    def retrieve_context(self, account_id: str, limit: int = 20) -> str:
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    def query(self, query_str: str, params: dict[str, Any]) -> list[list[Any]]:
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    def close(self) -> None:
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)
