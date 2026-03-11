from typing import Any

import networkx as nx

from src.graph.base import GraphStore


class NetworkXGraphStore(GraphStore):
    """
    In-memory graph backend using NetworkX.
    Intended for unit tests and quick local experimentation - no GPU or DB required.
    retrieve_context() produces identically structured strings to KuzuGraphStore
    so that prompt-level tests are backend-agnostic.
    """

    def __init__(self) -> None:
        self._graph: nx.DiGraph | None = None

    def connect(self) -> None:
        self._graph = nx.DiGraph()

    def create_schema(self) -> None:
        pass  # NetworkX is schemaless

    def ingest(self, nodes: list[dict], edges: list[dict]) -> None:
        for node in nodes:
            self._graph.add_node(node["id"], bank=node.get("bank"))
        for edge in edges:
            self._graph.add_edge(
                edge["from_id"],
                edge["to_id"],
                timestamp=edge.get("timestamp"),
                amount_paid=edge.get("amount_paid"),
                currency=edge.get("currency"),
                format=edge.get("format"),
                is_laundering=edge.get("is_laundering"),
            )

    def retrieve_context(self, account_id: str, limit: int = 20) -> str:
        if account_id not in self._graph:
            return f"No transactions found for account {account_id}."

        subgraph = nx.ego_graph(self._graph, account_id, radius=1)
        edges = list(subgraph.edges(data=True))[:limit]

        context = f"Transaction History for Account {account_id}:\n"
        for u, v, data in edges:
            context += (
                f"- {u} sent {data['amount_paid']} {data['currency']}"
                f" ({data['format']}) to {v} at {data['timestamp']}\n"
            )
        return context

    def query(self, query_str: str, params: dict[str, Any]) -> list[list[Any]]:
        """
        query_str is ignored - NetworkX has no query language.
        params must contain: {"account_id": str, "depth": int (optional)}
        Returns edge tuples as list[list].
        """
        account_id = params.get("account_id")
        depth = params.get("depth", 1)
        if account_id is None:
            raise ValueError("NetworkXGraphStore.query() requires params['account_id']")
        if account_id not in self._graph:
            return []
        subgraph = nx.ego_graph(self._graph, account_id, radius=depth)
        return [[u, v, d] for u, v, d in subgraph.edges(data=True)]

    def close(self) -> None:
        pass  # Nothing to release for in-memory graph
