from typing import Any

import networkx as nx

from src.config import Config
from src.graph.base import GraphStore
from src.graph.patterns import (
    Chain,
    InEdge,
    OutEdge,
    detect_patterns,
    format_patterns,
    parse_timestamp,
)


class NetworkXGraphStore(GraphStore):
    """
    In-memory graph backend using NetworkX.
    Intended for unit tests and quick local experimentation - no GPU or DB required.
    retrieve_context() produces identically structured strings to KuzuGraphStore
    so that prompt-level tests are backend-agnostic.
    """

    def __init__(self, config: Config) -> None:
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

    def retrieve_context(self, account_id: str, limit: int = 20, mode: str = "flat") -> str:
        flat = self._retrieve_flat_context(account_id, limit)
        if mode != "rag":
            return flat
        topology = self._retrieve_rag_context(account_id)
        if "No transactions found" in topology:
            return flat
        return flat + "\n\n" + topology

    def _retrieve_flat_context(self, account_id: str, limit: int) -> str:
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

    def _collect_edges(self, account_id: str):
        """
        Pull the bank-scoped edge set this node can observe. Mirrors the Kuzu
        backend so pattern detection is backend-agnostic.
        """
        if account_id not in self._graph:
            return None, [], [], [], None

        bank_id = self._graph.nodes[account_id].get("bank")

        outgoing = []
        for _, v, d in self._graph.out_edges(account_id, data=True):
            outgoing.append(OutEdge(
                to_id=v,
                to_bank=self._graph.nodes[v].get("bank"),
                amount=float(d.get("amount_paid", 0.0)),
                timestamp=parse_timestamp(d.get("timestamp")),
            ))

        incoming = []
        for u, _, d in self._graph.in_edges(account_id, data=True):
            if self._graph.nodes[u].get("bank") != bank_id:
                continue
            incoming.append(InEdge(
                from_id=u,
                amount=float(d.get("amount_paid", 0.0)),
                timestamp=parse_timestamp(d.get("timestamp")),
            ))

        chains = []
        for _, mid, d1 in self._graph.out_edges(account_id, data=True):
            if self._graph.nodes[mid].get("bank") != bank_id:
                continue
            for _, dest, d2 in self._graph.out_edges(mid, data=True):
                if self._graph.nodes[dest].get("bank") != bank_id or dest == account_id:
                    continue
                chains.append(Chain(
                    mid=mid,
                    dest=dest,
                    amt1=float(d1.get("amount_paid", 0.0)),
                    amt2=float(d2.get("amount_paid", 0.0)),
                    timestamp=parse_timestamp(d1.get("timestamp")),
                ))
                if len(chains) >= 10:
                    break
            if len(chains) >= 10:
                break

        cross_out = [o for o in outgoing if o.to_bank != bank_id]
        cross_bank_note = None
        if cross_out:
            ext_banks = {o.to_bank for o in cross_out}
            cross_bank_note = (
                f"- Cross-bank exposure: {len(cross_out)} tx to "
                f"{len(ext_banks)} other bank(s); chains past the bank boundary "
                f"are not observable (privacy scope)."
            )

        return bank_id, outgoing, incoming, chains, cross_bank_note

    def _retrieve_rag_context(self, account_id: str) -> str:
        """
        Locally-detected AML pattern labels (bank-scoped).
        Identical schema to KuzuGraphStore so prompt-level tests stay
        backend-agnostic.
        """
        bank_id, outgoing, incoming, chains, cross_bank_note = self._collect_edges(account_id)
        if bank_id is None or (not outgoing and not incoming):
            return f"No transactions found for account {account_id}."

        patterns = detect_patterns(account_id, outgoing, incoming, chains)
        return format_patterns(account_id, bank_id, patterns, cross_bank_note)

    def structural_signals(self, account_id: str) -> list[str]:
        """Return AML pattern names for this account - used by rationale-augmented training."""
        bank_id, outgoing, incoming, chains, _ = self._collect_edges(account_id)
        if bank_id is None:
            return []
        return [p.name for p in detect_patterns(account_id, outgoing, incoming, chains)]

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
