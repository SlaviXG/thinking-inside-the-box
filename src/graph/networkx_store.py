from typing import Any

import networkx as nx

from src.config import Config
from src.graph.base import GraphStore


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
        if mode != "graph":
            return flat
        topology = self._retrieve_graph_context(account_id)
        if "No transactions found" in topology:
            return flat
        return flat + "\n\nTopology Analysis:\n" + topology

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

    def _retrieve_graph_context(self, account_id: str) -> str:
        if account_id not in self._graph:
            return f"No transactions found for account {account_id}."

        bank_id = self._graph.nodes[account_id].get("bank")

        outgoing = [
            (v, self._graph.nodes[v].get("bank"), d)
            for _, v, d in self._graph.out_edges(account_id, data=True)
        ]
        incoming = [
            (u, d)
            for u, _, d in self._graph.in_edges(account_id, data=True)
            if self._graph.nodes[u].get("bank") == bank_id
        ]

        # Intra-bank 2-hop chains
        chains = []
        for _, mid, _ in self._graph.out_edges(account_id, data=True):
            if self._graph.nodes[mid].get("bank") == bank_id:
                for _, dest, _ in self._graph.out_edges(mid, data=True):
                    if self._graph.nodes[dest].get("bank") == bank_id and dest != account_id:
                        chains.append((mid, dest))

        if not outgoing and not incoming:
            return f"No transactions found for account {account_id}."

        lines = [f"Account {account_id} (Bank {bank_id}):"]

        if outgoing:
            amounts = [d["amount_paid"] for _, _, d in outgoing]
            intra_out = [(v, d) for v, bk, d in outgoing if bk == bank_id]
            cross_out = [(v, bk, d) for v, bk, d in outgoing if bk != bank_id]
            unique_dest = len(set(v for v, _, _ in outgoing))
            lines.append(
                f"\nOutgoing ({len(outgoing)} tx, {unique_dest} unique destinations):"
            )
            lines.append(
                f"  Volume: {sum(amounts):,.2f} total | "
                f"{sum(amounts)/len(amounts):,.2f} mean | {max(amounts):,.2f} max"
            )
            lines.append(
                f"  Intra-bank: {len(intra_out)} tx to "
                f"{len(set(v for v, _ in intra_out))} accounts at Bank {bank_id}"
            )
            if cross_out:
                ext_banks = set(bk for _, bk, _ in cross_out)
                lines.append(
                    f"  Cross-bank: {len(cross_out)} tx to {len(ext_banks)} other bank(s)"
                    f" - chain not visible beyond bank boundary"
                )
            if len(amounts) >= 4:
                q3 = sorted(amounts)[int(len(amounts) * 0.75)]
                high = [a for a in amounts if a >= q3]
                if len(high) >= 3:
                    lines.append(
                        f"  High-amount clustering: {len(high)} tx >= {q3:,.0f}"
                        f" (top-quartile concentration)"
                    )

        if incoming:
            in_amounts = [d["amount_paid"] for _, d in incoming]
            lines.append(
                f"\nIntra-bank incoming ({len(incoming)} tx, "
                f"{len(set(u for u, _ in incoming))} unique senders):"
            )
            lines.append(
                f"  Volume: {sum(in_amounts):,.2f} total | "
                f"{sum(in_amounts)/len(in_amounts):,.2f} mean"
            )

        if chains:
            lines.append(f"\nIntra-bank chains ({len(chains[:3])}):")
            for mid, dest in chains[:3]:
                lines.append(f"  {account_id} -> {mid} -> {dest}")

        return "\n".join(lines)

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
