import os
import tempfile
from typing import Any

import pandas as pd
import kuzu

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


class KuzuGraphStore(GraphStore):
    """
    Embedded graph database backend using Kuzu.
    One .db file per simulated bank node - maps directly to the federation architecture.

    Retrieval in "rag" mode is deliberately bank-scoped: only edges within the
    node's own partition are observable. Cross-bank chains are outside the
    privacy boundary by design - the architecture trades cross-bank chain
    visibility for raw-data confidentiality, and the pattern detectors in
    src/graph/patterns.py recover locally-observable structural signals
    (pass-through, structuring, cycle, rapid-burst, fan-out) without crossing
    that boundary.
    """

    def __init__(self, config: Config) -> None:
        self._db_path = os.path.join(config.db_base_dir, f"bank_{config.bank_id}.db")
        self._db: kuzu.Database | None = None
        self._conn: kuzu.Connection | None = None

    def connect(self) -> None:
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        self._db = kuzu.Database(self._db_path)
        self._conn = kuzu.Connection(self._db)

    def create_schema(self) -> None:
        # Idempotent - catch "already exists" errors silently
        for stmt in [
            """CREATE NODE TABLE Account(
                id STRING,
                bank INT64,
                PRIMARY KEY(id)
            )""",
            """CREATE REL TABLE Transaction(
                FROM Account TO Account,
                timestamp STRING,
                amount_paid DOUBLE,
                currency STRING,
                format STRING,
                is_laundering INT64
            )""",
        ]:
            try:
                self._conn.execute(stmt)
            except RuntimeError:
                pass  # Table already exists

    def ingest(self, nodes: list[dict], edges: list[dict]) -> None:
        """
        Uses COPY FROM temp CSVs for bulk ingestion - required for IBM AML scale.
        Row-by-row inserts would be unacceptably slow on the full dataset.

        Skips ingestion if data already exists - Kuzu's COPY FROM raises a PRIMARY KEY
        error on duplicate nodes, so re-running start_server() in the same Colab session
        would crash without this guard.
        """
        result = self._conn.execute("MATCH (a:Account) RETURN count(a) AS cnt")
        if result.get_next()[0] > 0:
            return  # already ingested

        with tempfile.TemporaryDirectory() as tmp:
            nodes_path = os.path.join(tmp, "accounts.csv")
            edges_path = os.path.join(tmp, "transactions.csv")

            pd.DataFrame(nodes).to_csv(nodes_path, index=False)
            pd.DataFrame(edges).to_csv(edges_path, index=False)

            self._conn.execute(f'COPY Account FROM "{nodes_path}" (HEADER=TRUE)')
            self._conn.execute(f'COPY Transaction FROM "{edges_path}" (HEADER=TRUE)')

    def retrieve_context(self, account_id: str, limit: int = 20, mode: str = "flat") -> str:
        flat = self._retrieve_flat_context(account_id, limit)
        if mode != "rag":
            return flat
        topology = self._retrieve_rag_context(account_id)
        if "No transactions found" in topology:
            return flat
        return flat + "\n\n" + topology

    def _retrieve_flat_context(self, account_id: str, limit: int) -> str:
        """
        Raw transaction list - up to limit rows covering outgoing and incoming.
        Uses UNION of two directed MATCHes: undirected MATCH returns each edge
        twice in Kuzu, so the UNION approach is required.
        """
        outgoing = self._conn.execute(
            """MATCH (a:Account {id: $id})-[t:Transaction]->(b:Account)
               RETURN a.id, b.id, t.amount_paid, t.currency, t.format, t.timestamp""",
            {"id": account_id},
        )
        incoming = self._conn.execute(
            """MATCH (b:Account)-[t:Transaction]->(a:Account {id: $id})
               RETURN b.id, a.id, t.amount_paid, t.currency, t.format, t.timestamp""",
            {"id": account_id},
        )
        rows = []
        for result in [outgoing, incoming]:
            while result.has_next() and len(rows) < limit:
                rows.append(result.get_next())
        if not rows:
            return f"No transactions found for account {account_id}."
        context = f"Transaction History for Account {account_id}:\n"
        for from_id, to_id, amount, currency, fmt, timestamp in rows[:limit]:
            context += f"- {from_id} sent {amount} {currency} ({fmt}) to {to_id} at {timestamp}\n"
        return context

    def _collect_edges(self, account_id: str):
        """
        Pull the bank-scoped edge set this node is allowed to observe:
        outgoing (full), intra-bank incoming, intra-bank 2-hop chains.
        Returns (bank_id, outgoing, incoming, chains, cross_bank_note) or
        (None, ...) if the account is not in this partition.
        """
        bank_res = self._conn.execute(
            "MATCH (a:Account {id: $id}) RETURN a.bank", {"id": account_id}
        )
        if not bank_res.has_next():
            return None, [], [], [], None
        bank_id = bank_res.get_next()[0]

        out_res = self._conn.execute(
            """MATCH (a:Account {id: $id})-[t:Transaction]->(b:Account)
               RETURN b.id, b.bank, t.amount_paid, t.timestamp""",
            {"id": account_id},
        )
        outgoing = []
        while out_res.has_next():
            to_id, to_bank, amount, timestamp = out_res.get_next()
            outgoing.append(OutEdge(
                to_id=to_id,
                to_bank=to_bank,
                amount=float(amount),
                timestamp=parse_timestamp(timestamp),
            ))

        in_res = self._conn.execute(
            """MATCH (b:Account)-[t:Transaction]->(a:Account {id: $id})
               WHERE b.bank = a.bank
               RETURN b.id, t.amount_paid, t.timestamp""",
            {"id": account_id},
        )
        incoming = []
        while in_res.has_next():
            from_id, amount, timestamp = in_res.get_next()
            incoming.append(InEdge(
                from_id=from_id,
                amount=float(amount),
                timestamp=parse_timestamp(timestamp),
            ))

        chain_res = self._conn.execute(
            """MATCH (a:Account {id: $id})-[t1:Transaction]->(b:Account)-[t2:Transaction]->(c:Account)
               WHERE a.bank = b.bank AND a.bank = c.bank AND c.id <> $id
               RETURN b.id, c.id, t1.amount_paid, t2.amount_paid, t1.timestamp
               LIMIT 10""",
            {"id": account_id},
        )
        chains = []
        while chain_res.has_next():
            mid, dest, amt1, amt2, ts = chain_res.get_next()
            chains.append(Chain(
                mid=mid,
                dest=dest,
                amt1=float(amt1),
                amt2=float(amt2),
                timestamp=parse_timestamp(ts),
            ))

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
        Uses edges this node observes to detect named patterns (pass-through,
        structuring, cycle, rapid-burst, fan-out) rather than emitting generic
        summary statistics. The flat transaction list already carries raw
        evidence - this section adds concrete, labelled signals the model
        can attend to during both training and inference.
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
        result = self._conn.execute(query_str, params)
        rows = []
        while result.has_next():
            rows.append(result.get_next())
        return rows

    def close(self) -> None:
        if self._conn:
            del self._conn
            self._conn = None
        if self._db:
            del self._db
            self._db = None
