import os
import tempfile
from typing import Any

import pandas as pd
import kuzu

from src.config import Config
from src.graph.base import GraphStore


class KuzuGraphStore(GraphStore):
    """
    Embedded graph database backend using Kuzu.
    One .db file per simulated bank node - maps directly to the federation architecture.
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
        if mode == "graph":
            return self._retrieve_graph_context(account_id)
        return self._retrieve_flat_context(account_id, limit)

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

    def _retrieve_graph_context(self, account_id: str) -> str:
        """
        Bank-scoped topology stats. Strictly limited to what one federation node
        can observe: outgoing transactions (full), intra-bank incoming (sender at
        same bank), and intra-bank 2-hop chains. Cross-bank flows are noted but
        not followed - the chain is invisible beyond the bank boundary.
        """
        bank_res = self._conn.execute(
            "MATCH (a:Account {id: $id}) RETURN a.bank", {"id": account_id}
        )
        if not bank_res.has_next():
            return f"No transactions found for account {account_id}."
        bank_id = bank_res.get_next()[0]

        # Outgoing: full visibility
        out_res = self._conn.execute(
            """MATCH (a:Account {id: $id})-[t:Transaction]->(b:Account)
               RETURN b.id, b.bank, t.amount_paid, t.timestamp""",
            {"id": account_id},
        )
        outgoing = []
        while out_res.has_next():
            outgoing.append(out_res.get_next())  # (to_id, to_bank, amount, timestamp)

        # Intra-bank incoming: sender must be at same bank
        in_res = self._conn.execute(
            """MATCH (b:Account)-[t:Transaction]->(a:Account {id: $id})
               WHERE b.bank = a.bank
               RETURN b.id, t.amount_paid, t.timestamp""",
            {"id": account_id},
        )
        incoming = []
        while in_res.has_next():
            incoming.append(in_res.get_next())  # (from_id, amount, timestamp)

        # Intra-bank 2-hop chains: all three accounts at same bank
        chain_res = self._conn.execute(
            """MATCH (a:Account {id: $id})-[t1:Transaction]->(b:Account)-[t2:Transaction]->(c:Account)
               WHERE a.bank = b.bank AND a.bank = c.bank AND c.id <> $id
               RETURN b.id, c.id, t1.amount_paid, t2.amount_paid, t1.timestamp
               LIMIT 3""",
            {"id": account_id},
        )
        chains = []
        while chain_res.has_next():
            chains.append(chain_res.get_next())  # (mid, dest, amt1, amt2, ts)

        if not outgoing and not incoming:
            return f"No transactions found for account {account_id}."

        lines = [f"Account {account_id} (Bank {bank_id}):"]

        if outgoing:
            amounts = [r[2] for r in outgoing]
            intra_out = [r for r in outgoing if r[1] == bank_id]
            cross_out = [r for r in outgoing if r[1] != bank_id]
            unique_dest = len(set(r[0] for r in outgoing))
            lines.append(
                f"\nOutgoing ({len(outgoing)} tx, {unique_dest} unique destinations):"
            )
            lines.append(
                f"  Volume: {sum(amounts):,.2f} total | "
                f"{sum(amounts)/len(amounts):,.2f} mean | {max(amounts):,.2f} max"
            )
            lines.append(
                f"  Intra-bank: {len(intra_out)} tx to "
                f"{len(set(r[0] for r in intra_out))} accounts at Bank {bank_id}"
            )
            if cross_out:
                ext_banks = set(r[1] for r in cross_out)
                lines.append(
                    f"  Cross-bank: {len(cross_out)} tx to {len(ext_banks)} other bank(s)"
                    f" - chain not visible beyond bank boundary"
                )
            # High-amount clustering: flag if 3+ transactions are in top quartile
            if len(amounts) >= 4:
                q3 = sorted(amounts)[int(len(amounts) * 0.75)]
                high = [a for a in amounts if a >= q3]
                if len(high) >= 3:
                    lines.append(
                        f"  High-amount clustering: {len(high)} tx >= {q3:,.0f}"
                        f" (top-quartile concentration)"
                    )

        if incoming:
            in_amounts = [r[1] for r in incoming]
            lines.append(
                f"\nIntra-bank incoming ({len(incoming)} tx, "
                f"{len(set(r[0] for r in incoming))} unique senders):"
            )
            lines.append(
                f"  Volume: {sum(in_amounts):,.2f} total | "
                f"{sum(in_amounts)/len(in_amounts):,.2f} mean"
            )

        if chains:
            lines.append(f"\nIntra-bank chains ({len(chains)}):")
            for mid, dest, amt1, amt2, ts in chains:
                lines.append(
                    f"  {account_id} -> {mid} -> {dest}"
                    f" ({amt1:,.0f} then {amt2:,.0f}, from {ts})"
                )

        return "\n".join(lines)

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
