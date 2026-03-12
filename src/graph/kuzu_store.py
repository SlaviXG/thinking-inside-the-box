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
        """
        with tempfile.TemporaryDirectory() as tmp:
            nodes_path = os.path.join(tmp, "accounts.csv")
            edges_path = os.path.join(tmp, "transactions.csv")

            pd.DataFrame(nodes).to_csv(nodes_path, index=False)
            pd.DataFrame(edges).to_csv(edges_path, index=False)

            self._conn.execute(f'COPY Account FROM "{nodes_path}" (HEADER=TRUE)')
            self._conn.execute(f'COPY Transaction FROM "{edges_path}" (HEADER=TRUE)')

    def retrieve_context(self, account_id: str, limit: int = 20) -> str:
        """
        Uses UNION of two directed MATCHes to capture both outgoing and incoming
        transactions without duplicates. Undirected MATCH (a)-[t]-(b) returns
        each edge twice in Kuzu - the UNION approach is the correct fix.
        """
        outgoing = self._conn.execute(
            """MATCH (a:Account {id: $id})-[t:Transaction]->(b:Account)
               RETURN a.id AS from_id, b.id AS to_id,
                      t.amount_paid AS amount, t.currency AS currency,
                      t.format AS format, t.timestamp AS timestamp""",
            {"id": account_id},
        )
        incoming = self._conn.execute(
            """MATCH (b:Account)-[t:Transaction]->(a:Account {id: $id})
               RETURN b.id AS from_id, a.id AS to_id,
                      t.amount_paid AS amount, t.currency AS currency,
                      t.format AS format, t.timestamp AS timestamp""",
            {"id": account_id},
        )

        rows = []
        for result in [outgoing, incoming]:
            while result.has_next():
                rows.append(result.get_next())
            if len(rows) >= limit:
                break

        if not rows:
            return f"No transactions found for account {account_id}."

        context = f"Transaction History for Account {account_id}:\n"
        for from_id, to_id, amount, currency, fmt, timestamp in rows[:limit]:
            context += f"- {from_id} sent {amount} {currency} ({fmt}) to {to_id} at {timestamp}\n"
        return context

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
