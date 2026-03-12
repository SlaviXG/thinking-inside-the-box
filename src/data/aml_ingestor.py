import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import Config
from src.graph.base import GraphStore

# Exact column names from the Kaggle IBM AML dataset schema
_IBM_COLUMNS = [
    "Timestamp", "From Bank", "Account", "To Bank", "Account.1",
    "Amount Received", "Receiving Currency", "Amount Paid",
    "Payment Currency", "Payment Format", "Is Laundering",
]


class AMLIngestor:
    """
    Reads the IBM AML CSV and drives GraphStore.ingest().
    Handles partitioning by bank ID and train/val/test splitting.
    """

    def __init__(self, config: Config) -> None:
        self._config = config

    def load_partition(self) -> pd.DataFrame:
        """
        Read CSV and filter to rows where From Bank == config.bank_id.
        bank_id=0 loads all banks (single-node Phase 2 testing).
        """
        df = pd.read_csv(self._config.csv_path)
        if self._config.bank_id != 0:
            df = df[df["From Bank"] == self._config.bank_id]
        return df.reset_index(drop=True)

    def split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Stratified 70/15/15 train/val/test split on the account level.
        Stratification preserves the Is Laundering ratio in each split.
        Returns (train_df, val_df, test_df) - each row is a unique source account
        with its max Is Laundering label (1 if any outgoing tx is suspicious).
        """
        # Collapse to one row per source account with its label
        account_labels = (
            df.groupby("Account")["Is Laundering"]
            .max()
            .reset_index()
            .rename(columns={"Account": "account_id", "Is Laundering": "label"})
        )
        account_labels["account_id"] = account_labels["account_id"].astype(str)

        train_ratio = self._config.train_ratio
        val_ratio = self._config.val_ratio

        # First split: train vs (val + test)
        train_df, temp_df = train_test_split(
            account_labels,
            train_size=train_ratio,
            stratify=account_labels["label"],
            random_state=42,
        )

        # Second split: val vs test (val_ratio out of remaining 1 - train_ratio)
        relative_val = val_ratio / (1.0 - train_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            train_size=relative_val,
            stratify=temp_df["label"],
            random_state=42,
        )

        return (
            train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True),
        )

    def prepare_nodes(self, df: pd.DataFrame) -> list[dict]:
        from_accounts = df[["Account", "From Bank"]].rename(
            columns={"Account": "id", "From Bank": "bank"}
        )
        to_accounts = df[["Account.1", "To Bank"]].rename(
            columns={"Account.1": "id", "To Bank": "bank"}
        )
        nodes = (
            pd.concat([from_accounts, to_accounts])
            .drop_duplicates(subset=["id"])
            .reset_index(drop=True)
        )
        nodes["id"] = nodes["id"].astype(str)
        nodes["bank"] = nodes["bank"].astype(int)
        return nodes.to_dict(orient="records")

    def prepare_edges(self, df: pd.DataFrame) -> list[dict]:
        edges = df[
            ["Account", "Account.1", "Timestamp", "Amount Paid",
             "Payment Currency", "Payment Format", "Is Laundering"]
        ].copy()
        edges.columns = [
            "from_id", "to_id", "timestamp", "amount_paid",
            "currency", "format", "is_laundering",
        ]
        edges["from_id"] = edges["from_id"].astype(str)
        edges["to_id"] = edges["to_id"].astype(str)
        edges["amount_paid"] = edges["amount_paid"].astype(float)
        edges["is_laundering"] = edges["is_laundering"].astype(int)
        return edges.to_dict(orient="records")

    def run(self, graph_store: GraphStore) -> None:
        """Load partition from CSV and ingest into graph_store."""
        df = self.load_partition()
        self.run_from_df(graph_store, df)

    def run_from_df(self, graph_store: GraphStore, df: pd.DataFrame) -> None:
        """Ingest an already-loaded DataFrame - avoids re-reading CSV."""
        print(f"Ingesting {len(df)} transactions...")
        nodes = self.prepare_nodes(df)
        edges = self.prepare_edges(df)
        print(f"  {len(nodes)} unique accounts, {len(edges)} transactions.")
        graph_store.create_schema()
        graph_store.ingest(nodes, edges)
        print("Ingest complete.")
