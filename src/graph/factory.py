import os
from src.graph.base import GraphStore


class GraphStoreFactory:
    """
    Creates and connects a GraphStore from config.
    New backends can be registered without modifying this class.
    """

    _registry: dict[str, type[GraphStore]] = {}

    @classmethod
    def _ensure_registered(cls) -> None:
        if not cls._registry:
            from src.graph.kuzu_store import KuzuGraphStore
            from src.graph.networkx_store import NetworkXGraphStore
            from src.graph.neo4j_store import Neo4jGraphStore

            cls._registry = {
                "kuzu": KuzuGraphStore,
                "networkx": NetworkXGraphStore,
                "neo4j": Neo4jGraphStore,
            }

    @classmethod
    def create(cls, config) -> GraphStore:
        """
        Instantiate and connect a GraphStore based on config.graph_backend.
        For "kuzu": db_path = {config.db_base_dir}/bank_{config.bank_id}.db
        For "networkx": no constructor args.
        For "neo4j": reads URI/credentials from config.
        """
        cls._ensure_registered()

        backend = config.graph_backend
        if backend not in cls._registry:
            raise ValueError(
                f"Unknown graph backend '{backend}'. "
                f"Available: {list(cls._registry.keys())}"
            )

        if backend == "kuzu":
            db_path = os.path.join(config.db_base_dir, f"bank_{config.bank_id}.db")
            store = cls._registry["kuzu"](db_path=db_path)
        elif backend == "networkx":
            store = cls._registry["networkx"]()
        elif backend == "neo4j":
            store = cls._registry["neo4j"](
                uri=config.neo4j_uri,
                user=config.neo4j_user,
                password=config.neo4j_password,
            )
        else:
            store = cls._registry[backend]()

        store.connect()
        return store

    @classmethod
    def register(cls, name: str, store_cls: type[GraphStore]) -> None:
        """Register a custom backend without modifying this class."""
        cls._ensure_registered()
        cls._registry[name] = store_cls
