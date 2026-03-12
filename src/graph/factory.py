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
        All backends accept Config as their sole constructor argument.
        """
        cls._ensure_registered()

        backend = config.graph_backend
        if backend not in cls._registry:
            raise ValueError(
                f"Unknown graph backend '{backend}'. "
                f"Available: {list(cls._registry.keys())}"
            )

        store = cls._registry[backend](config)
        store.connect()
        return store

    @classmethod
    def register(cls, name: str, store_cls: type[GraphStore]) -> None:
        """Register a custom backend without modifying this class."""
        cls._ensure_registered()
        cls._registry[name] = store_cls
