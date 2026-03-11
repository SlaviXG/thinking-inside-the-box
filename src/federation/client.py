import numpy as np
import flwr as fl

from src.config import Config
from src.data.aml_ingestor import AMLIngestor
from src.graph.factory import GraphStoreFactory
from src.pipeline.investigation import InvestigationPipeline


class AMLFlowerClient(fl.client.NumPyClient):
    """
    Flower client representing one bank node in the federation.

    Each client owns:
    - Its own KuzuGraphStore (.db file partitioned by bank_id)
    - An InvestigationPipeline backed by the shared frozen base model
    - Local LoRA adapter parameters updated each round

    The base model is loaded once on the server and passed in via the
    client_fn closure to avoid reloading 6GB of weights per client.
    """

    def __init__(self, config: Config, model, tokenizer) -> None:
        self._config = config
        graph_store = GraphStoreFactory.create(config)
        AMLIngestor(config).run(graph_store)
        self._pipeline = InvestigationPipeline(graph_store, model, tokenizer, config)

    def get_parameters(self, ins) -> list[np.ndarray]:
        """Extract current LoRA adapter weights as numpy arrays."""
        # TODO Phase 3: extract adapter weights via model.get_adapter_state_dict()
        return []

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        """Apply aggregated LoRA weights received from the central server."""
        # TODO Phase 3: apply adapter weights via model.load_adapter_state_dict()
        pass

    def fit(self, parameters, config) -> tuple[list[np.ndarray], int, dict]:
        """
        Local training round:
        1. Apply global adapter weights from server
        2. Run investigation on training accounts to generate reasoning traces
        3. Fine-tune LoRA adapters on those traces
        4. Return updated adapter weights, sample count, and metrics
        """
        self.set_parameters(parameters)
        # TODO Phase 3: implement LoRA fine-tuning loop
        return self.get_parameters(ins=None), 0, {}

    def evaluate(self, parameters, config) -> tuple[float, int, dict]:
        """
        Local evaluation round:
        Run investigation on held-out accounts and compute F1 on Is Laundering labels.
        """
        self.set_parameters(parameters)
        # TODO Phase 3: implement F1 evaluation against graph labels
        return 0.0, 0, {"f1": 0.0}
