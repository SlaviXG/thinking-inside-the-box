from typing import Optional

import numpy as np
import flwr as fl
from flwr.common import (
    Parameters,
    FitRes,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from flwr.server.client_proxy import ClientProxy

from src.config import Config
from src.federation.client import AMLFlowerClient
from src.model.model_loader import load_model, load_tokenizer, attach_lora


class FLoRAStrategy(fl.server.strategy.FedAvg):
    """
    FLoRA aggregation strategy.

    Instead of averaging LoRA adapter matrices (which introduces mathematical noise),
    this strategy stacks the A and B matrices from all clients and decomposes
    the result back to rank r via SVD. This gives the exact weighted sum of
    all adapter contributions with no approximation error.

    Parameter ordering convention (must match AMLFlowerClient.get_parameters()):
      parameters[:n_lora] = all lora_A matrices (shape: r x in_features each)
      parameters[n_lora:] = all lora_B matrices (shape: out_features x r each)
    """

    def __init__(self, lora_rank: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self._r = lora_rank

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures,
    ) -> tuple[Optional[Parameters], dict]:
        if not results:
            return None, {}

        print(f"\n[FLoRA] Round {server_round} - aggregating {len(results)} clients")

        # Collect parameters from all clients
        all_params = [
            parameters_to_ndarrays(fit_res.parameters)
            for _, fit_res in results
        ]

        # Validate all clients returned the same number of parameter arrays
        n_params = len(all_params[0])
        if not all(len(p) == n_params for p in all_params):
            raise ValueError(
                "Clients returned inconsistent parameter counts - "
                "all clients must use identical LoRA configurations."
            )

        n_lora = n_params // 2  # First half: A matrices. Second half: B matrices.

        # Stack A matrices vertically: each (r, in_features) -> (n*r, in_features)
        stacked_A = [
            np.concatenate([client[i] for client in all_params], axis=0)
            for i in range(n_lora)
        ]

        # Stack B matrices horizontally: each (out_features, r) -> (out_features, n*r)
        stacked_B = [
            np.concatenate([client[i] for client in all_params], axis=1)
            for i in range(n_lora, n_params)
        ]

        # SVD decompose each stacked pair back to rank r
        new_A, new_B = [], []
        for A_stack, B_stack in zip(stacked_A, stacked_B):
            A_new, B_new = self._flora_decompose(B_stack, A_stack)
            new_A.append(A_new)
            new_B.append(B_new)

        aggregated = new_A + new_B  # Preserve get_parameters() ordering
        print(f"[FLoRA] Round {server_round} - aggregation complete")
        return ndarrays_to_parameters(aggregated), {}

    def _flora_decompose(
        self, B_stack: np.ndarray, A_stack: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Given stacked adapter matrices:
          B_stack: (out_features, n*r)
          A_stack: (n*r, in_features)

        Compute delta_W = B_stack @ A_stack  (exact sum of all client contributions)
        then factorize back to rank r via SVD:
          delta_W = U @ diag(S) @ Vt
          B_new = U[:, :r] @ diag(sqrt(S[:r]))     shape: (out_features, r)
          A_new = diag(sqrt(S[:r])) @ Vt[:r, :]    shape: (r, in_features)
        """
        delta_W = B_stack @ A_stack  # (out_features, in_features)
        U, S, Vt = np.linalg.svd(delta_W, full_matrices=False)

        r = self._r
        S_sqrt = np.sqrt(np.maximum(S[:r], 0.0))  # clip for numerical stability

        A_new = (np.diag(S_sqrt) @ Vt[:r, :]).astype(np.float32)
        B_new = (U[:, :r] @ np.diag(S_sqrt)).astype(np.float32)
        return A_new, B_new


def build_client_fn(config_template: Config, model, tokenizer):
    """
    Returns a client_fn(cid: str) -> AMLFlowerClient closure.
    Each client gets its own bank_id from cid, giving it a unique data partition
    and its own Kuzu .db file.
    """
    def client_fn(cid: str) -> AMLFlowerClient:
        config = Config.from_dict({
            **config_template.__dict__,
            "bank_id": int(cid) + 1,  # bank_id=0 means "all banks"; start from 1
        })
        return AMLFlowerClient(config, model, tokenizer)

    return client_fn


def start_server(config: Config, model=None, tokenizer=None) -> None:
    """
    Launch the federated simulation via Flower's Virtual Client Engine.

    model and tokenizer can be passed in if already loaded (e.g. from a prior
    Colab cell) to avoid reloading 6GB of weights and exhausting T4 VRAM.
    If not provided, they are loaded and LoRA adapters attached here.
    """
    if model is None or tokenizer is None:
        print("Loading base model and tokenizer...")
        model = load_model(config)
        tokenizer = load_tokenizer(config)
        model = attach_lora(model, config)
        print("Model ready.\n")
    else:
        print("Reusing provided model and tokenizer.\n")

    strategy = FLoRAStrategy(
        lora_rank=config.lora_rank,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=config.num_clients,
        min_evaluate_clients=config.num_clients,
        min_available_clients=config.num_clients,
    )

    client_fn = build_client_fn(config, model, tokenizer)

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=config.num_clients,
        config=fl.server.ServerConfig(num_rounds=config.num_rounds),
        strategy=strategy,
        client_resources={"num_gpus": 1.0 / config.num_clients},
    )
