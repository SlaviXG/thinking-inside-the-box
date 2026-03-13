import numpy as np

from src.config import Config
from src.federation.client import AMLFlowerClient
from src.model.model_loader import load_model, load_tokenizer, attach_lora


class FLoRAStrategy:
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

    def __init__(self, lora_rank: int) -> None:
        self._r = lora_rank

    def aggregate(
        self,
        server_round: int,
        all_params: list[list[np.ndarray]],
    ) -> list[np.ndarray]:
        """
        Aggregate LoRA parameters from all clients using FLoRA stacking + SVD.
        all_params: one list of np.ndarray per client, ordered A's then B's.
        """
        print(f"\n[FLoRA] Round {server_round} - aggregating {len(all_params)} clients")

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

        new_A, new_B = [], []
        for A_stack, B_stack in zip(stacked_A, stacked_B):
            A_new, B_new = self._flora_decompose(B_stack, A_stack)
            new_A.append(A_new)
            new_B.append(B_new)

        print(f"[FLoRA] Round {server_round} - aggregation complete")
        return new_A + new_B  # Preserve get_parameters() ordering

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


def start_server(config: Config, model=None, tokenizer=None) -> None:
    """
    In-process federated simulation - all clients share one model instance.

    Flower's Virtual Client Engine uses Ray (separate processes), which copies
    the model into each worker via pickle, causing OOM even on A100. This manual
    loop runs all clients sequentially in the main process so the model is
    loaded exactly once and shared by reference.

    model and tokenizer can be passed in if already loaded in the session to
    avoid reloading weights into VRAM.
    """
    if model is None or tokenizer is None:
        print("Loading base model and tokenizer...")
        model = load_model(config)
        tokenizer = load_tokenizer(config)
        model = attach_lora(model, config)
        print("Model ready.\n")
    else:
        print("Reusing provided model and tokenizer.\n")

    # Instantiate all clients in the main process - all share the same model
    clients = []
    for cid in range(config.num_clients):
        client_config = Config.from_dict({
            **config.__dict__,
            "bank_id": cid + 1,  # bank_id=0 means "all banks"; start from 1
        })
        clients.append(AMLFlowerClient(client_config, model, tokenizer))

    strategy = FLoRAStrategy(lora_rank=config.lora_rank)
    global_params = clients[0].get_parameters()

    for round_num in range(1, config.num_rounds + 1):
        print(f"\n{'='*50}")
        print(f"Round {round_num}/{config.num_rounds}")
        print(f"{'='*50}")

        # Fit phase - sequential to keep one model in VRAM
        all_params = []
        for client in clients:
            params, n_samples, metrics = client.fit(
                global_params, {"local_epochs": config.local_epochs}
            )
            all_params.append(params)

        # Aggregate with FLoRA
        global_params = strategy.aggregate(round_num, all_params)

        # Evaluate phase
        for client in clients:
            client.evaluate(global_params, {})

    print("\nFederated simulation complete.")
