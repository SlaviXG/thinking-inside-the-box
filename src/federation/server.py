import time

import numpy as np

from src.config import Config
from src.federation.client import AMLFederatedClient
from src.model.model_loader import load_model, load_tokenizer, attach_lora


class FLoRAStrategy:
    """
    FLoRA aggregation strategy.

    Instead of averaging LoRA adapter matrices (which introduces mathematical noise),
    this strategy stacks the A and B matrices from all clients and decomposes
    the result back to rank r via SVD. This gives the exact weighted sum of
    all adapter contributions with no approximation error.

    Parameter ordering convention (must match AMLFederatedClient.get_parameters()):
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


def start_server(config: Config, model=None, tokenizer=None) -> dict:
    """
    In-process federated simulation - all clients share one model instance.

    Flower's Virtual Client Engine uses Ray (separate processes), which copies
    the model into each worker via pickle, causing OOM even on A100. This manual
    loop runs all clients sequentially in the main process so the model is
    loaded exactly once and shared by reference.

    model and tokenizer can be passed in if already loaded in the session to
    avoid reloading weights into VRAM.

    Returns a history dict with per-round metrics for all three Trilemma axes:
      - train_loss:              list[list[float]]  - per round, per client
      - f1 / precision / recall: list[list[float]]  - per round, per client
      - round_latency_s:         list[float]        - wall-clock seconds per round
      - comm_bytes_flora:        list[list[int]]    - adapter delta bytes per round per client
      - comm_bytes_fedavg_per_round: int            - theoretical FedAvg cost (constant)
      - mia_auc:                 list[list[float]]  - MIA AUC per round per client
    """
    if model is None or tokenizer is None:
        print("Loading base model and tokenizer...")
        model = load_model(config)
        tokenizer = load_tokenizer(config)
        model = attach_lora(model, config)
        print("Model ready.\n")
    else:
        print("Reusing provided model and tokenizer.\n")

    # Theoretical FedAvg communication cost: full model weights per client per round.
    # Each bank would need to upload and receive a full copy of the 8B model.
    # Use float16 (2 bytes/param) - standard FedAvg transmission format regardless
    # of how the model is quantized in memory for compute.
    full_model_params = sum(p.numel() for p in model.parameters())
    fedavg_bytes_per_round = full_model_params * 2 * config.num_clients  # float16 per client

    # Instantiate all clients in the main process - all share the same model
    clients = []
    for cid in range(config.num_clients):
        client_config = Config.from_dict({
            **config.__dict__,
            "bank_id": cid + 1,  # bank_id=0 means "all banks"; start from 1
        })
        clients.append(AMLFederatedClient(client_config, model, tokenizer))

    strategy = FLoRAStrategy(lora_rank=config.lora_rank)
    global_params = clients[0].get_parameters()

    history = {
        "train_loss": [],               # list[list[float]]
        "f1": [],                       # list[list[float]]
        "precision": [],                # list[list[float]]
        "recall": [],                   # list[list[float]]
        "round_latency_s": [],          # list[float]
        "comm_bytes_flora": [],         # list[list[int]]
        "comm_bytes_fedavg_per_round": fedavg_bytes_per_round,  # int (constant)
        "mia_auc": [],                  # list[list[float]]
    }

    for round_num in range(1, config.num_rounds + 1):
        print(f"\n{'='*50}")
        print(f"Round {round_num}/{config.num_rounds}")
        print(f"{'='*50}")

        round_start = time.time()

        # Fit phase - sequential to keep one model in VRAM
        all_params, round_losses, round_comm_bytes = [], [], []
        for client in clients:
            params, n_samples, metrics = client.fit(
                global_params, {"local_epochs": config.local_epochs}
            )
            all_params.append(params)
            round_losses.append(metrics["train_loss"])
            round_comm_bytes.append(sum(arr.nbytes for arr in params))

        # Aggregate with FLoRA
        global_params = strategy.aggregate(round_num, all_params)

        round_elapsed = time.time() - round_start
        history["train_loss"].append(round_losses)
        history["comm_bytes_flora"].append(round_comm_bytes)
        history["round_latency_s"].append(round_elapsed)

        # MIA per client - must run before evaluate() so the model still holds each
        # client's local adapter weights (what was actually transmitted on the wire).
        # evaluate() calls set_parameters(global_params) which would overwrite them.
        print(f"\n[MIA] Round {round_num}")
        round_mia = []
        for i, client in enumerate(clients):
            client.set_parameters(all_params[i])  # restore local adapter for this client
            auc = client.mia_score(config.mia_n_members, config.mia_n_nonmembers)
            print(f"  [mia] bank_id={client._config.bank_id} AUC={auc:.3f}")
            round_mia.append(auc)
        history["mia_auc"].append(round_mia)

        # Evaluate phase - loads global params into the model
        round_f1, round_precision, round_recall = [], [], []
        for client in clients:
            _, _, eval_metrics = client.evaluate(global_params, {})
            round_f1.append(eval_metrics["f1"])
            round_precision.append(eval_metrics["precision"])
            round_recall.append(eval_metrics["recall"])
        history["f1"].append(round_f1)
        history["precision"].append(round_precision)
        history["recall"].append(round_recall)

    print("\nFederated simulation complete.")
    return history
