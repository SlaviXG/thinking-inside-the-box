import json
import time

import numpy as np

from src.config import Config
from src.federation.client import AMLFederatedClient
from src.model.model_loader import load_model, load_tokenizer, attach_lora
from src.security.encryption import AdapterEncryption


class FLoRAStrategy:
    """
    FLoRA aggregation strategy.

    Instead of averaging LoRA adapter matrices (which introduces mathematical noise),
    this strategy stacks the A and B matrices from all clients, computes their
    mean weight delta, then decomposes back to rank r via SVD (Ye et al. 2023).
    This preserves the low-rank structure while faithfully representing all
    client contributions.

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
        all_n_samples: list[int] = None,  # unused by FLoRA but kept for interface parity
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

        Compute the mean weight delta across all clients, then factorize back
        to rank r via SVD (Ye et al. FLoRA, 2023):
          delta_W = U @ diag(S) @ Vt
          B_new = U[:, :r] @ diag(sqrt(S[:r]))     shape: (out_features, r)
          A_new = diag(sqrt(S[:r])) @ Vt[:r, :]    shape: (r, in_features)
        """
        n_clients = B_stack.shape[1] // self._r
        delta_W = (B_stack @ A_stack) / n_clients  # mean delta, not sum
        U, S, Vt = np.linalg.svd(delta_W, full_matrices=False)

        r = self._r
        S_sqrt = np.sqrt(np.maximum(S[:r], 0.0))  # clip for numerical stability

        A_new = (np.diag(S_sqrt) @ Vt[:r, :]).astype(np.float32)
        B_new = (U[:, :r] @ np.diag(S_sqrt)).astype(np.float32)
        return A_new, B_new


class FedAvgStrategy:
    """
    Standard FedAvg aggregation applied to LoRA adapter matrices.

    Computes a weighted average of each adapter parameter across all clients,
    weighted by the number of training samples each client contributed.
    This is the standard McMahan et al. (2017) aggregation rule applied to
    LoRA adapters instead of full model weights.

    Communication cost is still only the adapter deltas (not full weights),
    but aggregation quality is lower than FLoRA because matrix averaging
    introduces rank drift - the averaged adapters do not preserve the
    geometric structure of the original low-rank factorization.
    """

    def aggregate(
        self,
        server_round: int,
        all_params: list[list[np.ndarray]],
        all_n_samples: list[int] = None,
    ) -> list[np.ndarray]:
        n_clients = len(all_params)
        if not all_n_samples or sum(all_n_samples) == 0:
            weights = [1.0 / n_clients] * n_clients
        else:
            total = sum(all_n_samples)
            weights = [n / total for n in all_n_samples]

        print(f"\n[FedAvg] Round {server_round} - averaging {n_clients} clients")
        aggregated = [
            sum(w * p[i] for w, p in zip(weights, all_params)).astype(np.float32)
            for i in range(len(all_params[0]))
        ]
        print(f"[FedAvg] Round {server_round} - aggregation complete")
        return aggregated


def start_server(config: Config, model=None, tokenizer=None, aggregation: str = "flora") -> dict:
    """
    In-process federated simulation - all clients share one model instance.

    Flower's Virtual Client Engine uses Ray (separate processes), which copies
    the model into each worker via pickle, causing OOM even on A100. This manual
    loop runs all clients sequentially in the main process so the model is
    loaded exactly once and shared by reference.

    model and tokenizer can be passed in if already loaded in the session to
    avoid reloading weights into VRAM.

    aggregation selects the server-side aggregation strategy:
      "flora"  - FLoRA stacking + SVD (default, ours)
      "fedavg" - weighted average of adapter matrices (McMahan et al. 2017 baseline)

    Returns a history dict with per-round metrics for all three Trilemma axes:
      - train_loss:                  list[list[float]]  - per round, per client
      - f1 / precision / recall:     list[list[float]]  - per round, per client
      - fit_latency_s:               list[float]        - fit + aggregate wall-clock per round
      - mia_latency_s:               list[float]        - MIA scoring wall-clock per round
      - eval_latency_s:              list[float]        - evaluate() wall-clock per round
      - round_latency_s:             list[float]        - total wall-clock per round
      - comm_bytes_flora:            list[list[int]]    - adapter delta bytes per round per client
      - comm_bytes_fedavg_per_round: int                - theoretical full-weight FedAvg cost (constant)
      - mia_auc:                     list[list[float]]  - MIA AUC per round per client
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
    # bank_ids overrides the default sequential 1..num_clients assignment,
    # allowing selection of partitions with better class balance.
    bank_id_list = (
        list(config.bank_ids)
        if config.bank_ids is not None
        else list(range(1, config.num_clients + 1))
    )
    clients = []
    for bank_id in bank_id_list:
        client_config = Config.from_dict({**config.__dict__, "bank_id": bank_id})
        clients.append(AMLFederatedClient(client_config, model, tokenizer))

    if aggregation == "fedavg":
        strategy = FedAvgStrategy()
    else:
        strategy = FLoRAStrategy(lora_rank=config.lora_rank)
    global_params = clients[0].get_parameters()

    # Server-side decryption instances - one per client, built once from each
    # client's public key. Keys are stable for the lifetime of the simulation.
    server_cipher = [AdapterEncryption.from_key(c.encryption_key) for c in clients]

    history = {
        "train_loss": [],               # list[list[float]]
        "f1": [],                       # list[list[float]]
        "precision": [],                # list[list[float]]
        "recall": [],                   # list[list[float]]
        "fit_latency_s": [],            # list[float] - fit + aggregate only
        "mia_latency_s": [],            # list[float] - MIA scoring only
        "eval_latency_s": [],           # list[float] - evaluate() only
        "round_latency_s": [],          # list[float] - total wall-clock per round
        "comm_bytes_flora": [],         # list[list[int]]
        "comm_bytes_fedavg_per_round": fedavg_bytes_per_round,  # int (constant)
        "mia_auc": [],                  # list[list[float]]
    }

    for round_num in range(1, config.num_rounds + 1):
        print(f"\n{'='*50}")
        print(f"Round {round_num}/{config.num_rounds}")
        print(f"{'='*50}")

        round_start = time.time()

        # Fit phase - sequential to keep one model in VRAM.
        # After each client trains, its adapter delta is immediately encrypted
        # to simulate wire transmission. The server decrypts before aggregation
        # using the client's key - raw weight values never exist outside each node.
        fit_start = time.time()
        all_encrypted, round_losses, round_comm_bytes, all_n_samples = [], [], [], []
        for client, cipher in zip(clients, server_cipher):
            # Server -> client: encrypt global params with the client's key before dispatch
            global_enc = cipher.encrypt(global_params)
            global_for_client = client.decrypt_parameters(global_enc)

            params, n_samples, metrics = client.fit(
                global_for_client, {"local_epochs": config.local_epochs}
            )
            # Client -> server: client encrypts its adapter delta before transmission
            encrypted = client.encrypt_parameters(params)
            all_encrypted.append(encrypted)
            all_n_samples.append(n_samples)
            round_losses.append(metrics["train_loss"])
            round_comm_bytes.append(len(encrypted))

        # Decrypt and aggregate
        all_params = [
            cipher.decrypt(enc)
            for cipher, enc in zip(server_cipher, all_encrypted)
        ]
        global_params = strategy.aggregate(round_num, all_params, all_n_samples)
        fit_elapsed = time.time() - fit_start

        history["train_loss"].append(round_losses)
        history["comm_bytes_flora"].append(round_comm_bytes)
        history["fit_latency_s"].append(fit_elapsed)

        # MIA per client - must run before evaluate() so the model still holds each
        # client's local adapter weights (what was actually transmitted on the wire).
        # evaluate() calls set_parameters(global_params) which would overwrite them.
        print(f"\n[MIA] Round {round_num}")
        mia_start = time.time()
        round_mia = []
        for i, client in enumerate(clients):
            client.set_parameters(all_params[i])  # restore local adapter for this client
            auc = client.mia_score(config.mia_n_members, config.mia_n_nonmembers)
            print(f"  [mia] bank_id={client._config.bank_id} AUC={auc:.3f}")
            round_mia.append(auc)
        history["mia_auc"].append(round_mia)
        history["mia_latency_s"].append(time.time() - mia_start)

        # Evaluate phase - loads global params into the model
        eval_start = time.time()
        round_f1, round_precision, round_recall = [], [], []
        for client, cipher in zip(clients, server_cipher):
            global_enc = cipher.encrypt(global_params)
            _, _, eval_metrics = client.evaluate(client.decrypt_parameters(global_enc), {})
            round_f1.append(eval_metrics["f1"])
            round_precision.append(eval_metrics["precision"])
            round_recall.append(eval_metrics["recall"])
        history["f1"].append(round_f1)
        history["precision"].append(round_precision)
        history["recall"].append(round_recall)
        history["eval_latency_s"].append(time.time() - eval_start)

        history["round_latency_s"].append(time.time() - round_start)

        if config.checkpoint_path:
            with open(config.checkpoint_path, "w") as f:
                json.dump(history, f)
            print(f"[checkpoint] Round {round_num} saved to {config.checkpoint_path}")

    print("\nFederated simulation complete.")
    return history
