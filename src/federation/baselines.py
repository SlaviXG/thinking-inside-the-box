import time

from src.config import Config
from src.federation.client import AMLFederatedClient
from src.model.model_loader import load_model, load_tokenizer, attach_lora


def run_centralised(config: Config, model=None, tokenizer=None) -> dict:
    """
    Centralised baseline: single model trained on the merged data of all
    benchmark bank partitions (no federation, no privacy guarantees).

    This is the utility upper bound - the best F1 achievable when all banks
    pool their data and train together. It serves as the ceiling against which
    the federated FLoRA+RAG system is benchmarked.

    When config.bank_ids is set, only those banks are merged (same total data
    as the federated scenario). When bank_ids is None, all banks are used.

    Each "round" corresponds to one pass of local_epochs training epochs,
    matching the federated schedule so round-by-round F1 curves are comparable.

    Returns a history dict in the same format as start_server(), minus the
    federation-specific keys (comm_bytes_flora, comm_bytes_fedavg_per_round,
    mia_auc - these have no meaning for a centralised model).
    """
    if model is None or tokenizer is None:
        print("Loading base model and tokenizer...")
        model = load_model(config)
        tokenizer = load_tokenizer(config)
        model = attach_lora(model, config)
        print("Model ready.\n")
    else:
        print("Reusing provided model and tokenizer.\n")

    # bank_id=0 with bank_ids set merges only the specified partitions
    centralised_config = Config.from_dict({**config.__dict__, "bank_id": 0})
    print("[Centralised] Loading merged partition...")
    client = AMLFederatedClient(centralised_config, model, tokenizer)

    params = client.get_parameters()

    history = {
        "train_loss": [],   # list[list[float]] - wraps in list for format parity
        "f1": [],
        "precision": [],
        "recall": [],
        "round_latency_s": [],
    }

    for round_num in range(1, config.num_rounds + 1):
        print(f"\n{'='*50}")
        print(f"[Centralised] Round {round_num}/{config.num_rounds}")
        print(f"{'='*50}")

        round_start = time.time()
        params, n_samples, fit_metrics = client.fit(
            params, {"local_epochs": config.local_epochs}
        )
        _, _, eval_metrics = client.evaluate(params, {})
        elapsed = time.time() - round_start

        # Wrap in list to match start_server() history format (per-client lists)
        history["train_loss"].append([fit_metrics["train_loss"]])
        history["f1"].append([eval_metrics["f1"]])
        history["precision"].append([eval_metrics["precision"]])
        history["recall"].append([eval_metrics["recall"]])
        history["round_latency_s"].append(elapsed)

    print("\nCentralised baseline complete.")
    return history
