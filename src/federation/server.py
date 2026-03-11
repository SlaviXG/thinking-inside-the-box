import flwr as fl

from src.config import Config
from src.federation.client import AMLFlowerClient
from src.model.model_loader import load_model, load_tokenizer


def build_client_fn(config_template: Config, model, tokenizer):
    """
    Returns a client_fn(cid: str) -> AMLFlowerClient closure.
    cid is cast to int and used as config.bank_id so each simulated
    client gets its own data partition and .db file.
    """
    def client_fn(cid: str) -> AMLFlowerClient:
        config = Config.from_dict({
            **config_template.__dict__,
            "bank_id": int(cid),
        })
        return AMLFlowerClient(config, model, tokenizer)

    return client_fn


def start_server(config: Config) -> None:
    """
    Launch Flower simulation via fl.simulation.start_simulation().
    Model and tokenizer are loaded once here and shared across all
    simulated clients via the client_fn closure - critical for T4 VRAM budget.
    """
    print("Loading model and tokenizer (shared across all clients)...")
    model = load_model(config)
    tokenizer = load_tokenizer(config)

    client_fn = build_client_fn(config, model, tokenizer)

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=config.num_clients,
        config=fl.server.ServerConfig(num_rounds=config.num_rounds),
        strategy=fl.server.strategy.FedAvg(),
        # TODO Phase 3: replace FedAvg with FLoRA adapter aggregation strategy
    )
