from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Config:
    # Data
    csv_path: str = "data/LI-Small_Trans.csv"
    db_base_dir: str = "data/kuzu_dbs"
    bank_id: int = 0  # Which bank partition this node owns; 0 = all banks

    # Train/Val/Test split ratios (must sum to 1.0)
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    # test_ratio is implicit: 1 - train_ratio - val_ratio = 0.15

    # Graph backend
    graph_backend: Literal["kuzu", "networkx", "neo4j"] = "kuzu"
    retrieval_limit: int = 20   # Max transactions returned per context query
    retrieval_depth: int = 2    # Hop depth (used by NetworkX backend)

    # Neo4j (only needed if graph_backend="neo4j")
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"

    # Model
    model_id: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    use_fast_tokenizer: bool = False  # False required - fast tokenizer produces Ġ/Ċ artefacts
    load_in_4bit: bool = True
    max_new_tokens: int = 1024
    temperature: float = 0.3

    # LoRA
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: tuple = ("q_proj", "v_proj")

    # Federation training
    num_rounds: int = 3
    num_clients: int = 3
    local_epochs: int = 1
    learning_rate: float = 2e-4
    max_train_samples: int = 100   # Accounts sampled per fit() round (GPU budget)
    max_eval_samples: int = 50     # Accounts sampled per evaluate() round

    @staticmethod
    def from_dict(d: dict) -> "Config":
        return Config(**{k: v for k, v in d.items() if k in Config.__dataclass_fields__})
