from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Config:
    # Data
    csv_path: str = "data/LI-Small_Trans.csv"
    db_base_dir: str = "data/kuzu_dbs"
    bank_id: int = 0  # Which bank partition this node owns

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
    use_fast_tokenizer: bool = False  # False required for this model - fast tokenizer produces Ġ/Ċ artefacts
    load_in_4bit: bool = True
    max_new_tokens: int = 1024
    temperature: float = 0.3

    # Federation
    num_rounds: int = 3
    num_clients: int = 3
    lora_rank: int = 8
    lora_alpha: int = 16

    @staticmethod
    def from_dict(d: dict) -> "Config":
        return Config(**{k: v for k, v in d.items() if k in Config.__dataclass_fields__})
