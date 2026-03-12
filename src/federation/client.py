import numpy as np
import torch
from torch.optim import AdamW
from sklearn.metrics import f1_score, precision_score, recall_score
import flwr as fl

from src.config import Config
from src.data.aml_ingestor import AMLIngestor
from src.graph.base import GraphStore
from src.graph.factory import GraphStoreFactory
from src.model.model_loader import decode_output
from src.pipeline.investigation import InvestigationPipeline
from src.pipeline.prompt_builder import build_investigation_prompt

_VERDICT_SUSPICIOUS = "VERDICT: SUSPICIOUS"
_VERDICT_CLEAN = "VERDICT: CLEAN"


def _parse_verdict(response: str) -> int:
    """Extract binary AML prediction from LLM response text."""
    upper = response.upper()
    if "SUSPICIOUS" in upper or "LAUNDERING" in upper:
        return 1
    return 0


def _lora_params(model) -> dict:
    """Return only the trainable LoRA adapter parameters, sorted for deterministic ordering."""
    return dict(sorted(
        {k: v for k, v in model.named_parameters() if "lora_" in k}.items()
    ))


class AMLFlowerClient(fl.client.NumPyClient):
    """
    Flower client representing one bank node in the federation.

    On init: loads its partition, splits into train/val/test,
    ingests all data into its local Kuzu graph store, and wires
    up the InvestigationPipeline.

    The base model + LoRA adapters are passed in (loaded once on server)
    to avoid reloading 6GB of weights per client on the T4.
    """

    def __init__(self, config: Config, model, tokenizer) -> None:
        self._config = config
        self._model = model
        self._tokenizer = tokenizer
        self._device = next(model.parameters()).device

        # Load partition and split into train/val/test
        ingestor = AMLIngestor(config)
        df = ingestor.load_partition()
        self._train_df, self._val_df, self._test_df = ingestor.split(df)

        # Ingest full partition (train+val+test) into graph for RAG retrieval
        # Store reference so we can close it cleanly on destruction
        self._graph_store: GraphStore = GraphStoreFactory.create(config)
        ingestor.run_from_df(self._graph_store, df)

        self._pipeline = InvestigationPipeline(
            self._graph_store, model, tokenizer, config
        )

        print(
            f"[Client bank_id={config.bank_id}] "
            f"train={len(self._train_df)} val={len(self._val_df)} test={len(self._test_df)} accounts"
        )

    def __del__(self) -> None:
        if hasattr(self, "_graph_store"):
            self._graph_store.close()

    # --- Flower parameter interface ---

    def get_parameters(self, config=None) -> list[np.ndarray]:
        """
        Return LoRA adapter weights as numpy arrays.
        Ordering: all lora_A matrices first (sorted by name), then all lora_B matrices.
        This convention is relied upon by FLoRAStrategy for correct stacking.
        """
        params = _lora_params(self._model)
        a_vals = [v.detach().cpu().float().numpy() for k, v in params.items() if "lora_A" in k]
        b_vals = [v.detach().cpu().float().numpy() for k, v in params.items() if "lora_B" in k]
        return a_vals + b_vals

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        """
        Load adapter weights received from the server.
        Expects the same ordering as get_parameters(): all A's then all B's.
        """
        params = _lora_params(self._model)
        a_items = [(k, v) for k, v in params.items() if "lora_A" in k]
        b_items = [(k, v) for k, v in params.items() if "lora_B" in k]
        n = len(a_items)

        for (_, param), arr in zip(a_items, parameters[:n]):
            param.data = torch.tensor(arr, dtype=param.dtype).to(param.device)
        for (_, param), arr in zip(b_items, parameters[n:]):
            param.data = torch.tensor(arr, dtype=param.dtype).to(param.device)

    # --- Training ---

    def _build_training_example(
        self, account_id: str, is_laundering: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build (input_ids, attention_mask, labels) for one supervised training example.
        Labels are -100 for all prompt tokens (ignored in cross-entropy loss)
        and target token ids for the verdict response.
        """
        context = self._pipeline._graph.retrieve_context(
            account_id, limit=self._config.retrieval_limit
        )
        messages = build_investigation_prompt(account_id, context)
        target_text = _VERDICT_SUSPICIOUS if is_laundering else _VERDICT_CLEAN

        prompt = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._device)

        target_ids = self._tokenizer(
            target_text + self._tokenizer.eos_token,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids.to(self._device)

        input_ids = torch.cat([prompt.input_ids, target_ids], dim=1)
        attention_mask = torch.cat([
            prompt.attention_mask,
            torch.ones_like(target_ids),
        ], dim=1)
        labels = torch.cat([
            torch.full_like(prompt.input_ids, -100),
            target_ids,
        ], dim=1)

        return input_ids, attention_mask, labels

    def fit(self, parameters, config) -> tuple[list[np.ndarray], int, dict]:
        """
        Local training round on the train split.
        Fine-tunes LoRA adapters to produce correct SUSPICIOUS/CLEAN verdicts.
        """
        self.set_parameters(parameters)
        self._model.train()

        optimizer = AdamW(
            [p for p in self._model.parameters() if p.requires_grad],
            lr=self._config.learning_rate,
        )

        sample = self._train_df.sample(
            n=min(self._config.max_train_samples, len(self._train_df)),
            random_state=None,
        )

        total_loss = 0.0
        n_trained = 0
        local_epochs = config.get("local_epochs", self._config.local_epochs)

        for _ in range(local_epochs):
            for _, row in sample.iterrows():
                account_id = str(row["account_id"])
                label = int(row["label"])
                try:
                    input_ids, attention_mask, labels = self._build_training_example(
                        account_id, label
                    )
                    optimizer.zero_grad()
                    outputs = self._model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    outputs.loss.backward()
                    optimizer.step()
                    total_loss += outputs.loss.item()
                    n_trained += 1
                except (RuntimeError, ValueError, KeyError) as e:
                    print(f"  [fit] skipping {account_id}: {e}")
                    continue

        self._model.eval()
        avg_loss = total_loss / max(n_trained, 1)
        print(f"  [fit] bank_id={self._config.bank_id} loss={avg_loss:.4f} samples={n_trained}")

        return self.get_parameters(), n_trained, {"train_loss": avg_loss}

    # --- Evaluation ---

    def evaluate(self, parameters, config) -> tuple[float, int, dict]:
        """
        Evaluate on the test split.
        Parses LLM verdicts and computes F1 against ground truth Is Laundering labels.
        Returns (loss=1-F1, num_examples, metrics).
        """
        self.set_parameters(parameters)
        self._model.eval()

        sample = self._test_df.sample(
            n=min(self._config.max_eval_samples, len(self._test_df)),
            random_state=None,  # consistent with fit() - no fixed seed
        )

        y_true, y_pred = [], []
        for _, row in sample.iterrows():
            account_id = str(row["account_id"])
            true_label = int(row["label"])
            try:
                response = self._pipeline.investigate(account_id)
                pred_label = _parse_verdict(response)
            except (RuntimeError, ValueError, KeyError) as e:
                print(f"  [eval] skipping {account_id}: {e}")
                pred_label = 0
            y_true.append(true_label)
            y_pred.append(pred_label)

        if not y_true:
            return 0.0, 0, {"f1": 0.0, "precision": 0.0, "recall": 0.0}

        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))

        print(
            f"  [eval] bank_id={self._config.bank_id} "
            f"F1={f1:.3f} P={precision:.3f} R={recall:.3f}"
        )

        return 1.0 - f1, len(y_true), {
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }
