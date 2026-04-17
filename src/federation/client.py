import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch.optim import AdamW

from src.config import Config
from src.data.aml_ingestor import AMLIngestor
from src.graph.base import GraphStore
from src.graph.factory import GraphStoreFactory
from src.pipeline.investigation import InvestigationPipeline
from src.pipeline.prompt_builder import build_investigation_prompt
from src.security.encryption import AdapterEncryption

_VERDICT_SUSPICIOUS = "VERDICT: SUSPICIOUS"
_VERDICT_CLEAN = "VERDICT: CLEAN"


def _parse_verdict(response: str) -> int:
    """
    Extract binary AML prediction from LLM response text.

    Checks for the explicit verdict format the model is trained to produce first.
    This avoids false positives when the model mentions 'suspicious' in reasoning
    but concludes clean (e.g. 'I see no suspicious activity ... VERDICT: CLEAN').
    Falls back to keyword matching for early rounds before fine-tuning takes effect.
    """
    upper = response.upper()
    if "VERDICT: SUSPICIOUS" in upper:
        return 1
    if "VERDICT: CLEAN" in upper:
        return 0
    # Fallback: model hasn't learned the verdict format yet
    if "SUSPICIOUS" in upper or "LAUNDERING" in upper:
        return 1
    return 0


def _lora_params(model) -> dict:
    """Return only the trainable LoRA adapter parameters, sorted for deterministic ordering."""
    return dict(sorted(
        {k: v for k, v in model.named_parameters() if "lora_" in k}.items()
    ))


class AMLFederatedClient:
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

        train_pos = int(self._train_df["label"].sum())
        train_neg = len(self._train_df) - train_pos
        test_pos  = int(self._test_df["label"].sum())

        print(
            f"[Client bank_id={config.bank_id}] "
            f"train={len(self._train_df)} ({train_pos} pos / {train_neg} neg)  "
            f"val={len(self._val_df)}  "
            f"test={len(self._test_df)} ({test_pos} pos)"
        )
        if train_pos == 0:
            print(f"  WARNING bank_id={config.bank_id}: no positive training accounts - "
                  "F1 will be 0 regardless of training. Consider a different bank partition.")

        # Each client owns its own encryption key. The server uses client.encryption_key
        # to decrypt adapter deltas after transmission. Raw weight values are never
        # carried on the wire in plaintext.
        self._encryption = AdapterEncryption()

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

    # --- Encrypted parameter transmission ---

    @property
    def encryption_key(self) -> bytes:
        """Public key used by the server to decrypt this client's transmissions."""
        return self._encryption.key

    def encrypt_parameters(self, params: list[np.ndarray]) -> bytes:
        """Encrypt adapter deltas before they leave this node."""
        return self._encryption.encrypt(params)

    def decrypt_parameters(self, payload: bytes) -> list[np.ndarray]:
        """Decrypt adapter deltas received from the server (e.g. global params)."""
        return self._encryption.decrypt(payload)

    # --- Training ---

    def _build_target_text(self, account_id: str, is_laundering: int) -> str:
        """
        Rationale-augmented target for rag retrieval mode: the label is prefixed
        with a short sentence naming the locally-detected AML patterns (or their
        absence). This gives the LoRA adapter gradient signal on the pattern
        tokens so it learns to map topology signals to the verdict, rather than
        learning P(verdict | prompt_hash) while ignoring the topology block.
        Flat mode keeps the bare verdict - there are no signals to reason over.
        """
        verdict = _VERDICT_SUSPICIOUS if is_laundering else _VERDICT_CLEAN
        if self._config.retrieval_mode != "rag":
            return verdict
        signals = self._pipeline._graph.structural_signals(account_id)
        if signals:
            rationale = f"Structural signals: {', '.join(signals)}. "
        else:
            rationale = "No structural AML patterns detected. "
        return rationale + verdict

    def _build_training_example(
        self, account_id: str, is_laundering: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build (input_ids, attention_mask, labels) for one supervised training example.
        Labels are -100 for all prompt tokens (ignored in cross-entropy loss)
        and target token ids for the rationale + verdict response.
        """
        context = self._pipeline._graph.retrieve_context(
            account_id, limit=self._config.retrieval_limit,
            mode=self._config.retrieval_mode,
        )
        messages = build_investigation_prompt(account_id, context)
        target_text = self._build_target_text(account_id, is_laundering)

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

    def _compute_loss(self, account_id: str, label: int) -> float | None:
        """Compute cross-entropy loss for one account without updating weights."""
        try:
            input_ids, attention_mask, labels = self._build_training_example(account_id, label)
            with torch.no_grad():
                outputs = self._model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
            return outputs.loss.item()
        except (RuntimeError, ValueError, KeyError):
            return None

    def _balanced_sample(self, n: int) -> pd.DataFrame:
        """
        Build a 50/50 balanced training batch of n total accounts.

        The IBM AML dataset has ~0.05% laundering at transaction level, translating
        to very few positive accounts per bank partition. Random sampling almost
        never includes enough positives for the model to learn the SUSPICIOUS class.

        Strategy: take n//2 suspicious accounts (oversample with replacement if
        fewer than n//2 exist) and n//2 clean accounts, then shuffle.
        This guarantees the model sees equal class representation every round.
        """
        pos = self._train_df[self._train_df["label"] == 1]
        neg = self._train_df[self._train_df["label"] == 0]
        n_each = n // 2

        pos_sample = pos.sample(
            n=n_each, replace=len(pos) < n_each, random_state=None
        )
        neg_sample = neg.sample(
            n=n_each, replace=len(neg) < n_each, random_state=None
        )
        return (
            pd.concat([pos_sample, neg_sample])
            .sample(frac=1, random_state=None)  # shuffle
            .reset_index(drop=True)
        )

    def fit(self, parameters, config) -> tuple[list[np.ndarray], int, dict]:
        """
        Local training round on the train split.
        Fine-tunes LoRA adapters to produce correct SUSPICIOUS/CLEAN verdicts.
        Uses balanced 50/50 sampling to counter the extreme class imbalance in IBM AML.
        """
        self.set_parameters(parameters)
        self._model.train()

        optimizer = AdamW(
            [p for p in self._model.parameters() if p.requires_grad],
            lr=self._config.learning_rate,
        )

        n = min(self._config.max_train_samples, len(self._train_df))
        sample = self._balanced_sample(n)

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

    def _eval_sample(self) -> pd.DataFrame:
        """
        Balanced evaluation sample: 50/50 positive/negative, sized per the
        max_eval_samples budget.

        max_eval_samples == 0: full test split (rigorous but expensive; use
            for final publication runs).
        max_eval_samples  > 0: target total sample size. Each class is capped
            at max_eval_samples // 2 and at class availability (no upsampling
            with replacement - greedy decoding makes duplicate accounts
            produce identical predictions, so replacement adds no stability).

        The primary F1-stability mechanism is greedy decoding in
        InvestigationPipeline; this method just caps cost.
        """
        if self._config.max_eval_samples == 0:
            return self._test_df

        pos = self._test_df[self._test_df["label"] == 1]
        neg = self._test_df[self._test_df["label"] == 0]
        if len(pos) == 0 or len(neg) == 0:
            return self._test_df  # cannot balance - fall back to whatever is there

        cap_each = self._config.max_eval_samples // 2
        n_each = min(cap_each, len(pos), len(neg))
        if n_each == 0:
            return self._test_df

        return (
            pd.concat([
                pos.sample(n=n_each, random_state=42),
                neg.sample(n=n_each, random_state=42),
            ])
            .sample(frac=1, random_state=42)
            .reset_index(drop=True)
        )

    def evaluate(self, parameters, config) -> tuple[float, int, dict]:
        """
        Evaluate on the test split.
        Parses LLM verdicts and computes F1 against ground truth Is Laundering labels.
        Uses stratified sampling (_eval_sample) to guarantee positive class coverage.
        Returns (loss=1-F1, num_examples, metrics).
        """
        self.set_parameters(parameters)
        self._model.eval()

        sample = self._eval_sample()

        y_true, y_pred = [], []
        for _, row in sample.iterrows():
            account_id = str(row["account_id"])
            true_label = int(row["label"])
            try:
                response = self._pipeline.investigate(
                    account_id, max_new_tokens=self._config.max_eval_tokens
                )
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

    # --- Privacy: Membership Inference Attack ---

    def mia_score(self, n_members: int, n_nonmembers: int) -> float:
        """
        Loss-based Membership Inference Attack (Yeom et al., 2018).

        Computes per-sample cross-entropy loss on training accounts (members)
        and test accounts (non-members) using the current adapter weights.
        Fits a threshold classifier on those losses and returns AUC.

        AUC interpretation:
          0.5 = random chance = adapter deltas carry no membership signal (privacy holds)
          1.0 = perfect membership prediction = full leakage

        The adversary model: an attacker who intercepts adapter weight updates
        on the wire runs inference on candidate accounts and uses loss as a
        membership score. This is the weakest realistic attack - if it fails,
        stronger attacks are unlikely to succeed.
        """
        self._model.eval()

        member_sample = self._train_df.sample(
            n=min(n_members, len(self._train_df)), random_state=42
        )
        nonmember_sample = self._test_df.sample(
            n=min(n_nonmembers, len(self._test_df)), random_state=42
        )

        losses, labels = [], []

        for _, row in member_sample.iterrows():
            loss = self._compute_loss(str(row["account_id"]), int(row["label"]))
            if loss is not None:
                losses.append(loss)
                labels.append(1)

        for _, row in nonmember_sample.iterrows():
            loss = self._compute_loss(str(row["account_id"]), int(row["label"]))
            if loss is not None:
                losses.append(loss)
                labels.append(0)

        if len(set(labels)) < 2:
            return 0.5  # cannot compute AUC without both classes present

        # Lower loss -> more likely a member; negate so higher score = more likely member
        neg_losses = [-l for l in losses]
        return float(roc_auc_score(labels, neg_losses))
