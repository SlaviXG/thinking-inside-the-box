import torch
from src.config import Config
from src.graph.base import GraphStore
from src.model.model_loader import decode_output
from src.pipeline.prompt_builder import build_investigation_prompt


class InvestigationPipeline:
    """
    Facade combining GraphStore retrieval, prompt construction, and LLM reasoning.
    This is the primary public API for both notebooks and Flower federation clients.

    Designed to be instantiated once per node and reused across multiple investigate()
    calls within a federation round to avoid repeated model loading overhead.
    """

    def __init__(
        self,
        graph_store: GraphStore,
        model,
        tokenizer,
        config: Config,
    ) -> None:
        self._graph = graph_store
        self._model = model
        self._tokenizer = tokenizer
        self._config = config
        self._device = next(model.parameters()).device

    def investigate(
        self,
        account_id: str,
        max_new_tokens: int = None,
        greedy: bool = True,
    ) -> str:
        """
        Full RAG + reasoning pass for one account.
        max_new_tokens overrides config.max_new_tokens when provided - used by
        evaluate() to cap generation length for speed without affecting qualitative runs.
        greedy=True (default) uses argmax decoding - required for stable binary
        verdict classification, where sampling adds ~5pp of pure noise to F1.
        Set greedy=False for qualitative reasoning runs that want diverse outputs.
        Returns the decoded LLM response string.
        """
        print(f"--- Investigating account {account_id} ---")

        # Step 1: retrieve graph context (Knowledge)
        context = self._graph.retrieve_context(
            account_id,
            limit=self._config.retrieval_limit,
            mode=self._config.retrieval_mode,
        )

        # Step 2: build prompt
        messages = build_investigation_prompt(account_id, context)

        # Step 3: tokenize (truncate from the left if the prompt blows the
        # configured budget - keeps the assistant_generation marker intact)
        inputs = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            truncation=True,
            max_length=self._config.max_prompt_tokens,
        ).to(self._device)

        # Step 4: generate (Reasoning) - eval mode, no gradient tracking
        n_tokens = max_new_tokens if max_new_tokens is not None else self._config.max_new_tokens
        self._model.eval()
        gen_kwargs = {
            "max_new_tokens": n_tokens,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        if greedy:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = self._config.temperature
        with torch.no_grad():
            outputs = self._model.generate(**inputs, **gen_kwargs)

        # Step 5: decode
        return decode_output(self._tokenizer, outputs[0], inputs.input_ids.shape[1])

    def batch_investigate(self, account_ids: list[str], greedy: bool = True) -> dict[str, str]:
        """
        Run investigate() over a list of accounts.
        Returns {account_id: response_str}.
        Used by the Flower client's fit() and evaluate() methods.
        """
        return {acc_id: self.investigate(acc_id, greedy=greedy) for acc_id in account_ids}
