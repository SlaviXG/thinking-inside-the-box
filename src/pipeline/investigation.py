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

    def investigate(self, account_id: str) -> str:
        """
        Full RAG + reasoning pass for one account.
        Returns the decoded LLM response string.
        """
        print(f"--- Investigating account {account_id} ---")

        # Step 1: retrieve graph context (Knowledge)
        context = self._graph.retrieve_context(
            account_id, limit=self._config.retrieval_limit
        )

        # Step 2: build prompt
        messages = build_investigation_prompt(account_id, context)

        # Step 3: tokenize
        inputs = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to("cuda")

        # Step 4: generate (Reasoning)
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=self._config.max_new_tokens,
            temperature=self._config.temperature,
            do_sample=True,
            pad_token_id=self._tokenizer.eos_token_id,
        )

        # Step 5: decode
        return decode_output(self._tokenizer, outputs[0], inputs.input_ids.shape[1])

    def batch_investigate(self, account_ids: list[str]) -> dict[str, str]:
        """
        Run investigate() over a list of accounts.
        Returns {account_id: response_str}.
        Used by the Flower client's fit() and evaluate() methods.
        """
        return {acc_id: self.investigate(acc_id) for acc_id in account_ids}
