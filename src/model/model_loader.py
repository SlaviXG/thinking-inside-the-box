import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.config import Config

# Known artefact characters from the DeepSeek-R1-Distill-Llama-8B fast tokenizer.
# The slow tokenizer (use_fast=False) prevents these, but the replacement is kept
# as a safety net in case the tokenizer config changes.
_SPACE_ARTEFACT = "\u0120"  # Ġ - byte-level BPE representation of space
_NEWLINE_ARTEFACT = "\u010a"  # Ċ - byte-level BPE representation of newline


def get_bnb_config() -> BitsAndBytesConfig:
    """Standard 4-bit NF4 quantization config for T4 VRAM budget (~6GB load)."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def load_tokenizer(config: Config) -> AutoTokenizer:
    """
    Load tokenizer with use_fast=config.use_fast_tokenizer.
    Default is False - required for DeepSeek-R1-Distill-Llama-8B to avoid
    Ġ/Ċ artefacts in decoded output.
    """
    return AutoTokenizer.from_pretrained(
        config.model_id,
        use_fast=config.use_fast_tokenizer,
    )


def load_model(config: Config) -> AutoModelForCausalLM:
    """
    Load base model with 4-bit quantization.
    Does NOT attach LoRA adapters - that is the federation layer's responsibility.
    device_map="auto" routes to GPU on Colab T4.
    """
    bnb_config = get_bnb_config() if config.load_in_4bit else None
    return AutoModelForCausalLM.from_pretrained(
        config.model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )


def decode_output(
    tokenizer: AutoTokenizer,
    output_ids: torch.Tensor,
    prompt_length: int,
) -> str:
    """
    Single canonical decode function for all generated outputs.
    This is the ONLY place the artefact replacement should exist in the codebase.
    """
    response = tokenizer.decode(
        output_ids[prompt_length:],
        skip_special_tokens=True,
    )
    return response.replace(_SPACE_ARTEFACT, " ").replace(_NEWLINE_ARTEFACT, "\n")
