import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

from src.config import Config

# Known artefact characters from the DeepSeek-R1-Distill-Llama-8B fast tokenizer.
# The slow tokenizer (use_fast=False) prevents these, but the replacement is kept
# as a safety net in case the tokenizer config changes.
_SPACE_ARTEFACT = "\u0120"    # Ġ - byte-level BPE representation of space
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
    pad_token is set to eos_token to support batch processing.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        use_fast=config.use_fast_tokenizer,
    )
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(config: Config) -> AutoModelForCausalLM:
    """
    Load base model with 4-bit quantization.
    Does NOT attach LoRA adapters - call attach_lora() separately.
    device_map="auto" routes to GPU on Colab T4.
    """
    bnb_config = get_bnb_config() if config.load_in_4bit else None
    return AutoModelForCausalLM.from_pretrained(
        config.model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )


def attach_lora(model: AutoModelForCausalLM, config: Config) -> AutoModelForCausalLM:
    """
    Attach trainable LoRA adapters to the frozen base model using PEFT.
    prepare_model_for_kbit_training() must be called before get_peft_model()
    when using 4-bit quantization - without it gradients won't flow through
    the frozen quantized layers to the adapters.
    """
    if config.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.lora_target_modules),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    return model


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
