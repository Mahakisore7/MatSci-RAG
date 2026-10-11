"""Text generation: Groq-hosted model first, local Flan-T5 as the fallback.

torch/transformers are imported lazily so this module (and its tests) stay light,
and so they always load after HuggingFaceEmbeddings (Windows DLL order, ADR-008).
"""

import os

from src.core.config import GEN_BACKEND, GEN_MODEL_NAME
from src.core.logging_config import get_logger

log = get_logger(__name__)

_gen_model = None
_gen_tokenizer = None
_device = None
_groq_client = None


def get_llm():
    global _gen_model, _gen_tokenizer, _device
    if _gen_model is not None:
        return _gen_model, _gen_tokenizer, _device

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Loading local generator '{GEN_MODEL_NAME}' on {_device} ...")
    _gen_tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
    _gen_model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_NAME).to(_device)
    return _gen_model, _gen_tokenizer, _device


def _groq_generate(prompt: str, model: str, max_tokens: int) -> str:
    global _groq_client
    if _groq_client is None:
        from groq import Groq

        _groq_client = Groq(timeout=30.0, max_retries=2)
    kwargs = {"reasoning_effort": "low"} if "gpt-oss" in model else {"reasoning_format": "hidden"}
    response = _groq_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        # Reasoning tokens count against max_tokens on these models.
        max_tokens=max_tokens + 1024,
        **kwargs,
    )
    text = (response.choices[0].message.content or "").strip()
    if not text:
        raise ValueError("empty completion")
    return text


def _local_generate(prompt: str, max_new_tokens: int) -> str:
    import torch

    model, tokenizer, device = get_llm()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def generate(prompt: str, groq_model: str, max_new_tokens: int) -> tuple[str, str]:
    """Returns (text, model_used). Falls back to the local model if Groq is unavailable."""
    if GEN_BACKEND == "groq" and os.environ.get("GROQ_API_KEY"):
        try:
            return _groq_generate(prompt, groq_model, max_new_tokens), groq_model
        except Exception as e:
            log.warning(f"Groq generation failed ({e}); falling back to {GEN_MODEL_NAME}")
    return _local_generate(prompt, max_new_tokens), GEN_MODEL_NAME
