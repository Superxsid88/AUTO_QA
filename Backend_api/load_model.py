"""
Model Loader Module (OpenAI-only)
=================================
This module now uses ONLY the OpenAI API.

- Text LLM (scoring/sentiment/summarization): gpt-5-mini (Chat Completions)
- ASR (optional): OpenAI Whisper transcription API
- All Hugging Face / local checkpoints removed.

Author: Rituraj (modified for OpenAI-only)
Date: 2025-05-15 → Updated
"""

import os
import re
import logging
from typing import Tuple, Optional

# Torch is kept only for device flags that other modules might touch; not required for OpenAI calls.
import torch

from logger import logger

# ──────────────────────────────────────────────────────────────
# OpenAI client (requires env var OPENAI_API_KEY)
# ──────────────────────────────────────────────────────────────
try:
    from openai import OpenAI
    _oa_client = OpenAI()  # reads OPENAI_API_KEY
except Exception as e:
    _oa_client = None
    logging.getLogger(__name__).warning(f"OpenAI client init failed: {e}")

# ==============================
# Device Configuration (kept)
# ==============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.deterministic = False
except Exception:
    pass

# =========================
# Cached Model Variables
# =========================
_cached_models = {
    "gpt5mini": None,          # (model_wrapper, tokenizer_shim)
    "gpt5mini_llama": None,    # alias wrapper for "llama" loader
    "whisper_shim": None,      # (whisper_shim, None)
}

# ======================================================================
# Shim classes to mimic earlier HF-ish interfaces (minimal + sufficient)
# ======================================================================

class TokenizerShim:
    """
    Mimics a Transformers tokenizer enough for your existing code:
    - __call__(text, return_tensors=..., truncation=...) → {"prompt": text}
    - decode(str, skip_special_tokens=True) → returns str unchanged
    """
    def __call__(self, text: str, return_tensors: Optional[str] = None, truncation: bool = True):
        return {"prompt": text}

    def decode(self, output, skip_special_tokens: bool = True) -> str:
        # In our generate(), we return a plain string; pass it through.
        return output if isinstance(output, str) else str(output)


class GPT5MiniWrapper:
    """
    Minimal wrapper around OpenAI Chat Completions with gpt-5-mini.
    Provides .generate(**inputs) to resemble HF generate() used in your code.
    """
    def __init__(self, client: OpenAI, model_name: str = "gpt-5-mini"):
        self.client = client
        self.model = model_name
        self.device = "openai"

    def generate(self, **inputs):
        """
        Expected pattern from prior code:
          inputs will contain {"prompt": <str>} produced by TokenizerShim.__call__.
        Respect 'max_new_tokens' and 'temperature' if provided.
        Returns a list-like where index [0] is the full generated string,
        so tokenizer.decode(out[0], ...) continues to work.
        """
        prompt = inputs.get("prompt", "")
        max_new_tokens = inputs.get("max_new_tokens", 900)
        temperature = inputs.get("temperature", 0.0)

        if _oa_client is None:
            raise RuntimeError("OpenAI client not initialized. Set OPENAI_API_KEY and install openai>=1.0.")

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=float(temperature or 0.0),
            max_tokens=int(max_new_tokens or 900),
        )
        text = (resp.choices[0].message.content or "").strip()
        # Return as a 1-item list so code using out[0] still works.
        return [text]


# Optional: thin shim for OpenAI Whisper API (transcription)
class WhisperShim:
    """
    Simple wrapper to call OpenAI's transcription endpoint.
    Usage:
        text = whisper.transcribe("path/to/audio.wav")
    """
    def __init__(self, client: OpenAI, model: str = "whisper-1"):
        self.client = client
        self.model = model

    def transcribe(self, audio_path: str, prompt: Optional[str] = None, language: Optional[str] = None) -> str:
        if self.client is None:
            raise RuntimeError("OpenAI client not initialized. Set OPENAI_API_KEY and install openai.")
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        with open(audio_path, "rb") as f:
            resp = self.client.audio.transcriptions.create(
                model=self.model,
                file=f,
                prompt=prompt,
                language=language
            )
        # SDK returns text in .text
        return getattr(resp, "text", "")


# ======================================================================
# Model Loader Functions (OpenAI-only)
# ======================================================================

def load_whisper_model() -> Tuple[WhisperShim, None]:
    """
    OpenAI-only Whisper loader.
    Returns a (WhisperShim, None) tuple to preserve the earlier (model, processor) shape.
    """
    if _cached_models["whisper_shim"] is not None:
        logger.info("OpenAI Whisper shim loaded from cache.")
        return _cached_models["whisper_shim"]

    if _oa_client is None:
        raise RuntimeError("OpenAI client not initialized. Ensure OPENAI_API_KEY is set.")

    whisper = WhisperShim(_oa_client, model="whisper-1")
    _cached_models["whisper_shim"] = (whisper, None)
    logger.info("OpenAI Whisper shim initialized and cached.")
    return _cached_models["whisper_shim"]


def load_mt5_large():
    """
    Compatibility loader for your previous MT5 usage.
    Now returns a (GPT5MiniWrapper, TokenizerShim) pair, both OpenAI-backed.
    """
    if _cached_models["gpt5mini"] is not None:
        logger.info("gpt-5-mini wrapper (MT5 alias) loaded from cache.")
        return _cached_models["gpt5mini"]

    if _oa_client is None:
        raise RuntimeError("OpenAI client not initialized. Ensure OPENAI_API_KEY is set.")

    model = GPT5MiniWrapper(_oa_client, model_name="gpt-5-mini")
    tok = TokenizerShim()
    _cached_models["gpt5mini"] = (model, tok)
    logger.info("Initialized gpt-5-mini wrapper (as MT5 replacement) and cached.")
    return _cached_models["gpt5mini"]


def load_llama_model_3b():
    """
    Compatibility loader for your previous LLaMA usage.
    Returns the same OpenAI-backed (GPT5MiniWrapper, TokenizerShim).
    """
    if _cached_models["gpt5mini_llama"] is not None:
        logger.info("gpt-5-mini wrapper (LLaMA alias) loaded from cache.")
        return _cached_models["gpt5mini_llama"]

    if _oa_client is None:
        raise RuntimeError("OpenAI client not initialized. Ensure OPENAI_API_KEY is set.")

    model = GPT5MiniWrapper(_oa_client, model_name="gpt-5-mini")
    tok = TokenizerShim()
    _cached_models["gpt5mini_llama"] = (model, tok)
    logger.info("Initialized gpt-5-mini wrapper (as LLaMA replacement) and cached.")
    return _cached_models["gpt5mini_llama"]


# Optional utility to clear cached wrappers (no GPU memory to free here)
def clear_mt5_cache():
    """Clear the cached GPT-5 Mini wrapper (MT5 alias) if you want to re-init."""
    if _cached_models["gpt5mini"] is not None:
        logger.info("Clearing gpt-5-mini (MT5 alias) cache...")
        _cached_models["gpt5mini"] = None
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception:
                pass
        logger.info("✓ gpt-5-mini (MT5 alias) cache cleared")
