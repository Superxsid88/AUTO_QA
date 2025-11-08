import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# OpenAI client (requires env var OPENAI_API_KEY)
# pip install --upgrade openai
# ──────────────────────────────────────────────────────────────
try:
    from openai import OpenAI
    _oa_client = OpenAI()  # reads OPENAI_API_KEY
except Exception as e:
    _oa_client = None
    logger.warning(f"OpenAI client init failed: {e}")


# ──────────────────────────────────────────────────────────────
# Shims to mimic Transformers-style API
# ──────────────────────────────────────────────────────────────
class _PromptShim:
    """Holds a prompt string and ignores .to('cuda') calls for compatibility."""
    def __init__(self, text: str) -> None:
        self.text = text
    def to(self, *_args, **_kwargs):
        return self  # no-op
    def __str__(self) -> str:
        return self.text


class _InputShim(dict):
    """
    Mapping-like object with .items() used in callers.
    Contains a single key: 'prompt' -> _PromptShim.
    """
    def __init__(self, prompt: str) -> None:
        super().__init__(prompt=_PromptShim(prompt))
    def to(self, *_args, **_kwargs):
        return self  # compatibility no-op


class TokenizerShim:
    """
    Minimal tokenizer interface:
      - __call__(text, return_tensors='pt') -> _InputShim({'prompt': _PromptShim(text)})
      - decode(output, skip_special_tokens=True) -> str(output)
    """
    def __call__(self, text: str, return_tensors: str = "pt", **_):
        return _InputShim(text)

    def decode(self, output, skip_special_tokens: bool = True) -> str:
        if isinstance(output, str):
            return output
        # If output is bytes or list-like, make a reasonable best effort:
        try:
            return str(output)
        except Exception:
            return ""


class GPT5MiniWrapper:
    """
    Simple wrapper around OpenAI Chat Completions (gpt-5-mini) that exposes .generate(**inputs).
    Returns a list-like where [0] is the generated string, so tokenizer.decode(outputs[0]) works.
    """
    def __init__(self, client: OpenAI, model_name: str = "gpt-5-mini"):
        self.client = client
        self.model = model_name
        self.device = "openai"

    def generate(self, **inputs):
        if self.client is None:
            raise RuntimeError("OpenAI client not initialized. Set OPENAI_API_KEY and install openai>=1.0.")

        prompt_obj = inputs.get("prompt")
        # Unwrap _PromptShim or accept plain strings
        prompt = str(prompt_obj) if prompt_obj is not None else ""

        # Map optional args to OpenAI params
        max_new_tokens = inputs.get("max_length", inputs.get("max_new_tokens", 50))
        temperature = inputs.get("temperature", 0.0)

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=float(temperature or 0.0),
            max_tokens=int(max_new_tokens or 50),
        )
        text = (resp.choices[0].message.content or "").strip()
        return [text]


# ──────────────────────────────────────────────────────────────
# Loader (OpenAI-only)
# ──────────────────────────────────────────────────────────────
def load_mt5_large():
    """
    Replacement loader that returns (model, tokenizer) but both are OpenAI-backed:
      - model: GPT5MiniWrapper (uses OpenAI gpt-5-mini)
      - tokenizer: TokenizerShim (mimics HF tokenizer interface)
    """
    if _oa_client is None:
        raise RuntimeError("OpenAI client not initialized. Ensure OPENAI_API_KEY is set and `pip install openai`.")

    logger.info("Initializing gpt-5-mini wrapper (OpenAI-only) ...")
    model = GPT5MiniWrapper(_oa_client, model_name="gpt-5-mini")
    tokenizer = TokenizerShim()
    logger.info("✓ gpt-5-mini wrapper ready")
    return model, tokenizer


# ──────────────────────────────────────────────────────────────
# Test the loader
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        model, tokenizer = load_mt5_large()
        logger.info("Model loaded successfully!")

        # Test inference (keep the same style, but no tensor .to() needed)
        test_text = "translate English to French: Hello, how are you?"
        logger.info(f"Testing with: {test_text}")

        inputs = tokenizer(test_text, return_tensors="pt")

        # Your old code tried moving tensors to CUDA; our shim ignores .to()
        # and still supports dict-style iteration safely.
        # If you keep this block, it will no-op without errors:
        inputs = {k: v.to("cuda") for k, v in inputs.items()}  # harmless no-op with shims

        outputs = model.generate(**inputs, max_length=50)  # maps to OpenAI max_tokens
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Result: {result}")

    except Exception as e:
        logger.error(f"Error: {e}")
