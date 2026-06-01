"""
ai_waf_v2.utils.llm
-------------------
Unified LLM call abstraction used across pipeline stages.

Supported providers
-------------------
  anthropic  — Anthropic Messages API (requires ANTHROPIC_API_KEY)
  google     — Google GenerativeAI API (requires GOOGLE_API_KEY)
  local      — llama-cpp-python (GGUF, fully offline, no API key)
  ollama     — Ollama HTTP API   (local server, no API key)

All variants accept a system prompt and a user prompt, and return the
model's raw text response (str) or None on failure.  JSON parsing and
any further structure extraction are the caller's responsibility.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

log = logging.getLogger(__name__)

# Set LLM_DEBUG=1 to log a preview of every model response at DEBUG level.
if os.environ.get("LLM_DEBUG"):
    log.setLevel(logging.DEBUG)

# Module-level cache for the llama-cpp Llama instance (avoids reloading weights)
_llama_cache: dict[str, Any] = {}


def call_llm(
    provider:        str,
    system:          str,
    user:            str,
    model:           str  = "",
    max_tokens:      int  = 512,
    temperature:     float = 0.7,
    model_path:      str  = "mistral",
    ollama_base_url: str  = "http://localhost:11434",
    request_timeout: int  = 30,
) -> str | None:
    """
    Call an LLM and return the raw text response, or None on failure.

    Parameters
    ----------
    provider : str
        One of 'anthropic', 'google', 'local', 'ollama'.
    system : str
        System prompt.
    user : str
        User / human turn content.
    model : str
        Model identifier for cloud providers (e.g. 'claude-sonnet-4-6').
        Ignored for 'local'; used as the Ollama model tag for 'ollama'.
    max_tokens : int
        Maximum output tokens.
    temperature : float
        Sampling temperature.
    model_path : str
        Path to a GGUF file.  Required when provider='local'.
    ollama_base_url : str
        Base URL of the Ollama server (default: http://localhost:11434).
        Used only when provider='ollama'.

    Returns
    -------
    str | None
        Raw model output text, or None if the call failed.
    """
    if provider == "anthropic":
        text = _call_anthropic(system, user, model, max_tokens, temperature)
    elif provider == "google":
        text = _call_google(system, user, model, max_tokens, temperature)
    elif provider == "local":
        text = _call_local(system, user, model_path, max_tokens, temperature)
    elif provider == "ollama":
        text = _call_ollama(system, user, model, max_tokens, temperature, ollama_base_url, request_timeout)
    else:
        log.warning(f"Unknown LLM provider: {provider!r}")
        return None

    if text is not None:
        preview = text[:200].replace("\n", "\\n")
        log.debug(f"[{provider}] response preview: {preview!r}")
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Provider implementations
# ─────────────────────────────────────────────────────────────────────────────

def _call_anthropic(
    system:      str,
    user:        str,
    model:       str,
    max_tokens:  int,
    temperature: float,
) -> str | None:
    try:
        import anthropic
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return msg.content[0].text
    except ImportError:
        log.warning("anthropic package not installed — pip install anthropic")
        return None
    except Exception as e:
        log.warning(f"Anthropic API error ({type(e).__name__}): {e}")
        return None


def _call_google(
    system:      str,
    user:        str,
    model:       str,
    max_tokens:  int,
    temperature: float,
) -> str | None:
    if not model:
        log.warning("Google provider requires a model name — set augmentation.llm.model in pipeline.yaml")
        return None
    try:
        import google.generativeai as genai
        genai.configure()
        gm  = genai.GenerativeModel(model_name=model, system_instruction=system)
        rsp = gm.generate_content(
            user,
            generation_config={"max_output_tokens": max_tokens, "temperature": temperature},
        )
        return rsp.text
    except ImportError:
        log.warning("google-generativeai package not installed — pip install google-generativeai")
        return None
    except Exception as e:
        log.warning(f"Google API error ({type(e).__name__}): {e}")
        return None


def _call_local(
    system:      str,
    user:        str,
    model_path:  str,
    max_tokens:  int,
    temperature: float,
) -> str | None:
    if not model_path:
        log.warning("LLM provider='local' but model_path is empty")
        return None
    try:
        if model_path not in _llama_cache:
            from llama_cpp import Llama
            _llama_cache[model_path] = Llama(
                model_path=model_path,
                n_ctx=2048, n_gpu_layers=-1, verbose=False,
            )
            log.info(f"Loaded local LLM from {model_path}")
        llm = _llama_cache[model_path]
        prompt = f"[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST]"
        resp = llm(prompt, max_tokens=max_tokens, temperature=temperature, stop=["[INST]"])
        return resp["choices"][0]["text"].strip()
    except ImportError:
        log.warning(
            "llama-cpp-python not installed. "
            "pip install llama-cpp-python --extra-index-url "
            "https://abetlen.github.io/llama-cpp-python/whl/cu124"
        )
        return None
    except Exception as e:
        log.warning(f"Local LLM error ({type(e).__name__}): {e}")
        return None


def _call_ollama(
    system:      str,
    user:        str,
    model:       str,
    max_tokens:  int,
    temperature: float,
    base_url:    str,
    timeout:     int = 30,
) -> str | None:
    import socket
    import urllib.error
    import urllib.request

    if not model:
        log.warning("Ollama provider requires a model name — set augmentation.llm.model in pipeline.yaml")
        return None
    try:
        payload = json.dumps({
            "model":  model,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": temperature},
            "messages": [
                {"role": "system",  "content": system},
                {"role": "user",    "content": user},
            ],
        }).encode()
        req = urllib.request.Request(
            f"{base_url.rstrip('/')}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        return data["message"]["content"]
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:300]
        if e.code == 404:
            log.warning(f"Ollama model not found (404): model={model!r} — run: ollama pull {model}")
        elif e.code == 503:
            log.warning(f"Ollama model loading/busy (503): model={model!r}")
        else:
            log.warning(f"Ollama HTTP {e.code}: {body}")
        return None
    except urllib.error.URLError as e:
        reason = str(e.reason).lower()
        if "refused" in reason:
            log.warning(f"Ollama connection refused — is Ollama running at {base_url}?")
        else:
            log.warning(f"Ollama connection error: {e.reason}")
        return None
    except (TimeoutError, socket.timeout):
        log.warning(f"Ollama request timed out after {timeout}s (model={model!r}, max_tokens={max_tokens})")
        return None
    except json.JSONDecodeError as e:
        log.warning(f"Ollama returned invalid JSON: {e}")
        return None
    except Exception as e:
        log.warning(f"Ollama unexpected error ({type(e).__name__}): {e}")
        return None
