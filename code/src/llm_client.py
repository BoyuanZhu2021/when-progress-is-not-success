"""Unified OpenAI-compatible chat client (stdlib only, no requests/openai dep).

Handles the Qwen "thinking" channel (`reasoning_content`) and the `enable_thinking`
switch (SiliconFlow Qwen3.x), with retry/backoff on transient errors. Returns a
normalized dict so callers don't parse provider JSON.

Verified behavior (2026-06-29, SiliconFlow):
  - enable_thinking=False -> clean `content`, empty reasoning, ~2s.
  - enable_thinking=True  -> CoT in `reasoning_content`; needs >=512 max_tokens
    or `content` comes back empty (budget eaten by hidden reasoning).
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

try:
    from .providers import resolve_provider
except ImportError:  # imported as a top-level module (src on sys.path)
    from providers import resolve_provider

RETRYABLE = {408, 409, 429, 500, 502, 503, 504}


class LLMError(Exception):
    pass


def _post(url: str, headers: dict, body: dict, timeout: float) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST", headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def chat(
    provider: str,
    model: str,
    messages: list[dict],
    *,
    max_tokens: int = 512,
    temperature: float = 0.7,
    enable_thinking: bool | None = None,
    seed: int | None = None,
    timeout: float = 120.0,
    retries: int = 3,
    extra: dict | None = None,
) -> dict:
    """Return {content, reasoning, usage, latency, model, provider, finish_reason}.

    `enable_thinking` and `seed` are only sent when not None (non-Qwen providers
    ignore/omit them). Raises LLMError after exhausting retries.
    """
    base, key = resolve_provider(provider)
    url = base.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body: dict = {"model": model, "messages": messages,
                  "max_tokens": max_tokens, "temperature": temperature}
    if enable_thinking is not None:
        body["enable_thinking"] = enable_thinking
    if seed is not None:
        body["seed"] = seed
    if extra:
        body.update(extra)

    last: Exception | None = None
    for attempt in range(retries):
        t0 = time.time()
        try:
            j = _post(url, headers, body, timeout)
            choice = j["choices"][0]
            msg = choice.get("message", {})
            return {
                "content": msg.get("content") or "",
                "reasoning": msg.get("reasoning_content") or "",
                "usage": j.get("usage", {}) or {},
                "latency": time.time() - t0,
                "finish_reason": choice.get("finish_reason"),
                "model": model,
                "provider": provider,
            }
        except urllib.error.HTTPError as e:
            text = e.read().decode("utf-8", "replace")[:300]
            last = LLMError(f"HTTP {e.code} {provider}/{model}: {text}")
            if e.code in RETRYABLE and attempt < retries - 1:
                time.sleep(min(2 ** attempt, 8)); continue
            raise last
        except Exception as e:  # noqa: BLE001 (network/timeouts/JSON)
            last = LLMError(f"{type(e).__name__} {provider}/{model}: {e}")
            if attempt < retries - 1:
                time.sleep(min(2 ** attempt, 8)); continue
            raise last
    raise last  # pragma: no cover


def chat_batch(jobs: list[dict], max_workers: int = 8) -> list:
    """Run many chat() calls concurrently. Each job is a kwargs dict for chat().

    Returns a list aligned with `jobs`; failed calls become the LLMError object
    (not raised) so the caller can filter/inspect.
    """
    results: list = [None] * len(jobs)

    def run(i: int):
        try:
            return i, chat(**jobs[i])
        except Exception as e:  # noqa: BLE001
            return i, e

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, res in ex.map(run, range(len(jobs))):
            results[i] = res
    return results
