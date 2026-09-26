"""Provider resolution: map a logical provider name -> (base_url, api_key).

Secrets are read from the project-root `.env` (or the real process environment,
which takes precedence). Never hard-code keys. `.env` format is `KEY=VALUE`,
`#` comments and blank lines ignored. Values may contain spaces (e.g. REMOTE_HOST).
"""
from __future__ import annotations

import os
from pathlib import Path

# code/src/providers.py -> parents[2] == project root (where .env lives)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"

# logical name -> (base_url_env, api_key_env)
PROVIDERS = {
    "siliconflow":      ("SILICONFLOW_BASE_URL", "SILICONFLOW_API_KEY"),
    "aipaibox_claude":  ("CLAUDE_BASE_URL",      "CLAUDE_API_KEY"),
    "aipaibox_gpt":     ("GPT_BASE_URL",         "GPT_API_KEY"),
    "openai":           ("OPENAI_BASE_URL",      "OPENAI_API_KEY"),
}

_ENV_CACHE: dict | None = None


def load_env(path: Path | str | None = None) -> dict:
    path = Path(path) if path else ENV_PATH
    env: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, v = s.split("=", 1)
            env[k.strip()] = v.strip()
    return env


def _env() -> dict:
    global _ENV_CACHE
    if _ENV_CACHE is None:
        _ENV_CACHE = load_env()
    return _ENV_CACHE


def get(key: str, default: str | None = None) -> str | None:
    """Real environment overrides .env."""
    return os.environ.get(key) or _env().get(key, default)


def resolve_provider(name: str) -> tuple[str, str]:
    if name not in PROVIDERS:
        raise KeyError(f"unknown provider {name!r}; known: {sorted(PROVIDERS)}")
    base_k, key_k = PROVIDERS[name]
    base, key = get(base_k), get(key_k)
    if not base or not key:
        raise RuntimeError(f"missing {base_k}/{key_k} in environment or {ENV_PATH}")
    return base, key


def mask(s: str | None) -> str:
    if not s:
        return "(empty)"
    return (s[:6] + "..." + s[-4:]) if len(s) > 12 else "***"
