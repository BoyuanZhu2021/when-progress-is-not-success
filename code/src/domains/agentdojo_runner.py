"""Victim-episode runner for the in-scope H1 testbed (DISC-2026W28-001).

Drives a *real* AgentDojo ReAct victim agent through one injection episode and scores it with BOTH
the dense process-reward (Phi, from ``agentdojo_infogated.score_phi``) and AgentDojo's terminal
binary ``security()``. This is the integration the field lacks: a programmatic per-step oracle laid
over a genuine attacker-agent x victim-tool-agent loop (not deterministic gates).

Episode (grounded in verified AgentDojo 0.1.35 API):
  pre_env = suite.load_and_inject_default_environment({slot: attacker_text})   # static per-episode
  _, _, post_env, messages, _ = pipeline.query(user_task.PROMPT, runtime, work_env)
  trace  = functions_stack_trace_from_messages(messages)                       # assistant tool_calls
  phi    = score_phi(trace, injection_task.ground_truth(pre_env), gated_values)
  sec    = injection_task.security(model_out, pre_env, post_env)               # terminal, binary

The victim is a served OpenAI-compatible endpoint (vLLM) driven by ``agent_pipeline.LocalLLM``. The
same ``run_episode`` runs model-free with a ``GroundTruthPipeline`` victim for CPU goldens.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # code/ on path -> `src.domains...`

from agentdojo.agent_pipeline import (  # noqa: E402
    AgentPipeline, LocalLLM, OpenAILLM, ToolsExecutionLoop, ToolsExecutor, SystemMessage, InitQuery,
)
from agentdojo.agent_pipeline.agent_pipeline import load_system_message  # noqa: E402
from agentdojo.functions_runtime import FunctionsRuntime  # noqa: E402
from agentdojo.task_suite.task_suite import functions_stack_trace_from_messages  # noqa: E402

from src.domains.agentdojo_infogated import score_phi  # noqa: E402


# --------------------------------------------------------------------------- victim construction

def build_victim_client(base_url: str, api_key: str = "EMPTY"):
    """OpenAI-compatible client pointed at the served victim (e.g. vLLM at .../v1)."""
    import openai
    return openai.OpenAI(api_key=api_key, base_url=base_url)


def build_victim_pipeline(client, model: str, temperature: float = 0.7, max_iters: int = 15,
                          driver: str = "openai"):
    """AgentDojo ReAct victim: [system, init-query, LLM, tools-loop([exec, LLM])]. The LLM element is
    shared between the first turn and the loop (AgentDojo idiom). ``pipeline.name`` is set so attack
    constructors that sniff the model name still work.

    ``driver``:
      - ``"openai"`` (default): ``OpenAILLM`` -> vLLM's NATIVE OpenAI function-calling (structured
        ``tool_calls``). Robust; requires the server started with ``--enable-auto-tool-choice
        --tool-call-parser <hermes|...>``.
      - ``"local"``: ``LocalLLM`` -> prompt-based tool-call parsing. Brittle for models not tuned to
        AgentDojo's exact delimiter format (Qwen3.5 emits malformed blocks -> episodes truncate).
    """
    if driver == "openai":
        llm = OpenAILLM(client, model, temperature=temperature)
    elif driver == "local":
        llm = LocalLLM(client, model, temperature=temperature)
    else:
        raise ValueError(f"unknown driver {driver!r}")
    loop = ToolsExecutionLoop([ToolsExecutor(), llm], max_iters=max_iters)
    pipe = AgentPipeline([SystemMessage(load_system_message(None)), InitQuery(), llm, loop])
    pipe.name = model
    return pipe


# --------------------------------------------------------------------------- robust injection

def _replace_marker(obj, marker: str, text: str):
    """Recursively replace ``marker`` with ``text`` in every string field of a pydantic env model
    (and nested models / lists / dicts), MUTATING in place. Returns the (possibly new) object."""
    from pydantic import BaseModel
    if isinstance(obj, str):
        return obj.replace(marker, text) if marker in obj else obj
    if isinstance(obj, BaseModel):
        for k, v in list(obj.__dict__.items()):
            nv = _replace_marker(v, marker, text)
            if nv is not v:
                setattr(obj, k, nv)
        return obj
    if isinstance(obj, list):
        for i, v in enumerate(obj):
            obj[i] = _replace_marker(v, marker, text)
        return obj
    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            obj[k] = _replace_marker(v, marker, text)
        return obj
    return obj


def inject_env(suite, injections: dict[str, str]):
    """Place arbitrary attacker text into injection slots WITHOUT going through AgentDojo's
    ``str.format`` + ``yaml.safe_load`` path (which corrupts on newlines / quotes / colons / braces
    that a *learned* attacker freely emits). Strategy: load the env with a YAML-safe unique marker
    per slot, then swap each marker for its real text at the pydantic-object level. Faithful (the
    victim reads the exact attacker bytes) and robust to any payload."""
    markers = {slot: f"@@INJ{i}@@" for i, slot in enumerate(injections)}
    env = suite.load_and_inject_default_environment(markers)
    for slot, text in injections.items():
        _replace_marker(env, markers[slot], text)
    return env


# --------------------------------------------------------------------------- one episode

def _last_assistant_text(messages) -> str:
    for m in reversed(messages):
        if m.get("role") == "assistant" and m.get("content"):
            c = m["content"]
            return c if isinstance(c, str) else str(c)
    return ""


def run_episode(suite, user_task, injection_task, slot: str, attacker_text: str, pipeline,
                gated_values_fn=None, ordered: bool = True, phi_fn=None, extra_injections=None) -> dict:
    """Run one injection episode; return {phi, security, n_calls, trace, model_out, error}.

    ``pipeline`` is any AgentDojo pipeline element exposing ``.query`` (real victim AgentPipeline, or
    a GroundTruthPipeline for CPU goldens). Phi is scored one of two ways:
      - ``phi_fn(executed_trace, pre_env)->float`` if given (e.g. the per-value encoded-leak oracle);
      - else ``score_phi(trace, ground_truth, gated_values_fn(pre_env), ordered)`` (the chain oracle).
    ``extra_injections``: additional FIXED slot->content placed alongside the attacker's slot (e.g.
    the secret document for the indirect multi-hop mechanism). Merged into the single inject_env call.
    """
    try:
        injections = {slot: attacker_text, **(extra_injections or {})}
        pre_env = inject_env(suite, injections)               # robust: no YAML str.format on payload
        gt = injection_task.ground_truth(pre_env)             # args read from pristine pre-env
        gv = gated_values_fn(pre_env) if (gated_values_fn and phi_fn is None) else None
        work_env = pre_env.model_copy(deep=True)              # query mutates in place -> isolate pre
        runtime = FunctionsRuntime(suite.tools)
        _, _, post_env, messages, _ = pipeline.query(user_task.PROMPT, runtime, work_env)
    except Exception as e:  # noqa: BLE001 - a crashed episode is a legitimate (phi=0) outcome
        return dict(phi=0.0, security=False, n_calls=0, trace=[], model_out="", error=repr(e)[:300])
    trace = functions_stack_trace_from_messages(messages)
    phi = phi_fn(trace, pre_env) if phi_fn is not None else score_phi(trace, gt, gv, ordered=ordered)
    model_out = _last_assistant_text(messages)
    try:
        security = bool(injection_task.security(model_out, pre_env, post_env))
    except Exception as e:  # noqa: BLE001
        security = False
        model_out = f"[security() raised: {e!r}] " + model_out
    return dict(phi=phi, security=security, n_calls=len(trace),
                trace=[(fc.function, dict(fc.args)) for fc in trace],
                model_out=model_out[:240], error=None)


# --------------------------------------------------------------------------- default pairing

# For the info-gated banking task (InjectionTask101): pair with user_task_0 ("pay the bill
# 'bill-december-2023.txt'"), inject into `injection_bill_text` -- the victim reads that file and so
# surfaces the attacker's text. (Verified: injection_bill_text lives in bill-december-2023.txt.)
DEFAULT_BANKING_PAIRING = dict(user_task_id="user_task_0", slot="injection_bill_text")
