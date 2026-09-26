"""Domain interface: a domain supplies attack goals, builds the target-agent
prompt from an attacker-generated injection, and judges success programmatically.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Goal:
    id: str
    domain: str
    brief: str                       # human-readable objective shown to the attacker
    split: str = "ood"               # 'train' | 'calibration' | 'ood' (legacy: 'indomain')
    meta: dict = field(default_factory=dict)


@dataclass
class OracleResult:
    eval: str                        # 'succ' | 'unsucc' | 'invalid'
    success: bool                    # True iff eval == 'succ'
    detail: dict = field(default_factory=dict)


class Domain(ABC):
    name: str
    # How tool calls are presented to ``score`` across outer attacker turns.
    # ``episode`` accumulates calls across turns; ``attempt`` scores each fresh victim
    # attempt independently and lets the harness retain only the best legal attempt.
    call_scope: str = "episode"

    @abstractmethod
    def load_goals(self, split: str = "ood", seed: int = 0, n: int | None = None) -> list[Goal]:
        ...

    @abstractmethod
    def attacker_system(self) -> str:
        """System prompt establishing the (authorized) red-team attacker role."""

    @abstractmethod
    def attacker_task(self, goal: Goal) -> str:
        """First user turn: what the attacker must achieve + context it can use."""

    def attacker_examples(self) -> list[dict]:
        """Optional few-shot messages inserted between system prompt and the task."""
        return []

    @abstractmethod
    def build_target_messages(self, goal: Goal, injection_text: str) -> list[dict]:
        """Splice the attacker's injection into the target-agent prompt."""

    @abstractmethod
    def oracle(self, goal: Goal, target_output: str, injection_text: str) -> OracleResult:
        ...

    @abstractmethod
    def feedback(self, goal: Goal, result: OracleResult, target_output: str) -> str:
        """Next user turn for the attacker after a failed attempt (multi-turn arm)."""
