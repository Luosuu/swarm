"""Token usage and dollar cost.

Endpoints with prefix caching (Tinker, OpenAI, vLLM with prefix caching...)
bill repeated conversation prefixes at a discount. An agent resends its whole
history on every call, so most of its input is cached and the raw token count
overstates cost several times over. Track the three kinds separately.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Usage:
    prompt: int = 0  # all input tokens, cached included
    cached: int = 0  # input tokens served from the prompt cache
    completion: int = 0

    @property
    def uncached(self) -> int:
        return max(self.prompt - self.cached, 0)

    @property
    def total(self) -> int:
        return self.prompt + self.completion

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(self.prompt + other.prompt, self.cached + other.cached, self.completion + other.completion)

    def as_dict(self) -> dict[str, int]:
        return {"prompt": self.prompt, "cached": self.cached, "uncached": self.uncached, "completion": self.completion}


@dataclass(frozen=True)
class Pricing:
    """USD per million tokens."""

    input: float
    cached_input: float
    output: float

    def cost(self, u: Usage) -> float:
        return (u.uncached * self.input + u.cached * self.cached_input + u.completion * self.output) / 1_000_000


# From https://tinker-docs.thinkingmachines.ai/tinker/models/models_and_pricing/ (checked 2026-11-09).
# Prices change: pass --price-input/--price-cached/--price-output to override.
KNOWN_PRICES: dict[str, Pricing] = {
    "thinkingmachines/Inkling-2-Small": Pricing(input=0.30, cached_input=0.03, output=1.00),
    "thinkingmachines/Inkling-2-Small:peft:262144": Pricing(input=0.30, cached_input=0.03, output=1.00),
    "thinkingmachines/Inkling-Small": Pricing(input=0.58, cached_input=0.116, output=1.44),
}


def lookup(model: str) -> Pricing | None:
    return KNOWN_PRICES.get(model)
