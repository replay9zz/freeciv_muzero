"""Outcome evaluation and optional strategic helpers for role agents."""

from .strategic_value import (
    StrategicBreakdown,
    StrategicValueWeights,
    potential_shaping_reward,
    production_asset_value,
    research_completion_value,
    strategic_potential,
)
from .outcome import GameOutcome, game_outcome, normalized_score_margin, optuna_objective

__all__ = [
    "StrategicBreakdown",
    "StrategicValueWeights",
    "potential_shaping_reward",
    "production_asset_value",
    "research_completion_value",
    "strategic_potential",
    "GameOutcome",
    "game_outcome",
    "normalized_score_margin",
    "optuna_objective",
]
