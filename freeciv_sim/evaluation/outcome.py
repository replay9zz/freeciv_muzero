from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

PlayerScore = tuple[float | None, bool | None, str]


@dataclass(frozen=True)
class GameOutcome:
    value: float
    win_point: float
    own_score: float | None
    opponent_score: float | None
    decided_by: str


def normalized_score_margin(own_score: float, opponent_score: float) -> float:
    """Return a bounded score margin in [-1, 1]."""
    denominator = abs(float(own_score)) + abs(float(opponent_score)) + 1.0
    return max(-1.0, min(1.0, (float(own_score) - float(opponent_score)) / denominator))


def game_outcome(
    scores: Mapping[int, PlayerScore],
    player_id: int,
) -> GameOutcome:
    """Derive the only environment reward: result, then fixed-horizon score margin."""
    own = scores.get(int(player_id), (None, None, ""))
    opponents = [entry for pid, entry in scores.items() if int(pid) != int(player_id)]
    own_score, own_winner, _own_name = own

    if own_winner is True:
        return GameOutcome(1.0, 1.0, own_score, _strongest_score(opponents), "winner")
    if any(winner is True for _score, winner, _name in opponents):
        return GameOutcome(-1.0, 0.0, own_score, _strongest_score(opponents), "winner")

    opponent_score = _strongest_score(opponents)
    if own_score is None or opponent_score is None:
        return GameOutcome(0.0, 0.5, own_score, opponent_score, "unavailable")

    margin = normalized_score_margin(own_score, opponent_score)
    if margin > 0:
        win_point = 1.0
    elif margin < 0:
        win_point = 0.0
    else:
        win_point = 0.5
    return GameOutcome(margin, win_point, own_score, opponent_score, "score")


def optuna_objective(outcomes: Sequence[GameOutcome]) -> float:
    """Rank by win rate; use score margin only as a bounded tie-break signal."""
    if not outcomes:
        raise ValueError("at least one game outcome is required")
    count = len(outcomes)
    win_rate = sum(item.win_point for item in outcomes) / count
    mean_margin = sum(_score_margin(item) for item in outcomes) / count
    return win_rate + (0.2 / count) * mean_margin


def _strongest_score(entries: Sequence[PlayerScore]) -> float | None:
    values = [float(score) for score, _winner, _name in entries if score is not None]
    return max(values) if values else None


def _score_margin(outcome: GameOutcome) -> float:
    if outcome.own_score is None or outcome.opponent_score is None:
        return 0.0
    return normalized_score_margin(outcome.own_score, outcome.opponent_score)
