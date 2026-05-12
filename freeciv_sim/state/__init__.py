"""Core Freeciv state and environment primitives."""

from .config import MapConfig, TrainingConfig
from .movement import FreecivMovement
from .providers import BaseProvider, FixedMapProvider, GroundTruth, LuaRemoteProvider, RandomMapProvider
try:  # Optional because ruleset assets may be unavailable in lightweight environments.
    from .multihead_state import MultiheadState
except Exception:  # pragma: no cover - optional dependency
    MultiheadState = None  # type: ignore

__all__ = [
    "BaseProvider",
    "FixedMapProvider",
    "FreecivMovement",
    "GroundTruth",
    "LuaRemoteProvider",
    "MapConfig",
    "MultiheadState",
    "RandomMapProvider",
    "TrainingConfig",
]
