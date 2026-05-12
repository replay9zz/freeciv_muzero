"""Simulation helpers for Freeciv MuZero training."""

from .state import BaseProvider, GroundTruth, MapConfig, RandomMapProvider  # noqa: F401
try:  # Optional dependency for full Freeciv state helpers.
    from .state import MultiheadState  # noqa: F401
except Exception:  # pragma: no cover - optional dependency
    MultiheadState = None  # type: ignore
try:  # Optional dependency for LuaRemote helpers.
    from .remote import LuaRemoteClient  # noqa: F401
except Exception:  # pragma: no cover - optional dependency
    LuaRemoteClient = None  # type: ignore
