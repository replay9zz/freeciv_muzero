from __future__ import annotations

import socket
from collections.abc import Mapping
from dataclasses import dataclass

PlayerGameStatus = tuple[float | None, bool | None, bool | None, str]


@dataclass(frozen=True)
class LiveTerminalStatus:
    terminal: bool
    winner_ids: tuple[int, ...]
    reason: str | None


def detect_live_terminal(
    statuses: Mapping[int, PlayerGameStatus],
    client_state: str | None = None,
) -> LiveTerminalStatus:
    """Detect only terminal states proven by a complete live snapshot."""
    winners = tuple(
        sorted(
            pid
            for pid, (_, winner, _, _) in statuses.items()
            if winner is True
        )
    )
    if winners:
        return LiveTerminalStatus(True, winners, "winner")

    normalized_state = (client_state or "").strip().lower()
    if normalized_state == "over":
        return LiveTerminalStatus(True, (), "draw")
    if normalized_state == "disconnected":
        return LiveTerminalStatus(True, (), "disconnect")

    if not statuses:
        return LiveTerminalStatus(False, (), None)

    alive_flags = [alive for _, _, alive, _ in statuses.values()]
    if alive_flags and all(alive is False for alive in alive_flags):
        return LiveTerminalStatus(True, (), "draw")

    return LiveTerminalStatus(False, (), None)


def terminal_status_for_exception(exc: BaseException) -> LiveTerminalStatus:
    """Classify transport failures without interpreting malformed game data."""
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return LiveTerminalStatus(True, (), "timeout")
    if isinstance(exc, (ConnectionError, OSError)):
        return LiveTerminalStatus(True, (), "disconnect")
    return LiveTerminalStatus(True, (), "error")
