"""Simple state machine for agent conversation flow."""

from __future__ import annotations

from .core import (
    ConversationMachine,
    ConversationState,
    InvalidTransitionError,
    StateTransition,
)

__all__ = [
    "ConversationState",
    "StateTransition",
    "InvalidTransitionError",
    "ConversationMachine",
]
