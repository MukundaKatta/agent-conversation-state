"""Simple state machine for agent conversation flow.

Track the lifecycle of an agent conversation from idle through
tool use to completion.  Raises :class:`InvalidTransitionError` on
any attempt to move to an illegal next state.

Example::

    from agent_conversation_state import ConversationMachine, ConversationState

    machine = ConversationMachine()
    machine.start()              # IDLE → PROCESSING
    machine.use_tool()           # PROCESSING → TOOL_USE
    machine.resume()             # TOOL_USE → PROCESSING
    machine.await_user()         # PROCESSING → AWAITING_USER
    machine.user_replied()       # AWAITING_USER → PROCESSING
    machine.complete()           # PROCESSING → COMPLETED

    print(machine.state)                # ConversationState.COMPLETED
    print(len(machine.history))         # 6
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ConversationState(str, Enum):
    """States in an agent conversation lifecycle."""

    IDLE = "idle"
    PROCESSING = "processing"
    TOOL_USE = "tool_use"
    AWAITING_USER = "awaiting_user"
    COMPLETED = "completed"
    FAILED = "failed"


# Terminal states — no transitions out
_TERMINAL_STATES = {ConversationState.COMPLETED, ConversationState.FAILED}

# Valid transitions: from_state → set of allowed to_states
_VALID_TRANSITIONS: dict[ConversationState, set[ConversationState]] = {
    ConversationState.IDLE: {
        ConversationState.PROCESSING,
        ConversationState.FAILED,
    },
    ConversationState.PROCESSING: {
        ConversationState.TOOL_USE,
        ConversationState.AWAITING_USER,
        ConversationState.COMPLETED,
        ConversationState.FAILED,
    },
    ConversationState.TOOL_USE: {
        ConversationState.PROCESSING,
        ConversationState.FAILED,
    },
    ConversationState.AWAITING_USER: {
        ConversationState.PROCESSING,
        ConversationState.FAILED,
    },
    ConversationState.COMPLETED: set(),
    ConversationState.FAILED: set(),
}


class InvalidTransitionError(RuntimeError):
    """Raised when a state transition is not allowed.

    Attributes:
        from_state: The state the machine was in.
        to_state:   The state that was requested.
    """

    def __init__(
        self, from_state: ConversationState, to_state: ConversationState
    ) -> None:
        self.from_state = from_state
        self.to_state = to_state
        allowed = sorted(s.value for s in _VALID_TRANSITIONS.get(from_state, set()))
        super().__init__(
            f"Cannot transition from {from_state.value!r} to {to_state.value!r}."
            f" Allowed: {allowed}"
        )


@dataclass
class StateTransition:
    """A recorded state change.

    Attributes:
        from_state: State before the transition.
        to_state:   State after the transition.
        event:      Optional label for the event that triggered the change.
        timestamp:  Unix timestamp when the transition occurred.
        metadata:   Optional free-form dict attached to the transition.
    """

    from_state: ConversationState
    to_state: ConversationState
    event: str
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict."""
        return {
            "from_state": self.from_state.value,
            "to_state": self.to_state.value,
            "event": self.event,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        return (
            f"StateTransition({self.from_state.value!r}"
            f" → {self.to_state.value!r}, event={self.event!r})"
        )


class ConversationMachine:
    """State machine for a single agent conversation.

    Args:
        clock:        Optional callable returning current Unix time.  Defaults
                      to :func:`time.time`.
        allow_custom: If ``True``, :meth:`transition` will accept any
                      :class:`ConversationState` target without validation.
                      Default ``False`` — raises :class:`InvalidTransitionError`
                      on invalid moves.
    """

    def __init__(
        self,
        *,
        clock: Any = None,
        allow_custom: bool = False,
    ) -> None:
        self._state = ConversationState.IDLE
        self._history: list[StateTransition] = []
        self._clock: Any = clock if clock is not None else time.time
        self._allow_custom = allow_custom

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> ConversationState:
        """Current state."""
        return self._state

    @property
    def history(self) -> list[StateTransition]:
        """All recorded state transitions (oldest first)."""
        return list(self._history)

    @property
    def is_terminal(self) -> bool:
        """``True`` when the machine is in a terminal state."""
        return self._state in _TERMINAL_STATES

    @property
    def can_transition(self) -> bool:
        """``True`` when at least one outgoing transition exists."""
        return bool(_VALID_TRANSITIONS.get(self._state))

    # ------------------------------------------------------------------
    # Low-level transition
    # ------------------------------------------------------------------

    def transition(
        self,
        to: ConversationState,
        *,
        event: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> StateTransition:
        """Move to state *to*.

        Args:
            to:       Target state.
            event:    Optional label for the event.
            metadata: Optional dict attached to the :class:`StateTransition`.

        Raises:
            :class:`InvalidTransitionError` if the transition is not allowed
            and ``allow_custom`` is ``False``.

        Returns:
            The recorded :class:`StateTransition`.
        """
        if not self._allow_custom:
            allowed = _VALID_TRANSITIONS.get(self._state, set())
            if to not in allowed:
                raise InvalidTransitionError(self._state, to)

        tx = StateTransition(
            from_state=self._state,
            to_state=to,
            event=event,
            timestamp=self._clock(),
            metadata=metadata or {},
        )
        self._state = to
        self._history.append(tx)
        return tx

    # ------------------------------------------------------------------
    # Named convenience methods
    # ------------------------------------------------------------------

    def start(self, **meta: Any) -> StateTransition:
        """Transition IDLE → PROCESSING."""
        return self.transition(
            ConversationState.PROCESSING, event="start", metadata=meta
        )

    def use_tool(self, tool_name: str = "", **meta: Any) -> StateTransition:
        """Transition PROCESSING → TOOL_USE."""
        m = {"tool_name": tool_name, **meta} if tool_name else meta
        return self.transition(ConversationState.TOOL_USE, event="use_tool", metadata=m)

    def resume(self, **meta: Any) -> StateTransition:
        """Transition TOOL_USE → PROCESSING."""
        return self.transition(
            ConversationState.PROCESSING, event="resume", metadata=meta
        )

    def await_user(self, **meta: Any) -> StateTransition:
        """Transition PROCESSING → AWAITING_USER."""
        return self.transition(
            ConversationState.AWAITING_USER, event="await_user", metadata=meta
        )

    def user_replied(self, **meta: Any) -> StateTransition:
        """Transition AWAITING_USER → PROCESSING."""
        return self.transition(
            ConversationState.PROCESSING, event="user_replied", metadata=meta
        )

    def complete(self, **meta: Any) -> StateTransition:
        """Transition PROCESSING → COMPLETED."""
        return self.transition(
            ConversationState.COMPLETED, event="complete", metadata=meta
        )

    def fail(self, *, reason: str = "", **meta: Any) -> StateTransition:
        """Transition any non-terminal state → FAILED."""
        m = {"reason": reason, **meta} if reason else meta
        return self.transition(ConversationState.FAILED, event="fail", metadata=m)

    def reset(self) -> None:
        """Return to IDLE and clear history."""
        self._state = ConversationState.IDLE
        self._history.clear()

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    def allowed_transitions(self) -> list[ConversationState]:
        """Return states reachable from the current state."""
        return sorted(
            _VALID_TRANSITIONS.get(self._state, set()),
            key=lambda s: s.value,
        )

    def elapsed_in_state_ms(self) -> float | None:
        """Milliseconds since the last transition (or machine start).

        Returns ``None`` for a fresh machine that has never transitioned.
        """
        if not self._history:
            return None
        last_ts = self._history[-1].timestamp
        return (self._clock() - last_ts) * 1000.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot."""
        return {
            "state": self._state.value,
            "is_terminal": self.is_terminal,
            "history": [t.to_dict() for t in self._history],
        }

    def __repr__(self) -> str:
        n = len(self._history)
        return f"ConversationMachine(state={self._state.value!r}, transitions={n})"
