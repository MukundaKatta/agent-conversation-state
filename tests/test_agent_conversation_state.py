"""Tests for agent-conversation-state."""

from __future__ import annotations

import pytest

from agent_conversation_state import (
    ConversationMachine,
    ConversationState,
    InvalidTransitionError,
    StateTransition,
)

# ---------------------------------------------------------------------------
# ConversationState enum
# ---------------------------------------------------------------------------


def test_state_values():
    assert ConversationState.IDLE.value == "idle"
    assert ConversationState.PROCESSING.value == "processing"
    assert ConversationState.TOOL_USE.value == "tool_use"
    assert ConversationState.AWAITING_USER.value == "awaiting_user"
    assert ConversationState.COMPLETED.value == "completed"
    assert ConversationState.FAILED.value == "failed"


def test_state_is_str():
    assert isinstance(ConversationState.IDLE, str)


# ---------------------------------------------------------------------------
# StateTransition
# ---------------------------------------------------------------------------


def test_state_transition_to_dict():
    tx = StateTransition(
        from_state=ConversationState.IDLE,
        to_state=ConversationState.PROCESSING,
        event="start",
        timestamp=1000.0,
    )
    d = tx.to_dict()
    assert d["from_state"] == "idle"
    assert d["to_state"] == "processing"
    assert d["event"] == "start"
    assert d["timestamp"] == 1000.0


def test_state_transition_repr():
    tx = StateTransition(
        from_state=ConversationState.IDLE,
        to_state=ConversationState.PROCESSING,
        event="start",
        timestamp=1000.0,
    )
    r = repr(tx)
    assert "idle" in r
    assert "processing" in r


def test_state_transition_metadata_default_empty():
    tx = StateTransition(
        from_state=ConversationState.IDLE,
        to_state=ConversationState.PROCESSING,
        event="x",
        timestamp=0.0,
    )
    assert tx.metadata == {}


# ---------------------------------------------------------------------------
# InvalidTransitionError
# ---------------------------------------------------------------------------


def test_invalid_transition_error_attributes():
    err = InvalidTransitionError(ConversationState.IDLE, ConversationState.COMPLETED)
    assert err.from_state == ConversationState.IDLE
    assert err.to_state == ConversationState.COMPLETED
    assert "idle" in str(err)
    assert "completed" in str(err)


# ---------------------------------------------------------------------------
# ConversationMachine — initial state
# ---------------------------------------------------------------------------


def test_initial_state_is_idle():
    m = ConversationMachine()
    assert m.state == ConversationState.IDLE


def test_initial_history_empty():
    m = ConversationMachine()
    assert m.history == []


def test_initial_not_terminal():
    m = ConversationMachine()
    assert not m.is_terminal


def test_initial_can_transition():
    m = ConversationMachine()
    assert m.can_transition


# ---------------------------------------------------------------------------
# ConversationMachine — valid transitions
# ---------------------------------------------------------------------------


def test_start_transitions_to_processing():
    m = ConversationMachine()
    tx = m.start()
    assert m.state == ConversationState.PROCESSING
    assert tx.event == "start"


def test_use_tool_from_processing():
    m = ConversationMachine()
    m.start()
    tx = m.use_tool()
    assert m.state == ConversationState.TOOL_USE
    assert tx.event == "use_tool"


def test_use_tool_with_name():
    m = ConversationMachine()
    m.start()
    tx = m.use_tool(tool_name="web_search")
    assert tx.metadata["tool_name"] == "web_search"


def test_resume_from_tool_use():
    m = ConversationMachine()
    m.start()
    m.use_tool()
    tx = m.resume()
    assert m.state == ConversationState.PROCESSING
    assert tx.event == "resume"


def test_await_user_from_processing():
    m = ConversationMachine()
    m.start()
    tx = m.await_user()
    assert m.state == ConversationState.AWAITING_USER
    assert tx.event == "await_user"


def test_user_replied_from_awaiting():
    m = ConversationMachine()
    m.start()
    m.await_user()
    tx = m.user_replied()
    assert m.state == ConversationState.PROCESSING
    assert tx.event == "user_replied"


def test_complete_from_processing():
    m = ConversationMachine()
    m.start()
    tx = m.complete()
    assert m.state == ConversationState.COMPLETED
    assert tx.event == "complete"
    assert m.is_terminal


def test_fail_from_processing():
    m = ConversationMachine()
    m.start()
    tx = m.fail(reason="timeout")
    assert m.state == ConversationState.FAILED
    assert m.is_terminal
    assert tx.metadata["reason"] == "timeout"


def test_fail_from_tool_use():
    m = ConversationMachine()
    m.start()
    m.use_tool()
    m.fail()
    assert m.state == ConversationState.FAILED


def test_fail_from_awaiting_user():
    m = ConversationMachine()
    m.start()
    m.await_user()
    m.fail()
    assert m.state == ConversationState.FAILED


# ---------------------------------------------------------------------------
# ConversationMachine — invalid transitions
# ---------------------------------------------------------------------------


def test_cannot_transition_from_idle_to_completed():
    m = ConversationMachine()
    with pytest.raises(InvalidTransitionError):
        m.complete()


def test_cannot_transition_from_completed():
    m = ConversationMachine()
    m.start()
    m.complete()
    with pytest.raises(InvalidTransitionError):
        m.transition(ConversationState.PROCESSING)


def test_cannot_transition_from_failed():
    m = ConversationMachine()
    m.start()
    m.fail()
    with pytest.raises(InvalidTransitionError):
        m.start()


def test_cannot_use_tool_from_idle():
    m = ConversationMachine()
    with pytest.raises(InvalidTransitionError):
        m.use_tool()


def test_cannot_complete_from_tool_use():
    m = ConversationMachine()
    m.start()
    m.use_tool()
    with pytest.raises(InvalidTransitionError):
        m.complete()


# ---------------------------------------------------------------------------
# ConversationMachine — history
# ---------------------------------------------------------------------------


def test_history_records_all_transitions():
    m = ConversationMachine()
    m.start()
    m.use_tool()
    m.resume()
    m.complete()
    assert len(m.history) == 4


def test_history_returns_copy():
    m = ConversationMachine()
    h1 = m.history
    m.start()
    h2 = m.history
    assert len(h1) == 0
    assert len(h2) == 1


def test_history_events():
    m = ConversationMachine()
    m.start()
    m.use_tool(tool_name="calc")
    events = [t.event for t in m.history]
    assert events == ["start", "use_tool"]


# ---------------------------------------------------------------------------
# ConversationMachine — allowed_transitions
# ---------------------------------------------------------------------------


def test_allowed_transitions_from_idle():
    m = ConversationMachine()
    allowed = m.allowed_transitions()
    assert ConversationState.PROCESSING in allowed


def test_allowed_transitions_from_processing():
    m = ConversationMachine()
    m.start()
    allowed = m.allowed_transitions()
    assert ConversationState.TOOL_USE in allowed
    assert ConversationState.COMPLETED in allowed
    assert ConversationState.FAILED in allowed


def test_allowed_transitions_from_terminal():
    m = ConversationMachine()
    m.start()
    m.complete()
    assert m.allowed_transitions() == []
    assert not m.can_transition


# ---------------------------------------------------------------------------
# ConversationMachine — reset
# ---------------------------------------------------------------------------


def test_reset_returns_to_idle():
    m = ConversationMachine()
    m.start()
    m.use_tool()
    m.reset()
    assert m.state == ConversationState.IDLE
    assert m.history == []


def test_reset_allows_restart():
    m = ConversationMachine()
    m.start()
    m.complete()
    m.reset()
    m.start()  # should not raise
    assert m.state == ConversationState.PROCESSING


# ---------------------------------------------------------------------------
# ConversationMachine — metadata and custom clock
# ---------------------------------------------------------------------------


def test_transition_metadata():
    m = ConversationMachine()
    tx = m.transition(
        ConversationState.PROCESSING,
        event="start",
        metadata={"source": "api"},
    )
    assert tx.metadata["source"] == "api"


def test_custom_clock():
    times = [100.0, 101.5]
    idx = 0

    def fake_clock():
        nonlocal idx
        v = times[idx]
        idx += 1
        return v

    m = ConversationMachine(clock=fake_clock)
    m.start()
    # clock is called once in transition() — returns times[0]
    assert m.history[0].timestamp == 100.0


def test_elapsed_in_state_ms_none_before_any_transition():
    m = ConversationMachine()
    assert m.elapsed_in_state_ms() is None


def test_elapsed_in_state_ms_positive():
    times = [0.0, 0.0, 1.0]  # init ignored, start recorded at 0.0, check at 1.0
    idx = 0

    def fake_clock():
        nonlocal idx
        v = times[idx]
        idx = min(idx + 1, len(times) - 1)
        return v

    m = ConversationMachine(clock=fake_clock)
    m.start()
    elapsed = m.elapsed_in_state_ms()
    assert elapsed is not None
    assert elapsed >= 0.0


# ---------------------------------------------------------------------------
# ConversationMachine — to_dict / repr
# ---------------------------------------------------------------------------


def test_to_dict_keys():
    m = ConversationMachine()
    d = m.to_dict()
    assert "state" in d
    assert "is_terminal" in d
    assert "history" in d


def test_to_dict_state_value():
    m = ConversationMachine()
    m.start()
    d = m.to_dict()
    assert d["state"] == "processing"


def test_repr_contains_state():
    m = ConversationMachine()
    r = repr(m)
    assert "idle" in r
    assert "ConversationMachine" in r


# ---------------------------------------------------------------------------
# allow_custom mode
# ---------------------------------------------------------------------------


def test_allow_custom_bypasses_validation():
    m = ConversationMachine(allow_custom=True)
    # Jump straight from IDLE to COMPLETED (normally invalid)
    m.transition(ConversationState.COMPLETED)
    assert m.state == ConversationState.COMPLETED
