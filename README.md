# agent-conversation-state

Simple state machine for agent conversation flow. Zero dependencies.

## Install

```bash
pip install agent-conversation-state
```

## Quick start

```python
from agent_conversation_state import ConversationMachine

machine = ConversationMachine()
machine.start()         # IDLE → PROCESSING
machine.use_tool("web_search")  # PROCESSING → TOOL_USE
machine.resume()        # TOOL_USE → PROCESSING
machine.await_user()    # PROCESSING → AWAITING_USER
machine.user_replied()  # AWAITING_USER → PROCESSING
machine.complete()      # PROCESSING → COMPLETED

print(machine.state)         # ConversationState.COMPLETED
print(machine.is_terminal)   # True
print(len(machine.history))  # 6
```

## States

```
IDLE → PROCESSING → TOOL_USE → PROCESSING
              ↓                      ↑
        AWAITING_USER ─────────────→
              ↓
         COMPLETED / FAILED
```

| State | Description |
|---|---|
| `idle` | Initial state |
| `processing` | LLM is generating a response |
| `tool_use` | A tool call is in flight |
| `awaiting_user` | Waiting for user input |
| `completed` | Run finished successfully |
| `failed` | Run terminated with an error |

## API

### `ConversationMachine`

| Method | Transition | Description |
|---|---|---|
| `start(**meta)` | IDLE → PROCESSING | Begin processing |
| `use_tool(*, tool_name, **meta)` | PROCESSING → TOOL_USE | Start a tool call |
| `resume(**meta)` | TOOL_USE → PROCESSING | Tool call finished |
| `await_user(**meta)` | PROCESSING → AWAITING_USER | Need user input |
| `user_replied(**meta)` | AWAITING_USER → PROCESSING | User responded |
| `complete(**meta)` | PROCESSING → COMPLETED | Run finished |
| `fail(*, reason, **meta)` | any → FAILED | Mark as failed |
| `transition(to, *, event, metadata)` | explicit | Low-level transition |
| `reset()` | → IDLE | Clear history, return to idle |
| `allowed_transitions()` | — | States reachable from current |

Properties: `state`, `history`, `is_terminal`, `can_transition`, `elapsed_in_state_ms()`

Invalid transitions raise `InvalidTransitionError` with a message listing allowed targets.

## License

MIT
