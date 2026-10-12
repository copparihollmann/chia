import pytest

from chia.models.opencode import OpenCodeLLM


def test_tool_wait_is_not_counted_as_model_active_time():
    export = {"messages": [{"info": {
        "role": "assistant", "id": "turn-one",
        "time": {"created": 1000, "completed": 3000},
        "tokens": {"input": 10, "output": 2,
                   "cache": {"read": 7, "write": 3}}, "cost": 0.001,
    }, "parts": [{"type": "tool", "tool": "task", "state": {
        "status": "completed", "time": {"start": 2000, "end": 2900},
    }}]}]}
    requests, tools = OpenCodeLLM._export_requests_and_tools(export)
    assert len(requests) == len(tools) == 1
    assert requests[0]["end_ts"] == tools[0]["start_ts"] == 2.0
    assert requests[0]["duration_s"] == pytest.approx(1.0)
    assert tools[0]["duration_s"] == pytest.approx(0.9)
    assert (requests[0]["input_tokens"], requests[0]["output_tokens"]) == (10, 2)
    assert (requests[0]["cache_read_input_tokens"],
            requests[0]["cache_creation_input_tokens"]) == (7, 3)
