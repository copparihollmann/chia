import json
from io import StringIO

from chia.models.claude import ClaudeCodeLLM


def test_delegated_result_accounts_for_both_models_and_cache_classes():
    llm = ClaudeCodeLLM(model="sonnet")
    event = {
        "type": "result", "subtype": "success",
        "usage": {"input_tokens": 10, "output_tokens": 2},
        "modelUsage": {
            "sonnet": {"inputTokens": 10, "outputTokens": 2,
                       "cacheReadInputTokens": 40, "cacheCreationInputTokens": 8},
            "fixture/weak": {"inputTokens": 7, "outputTokens": 3,
                             "cacheReadInputTokens": 11,
                             "cacheCreationInputTokens": 4},
        },
    }
    llm._process_event_line(json.dumps(event), StringIO(), [])
    meta = llm._last_metadata
    assert (meta["input_tokens"], meta["output_tokens"]) == (17, 5)
    assert (meta["cache_read_input_tokens"], meta["cache_creation_input_tokens"]) == (51, 12)
    assert meta["model_usage"]["fixture/weak"]["input_tokens"] == 7
