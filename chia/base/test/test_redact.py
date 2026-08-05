"""Tests for :mod:`chia.base.redact`.

The interesting cases are all about *not* over-masking. A redactor that masks everything
is trivially safe and useless, so most of what follows pins the boundaries: how short a
value has to be before it is ignored, which shapes are excluded, and that the mask itself
survives being re-redacted.
"""

import logging

import pytest

from chia.base.redact import (
    MASK,
    MIN_SECRET_LEN,
    is_secret_name,
    redact,
    secret_values,
    truncate,
)


TOKEN = "bedrock-abcdef0123456789abcdef"


# --------------------------------------------------------------------------- names
@pytest.mark.parametrize("name", [
    "AWS_BEARER_TOKEN_BEDROCK",
    "ANTHROPIC_API_KEY",
    "AWS_SECRET_ACCESS_KEY",
    "MY_SERVICE_TOKEN",
    "SOME_PASSWORD",
    "app_secret",
    "GCP_CREDENTIALS",
])
def test_credential_shaped_names_are_recognized(name):
    assert is_secret_name(name)


@pytest.mark.parametrize("name", [
    "AWS_REGION",
    "PATH",
    "CHIA_AET_RUN_DIR",
    "ANTHROPIC_MODEL",
    "KEYBOARD_LAYOUT",   # 'KEY' only counts on a word boundary
    "TOKENIZER",
])
def test_ordinary_names_are_not_treated_as_credentials(name):
    assert not is_secret_name(name)


# --------------------------------------------------------------------------- values
def test_a_credential_value_is_masked_and_named():
    env = {"AWS_BEARER_TOKEN_BEDROCK": TOKEN}
    out = redact(f"AccessDenied: token {TOKEN} is expired", env=env)
    assert TOKEN not in out
    assert MASK.format(name="AWS_BEARER_TOKEN_BEDROCK") in out
    # The surrounding text is untouched — a redacted log still has to be readable.
    assert out.startswith("AccessDenied: token ")
    assert out.endswith(" is expired")


def test_a_short_value_is_left_alone():
    """The floor that a literal port of spec's version would not have.

    spec reads a curated .env; chia inherits the caller's whole environment, where a
    credential-shaped name can hold something tiny. Masking it would match everywhere.
    """
    short = "a" * (MIN_SECRET_LEN - 1)
    text = f"cannot open {short}bcdef.log: no such file"
    assert redact(text, env={"MY_KEY": short}) == text


def test_a_path_valued_variable_is_left_alone():
    """A path is not a credential, and masking it would obscure every log line."""
    env = {"AWS_CONFIG_KEY": "/home/someone/.aws/config"}
    text = "reading /home/someone/.aws/config failed"
    assert redact(text, env=env) == text


def test_a_whitespace_valued_variable_is_left_alone():
    env = {"BUILD_SECRET": "make -j8 all"}
    text = "ran make -j8 all"
    assert redact(text, env=env) == text


def test_a_non_credential_variable_is_not_masked_however_long():
    env = {"ANTHROPIC_MODEL": "us.anthropic.claude-sonnet-4-6"}
    text = "model us.anthropic.claude-sonnet-4-6 not found in region"
    assert redact(text, env=env) == text


def test_empty_and_none_pass_through():
    assert redact(None, env={"A_TOKEN": TOKEN}) is None
    assert redact("", env={"A_TOKEN": TOKEN}) == ""


def test_redaction_is_idempotent():
    env = {"A_TOKEN": TOKEN}
    once = redact(f"failed with {TOKEN}", env=env)
    assert redact(once, env=env) == once


def test_an_explicit_secret_not_in_the_environment_is_masked():
    """``api_key`` reaches a backend as a constructor argument, not through the env."""
    out = redact(f"401 for key {TOKEN}", env={}, extra=(TOKEN,))
    assert TOKEN not in out
    assert MASK.format(name="secret") in out


def test_a_secret_containing_another_is_fully_masked():
    """Longest-first ordering: masking the short one first would leave the rest of the
    long one in the text."""
    short = "abcdef0123456789"
    long = short + "TAILTAILTAIL"
    env = {"SHORT_TOKEN": short, "LONG_TOKEN": long}
    out = redact(f"denied {long}", env=env)
    assert short not in out
    assert long not in out


def test_values_are_ordered_longest_first():
    env = {"A_TOKEN": "a" * 10, "B_TOKEN": "b" * 40, "C_TOKEN": "c" * 20}
    lengths = [len(value) for _, value in secret_values(env)]
    assert lengths == sorted(lengths, reverse=True)


def test_the_mask_is_json_safe():
    """The CLI backends redact JSON payloads before parsing them, so the mask must not
    introduce a quote or a backslash."""
    import json
    env = {"A_TOKEN": TOKEN}
    payload = json.dumps({"error": f"bad token {TOKEN}"})
    assert json.loads(redact(payload, env=env))["error"].endswith(
        MASK.format(name="A_TOKEN"))


# --------------------------------------------------------------------------- truncate
def test_truncate_redacts_before_cutting():
    """The bug this function exists to prevent.

    Every ``raw_message=...[:300]`` site in the CLI backends sliced first. A secret
    straddling the boundary then survives as a fragment, and a fragment of a token is
    still a disclosure.
    """
    env = {"A_TOKEN": TOKEN}
    text = "x" * 295 + TOKEN
    out = truncate(text, 300, env=env)
    assert TOKEN[:5] not in out
    assert len(out) <= 300


def test_truncate_of_none_is_empty():
    assert truncate(None, 10) == ""


# --------------------------------------------------------------------------- integration
def test_the_cli_backend_masks_stderr_result_and_log(tmp_path, monkeypatch, caplog):
    """End to end through ``ClaudeCodeLLM``: the token reaches none of the four sinks.

    Faked at ``subprocess.run`` rather than by launching a CLI, so the test is offline
    and the assertion is about chia's handling rather than about any provider's behaviour.
    """
    import subprocess

    from chia.models.claude import ClaudeCodeLLM

    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", TOKEN)

    class _Completed:
        returncode = 1
        stdout = f"partial output mentioning {TOKEN}\n"
        stderr = f"AccessDeniedException: the token {TOKEN} is not authorized\n"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Completed())

    llm = ClaudeCodeLLM(system_message="", log_dir=str(tmp_path), log_stream=False)
    with caplog.at_level(logging.WARNING):
        result = llm._run_claude("hello")

    assert TOKEN not in result.stderr
    assert TOKEN not in result.result
    assert "AWS_BEARER_TOKEN_BEDROCK" in result.stderr   # named, so it is still diagnosable
    assert TOKEN not in caplog.text
    on_disk = "".join(p.read_text() for p in tmp_path.rglob("*.log"))
    assert TOKEN not in on_disk
    assert "partial output mentioning" in on_disk         # the log is still useful


def test_the_typed_error_carries_no_secret(monkeypatch):
    """The 300-character ``raw_message`` slice is the most-copied of the sinks: it travels
    inside a Ray exception to whatever node called ``prompt``."""
    import subprocess

    from chia.models.claude import AuthenticationError, ClaudeCodeLLM

    monkeypatch.setenv("ANTHROPIC_API_KEY", TOKEN)

    class _Completed:
        returncode = 1
        stdout = ""
        stderr = f"unauthorized: 401 for api key {TOKEN}"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Completed())

    llm = ClaudeCodeLLM(system_message="")
    cli = llm._run_claude("hello")
    with pytest.raises(AuthenticationError) as excinfo:
        llm._classify_error(cli)
    assert TOKEN not in excinfo.value.raw_message
    assert TOKEN not in str(excinfo.value)
