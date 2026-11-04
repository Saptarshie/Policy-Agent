"""Prompt invariants that would otherwise drift silently.

A refusal sentence appears in three places - the constant the code returns, the v1
agent prompt and the v2 agent prompt. If they disagree, the model is told to say one
thing while the code (and the UI's "was this a refusal?" logic) expects another, and
nothing fails loudly. Pin them together instead.
"""
from __future__ import annotations

import re

from app import prompts


def test_refusal_text_is_quoted_verbatim_by_both_agent_prompts():
    for name in ("AGENT_SYSTEM", "AGENT_SYSTEM_V2"):
        prompt = getattr(prompts, name)
        assert prompts.REFUSAL_TEXT in prompt, f"{name} does not quote REFUSAL_TEXT"


def test_refusal_points_the_reader_somewhere():
    # the assignment's expectation: a refusal is not a dead end, it names who to ask
    text = prompts.REFUSAL_TEXT.lower()
    assert "document owner" in text
    assert "hr" in text


def test_refusal_is_not_a_phrase_list():
    """The refusal must be decided by the model reading the passages, not by a word list.

    A hardcoded list of "not covered" phrases used to classify refusals; it is gone.
    Any reappearance in the answer path is a regression of the same class.
    """
    for module in ("app.admin", "app.agent.react", "app.agent.answer_shape"):
        source = __import__(module, fromlist=["x"]).__file__
        with open(source, encoding="utf-8") as fh:
            body = fh.read()
        assert "REFUSAL_PHRASES" not in body, f"{module} reintroduced a phrase list"


def test_prompts_do_not_contain_fenced_tool_calls():
    """Prompts describe the protocol in prose; they must not demo a fenced JSON call.

    A fenced example inside the system prompt invites the model to emit a fenced
    block instead of a native tool call, which the endpoint then rejects.
    """
    for name in ("AGENT_SYSTEM", "AGENT_SYSTEM_V2"):
        prompt = getattr(prompts, name)
        assert not re.search(r"```(?:json)?\s*\{\s*\"name\"", prompt), name
