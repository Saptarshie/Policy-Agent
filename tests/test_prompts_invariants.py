"""Prompt invariants that would otherwise drift silently.

A refusal sentence appears in three places - the constant the code returns, the v1
agent prompt and the v2 agent prompt. If they disagree, the model is told to say one
thing while the code (and the UI's "was this a refusal?" logic) expects another, and
nothing fails loudly. Pin them together instead.
"""
from __future__ import annotations

import re

from app import prompts
import unittest


class TestPromptInvariants(unittest.TestCase):
    def test_refusal_text_is_quoted_verbatim_by_both_agent_prompts(self):
        for name in ("AGENT_SYSTEM", "AGENT_SYSTEM_V2"):
            prompt = getattr(prompts, name)
            assert prompts.REFUSAL_TEXT in prompt, f"{name} does not quote REFUSAL_TEXT"


    def test_refusal_points_the_reader_somewhere(self):
        # the assignment's expectation: a refusal is not a dead end, it names who to ask
        text = prompts.REFUSAL_TEXT.lower()
        assert "document owner" in text
        assert "hr" in text


    def test_refusal_is_not_a_phrase_list(self):
        """The refusal must be decided by the model reading the passages, not by a word list.

        A hardcoded list of "not covered" phrases used to classify refusals; it is gone.
        Any reappearance in the answer path is a regression of the same class.
        """
        for module in ("app.admin", "app.agent.react", "app.agent.answer_shape"):
            source = __import__(module, fromlist=["x"]).__file__
            with open(source, encoding="utf-8") as fh:
                body = fh.read()
            assert "REFUSAL_PHRASES" not in body, f"{module} reintroduced a phrase list"


    def test_prompts_do_not_contain_fenced_tool_calls(self):
        """Prompts describe the protocol in prose; they must not demo a fenced JSON call.

        A fenced example inside the system prompt invites the model to emit a fenced
        block instead of a native tool call, which the endpoint then rejects.
        """
        for name in ("AGENT_SYSTEM", "AGENT_SYSTEM_V2"):
            prompt = getattr(prompts, name)
            assert not re.search(r"```(?:json)?\s*\{\s*\"name\"", prompt), name


    def test_v3_is_v1_plus_the_conditions_rule_and_nothing_else(self):
        """v3 exists to test one idea, so it must not quietly become a rewrite.

        The ablation compares v1 against v3; if v3 also changed the refusal text, rule
        numbering or the tool-format fallback, a difference in the results could not be
        attributed to the conditions rule.
        """
        assert "9. CONDITIONS AND SPECIAL CASES" in prompts.AGENT_SYSTEM_V3
        assert "9. CONDITIONS" not in prompts.AGENT_SYSTEM, "v1 must stay byte-identical"
        # everything else survives, including the rules v1 ships
        for marker in ("1. GROUNDING", "6. CONFLICTS", "8. NUMBERS YOU DERIVE",
                       "## How to work (ReAct loop)", "## Final answer format",
                       "## Tool-call format fallback"):
            assert marker in prompts.AGENT_SYSTEM_V3, marker
        assert prompts.REFUSAL_TEXT in prompts.AGENT_SYSTEM_V3
        # and the rule lands before the loop instructions, not after the answer format
        assert (prompts.AGENT_SYSTEM_V3.index("9. CONDITIONS")
                < prompts.AGENT_SYSTEM_V3.index("## How to work (ReAct loop)"))
        # v3 is a superset, not a replacement
        assert len(prompts.AGENT_SYSTEM_V3) > len(prompts.AGENT_SYSTEM)


    def test_v3_intent_prompt_asks_for_conditions_without_asking_for_conclusions(self):
        text = prompts.INTENT_SYSTEM_V3
        assert "### 3.5 CONDITIONS AND SPECIAL CASES" in text
        assert '"conditions"' in text
        assert "### 3.5" not in prompts.INTENT_SYSTEM, "v1 must stay byte-identical"
        # the instruction must forbid deciding the answer in triage
        assert "do NOT decide the answer here" in text  # triage classifies, it does not answer
        # the schema stays valid JSON with the new field present once
        body = text[text.index("{\n  \"intent\""):]
        body = body[:body.index("\n}") + 2]
        import json
        parsed = json.loads(body)
        assert "conditions" in parsed
