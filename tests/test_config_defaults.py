"""The shipped defaults, checked against the code that actually ships them.

`Config` has dataclass defaults AND a from_env that re-specifies each one; the second wins at
runtime. Flipping only the dataclass (prompt_agent / prompt_intent) left the app running the v1
prompts for a day while every test and doc said v3 - the live UI kept refusing follow-ups and
nothing in the suite noticed.
"""

import dataclasses
import os
import unittest
from unittest import mock

from app import config as config_module
from app.config import Config, get_config


def _fresh_config():
    """get_config() caches the process-wide Config; clear it so env changes are seen."""
    config_module._CACHE = None
    try:
        return get_config()
    finally:
        config_module._CACHE = None


class TestShippedDefaults(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(os.environ, {}, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(setattr, config_module, "_CACHE", None)
        for key in ("PROMPT_AGENT", "PROMPT_INTENT", "PROMPT_JUDGE"):
            os.environ.pop(key, None)

    def test_the_prompts_that_actually_run_are_the_ones_documented(self):
        cfg = _fresh_config()
        self.assertEqual(cfg.prompt_agent, "v3", "v3 carries the conditions rule")
        self.assertEqual(cfg.prompt_intent, "v4", "v4 resolves follow-ups with the conversation")
        self.assertEqual(cfg.prompt_judge, "v1")

    def test_env_loader_and_dataclass_do_not_disagree(self):
        """Every field the dataclass defaults must be defaulted the same way from env.

        `rag_top_k` has no dataclass default (it is required), so it is checked separately
        against the value the docs publish instead.
        """
        defaults = {f.name: f.default for f in dataclasses.fields(Config)
                    if f.default is not dataclasses.MISSING}
        cfg = _fresh_config()
        for field in ("prompt_agent", "prompt_intent", "prompt_judge",
                      "hybrid_vector_weight", "hybrid_keyword_weight"):
            with self.subTest(field=field):
                self.assertEqual(getattr(cfg, field), defaults[field],
                                 f"{field}: from_env() and the dataclass have drifted")
        self.assertEqual(cfg.rag_top_k, 12, "README publishes 12")

    def test_explicit_env_still_wins(self):
        os.environ["PROMPT_INTENT"] = "v1"
        self.assertEqual(_fresh_config().prompt_intent, "v1")


if __name__ == "__main__":
    unittest.main()
