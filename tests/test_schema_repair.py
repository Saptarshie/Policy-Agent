import unittest

from app.schema_repair import (
    extract_tool_calls_from_text,
    make_tool_error_feedback,
    repair_json,
    validate_arguments,
)


class TestRepairJson(unittest.TestCase):
    def test_valid_json_untouched(self):
        obj, notes = repair_json('{"a": 1}')
        self.assertEqual(obj, {"a": 1})
        self.assertEqual(notes, [])

    def test_code_fences(self):
        obj, notes = repair_json('```json\n{"a": 1}\n```')
        self.assertEqual(obj, {"a": 1})
        self.assertIn("removed markdown code fences", notes)

    def test_single_quotes_and_trailing_comma(self):
        obj, notes = repair_json("{'query': 'remote work', 'top_k': 3,}")
        self.assertEqual(obj, {"query": "remote work", "top_k": 3})
        self.assertIn("converted single quotes to double quotes", notes)
        self.assertIn("removed trailing commas", notes)

    def test_unquoted_keys(self):
        obj, notes = repair_json('{query: "leave policy", top_k: 2}')
        self.assertEqual(obj, {"query": "leave policy", "top_k": 2})
        self.assertIn("quoted unquoted keys", notes)

    def test_python_literals(self):
        obj, notes = repair_json('{"ok": True, "x": None, "y": False}')
        self.assertEqual(obj, {"ok": True, "x": None, "y": False})
        self.assertIn("converted Python literals (True/False/None)", notes)

    def test_prose_around_json(self):
        obj, notes = repair_json('Here is the JSON you asked for:\n{"a": 1}\nHope that helps!')
        self.assertEqual(obj, {"a": 1})
        self.assertIn("extracted JSON from surrounding prose", notes)

    def test_unbalanced_brackets(self):
        obj, notes = repair_json('{"a": {"b": 1}')
        self.assertEqual(obj, {"a": {"b": 1}})
        self.assertIn("balanced unbalanced brackets", notes)

    def test_truncated_tail(self):
        obj, _ = repair_json('{"a": 1, "b": "never fini')
        self.assertEqual(obj, {"a": 1})

    def test_garbage_returns_none(self):
        obj, _ = repair_json("this is definitely not json at all")
        self.assertIsNone(obj)

    def test_empty(self):
        self.assertIsNone(repair_json("")[0])
        self.assertIsNone(repair_json(None)[0])


class TestValidateArguments(unittest.TestCase):
    SCHEMA = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "top_k": {"type": "integer"},
        },
        "required": ["query"],
    }

    def test_happy_path_with_coercion(self):
        cleaned, errors = validate_arguments({"query": "leave", "top_k": "5"}, self.SCHEMA)
        self.assertEqual(errors, [])
        self.assertEqual(cleaned, {"query": "leave", "top_k": 5})

    def test_missing_required(self):
        cleaned, errors = validate_arguments({"top_k": 3}, self.SCHEMA)
        self.assertIn("missing required parameter 'query'", errors)

    def test_unknown_param(self):
        cleaned, errors = validate_arguments(
            {"query": "x", "bogus": 1}, self.SCHEMA)
        self.assertTrue(any("unknown parameter 'bogus'" in e for e in errors))

    def test_wrong_type(self):
        cleaned, errors = validate_arguments({"query": "x", "top_k": "abc"}, self.SCHEMA)
        self.assertTrue(any("top_k" in e for e in errors))

    def test_feedback_message(self):
        msg = make_tool_error_feedback("search_policy_docs",
                                       ["missing required parameter 'query'"],
                                       "{top_k: 3}")
        self.assertIn("TOOL_CALL_ERROR", msg)
        self.assertIn("missing required parameter 'query'", msg)
        self.assertIn("Do NOT invent", msg)


class TestExtractToolCallsFromText(unittest.TestCase):
    NAMES = ["search_policy_docs", "read_policy_chunk"]

    def test_tool_call_tag(self):
        text = 'Let me search.\n<tool_call>{"name": "search_policy_docs", "arguments": {"query": "remote work"}}</tool_call>'
        calls = extract_tool_calls_from_text(text, self.NAMES)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "search_policy_docs")
        self.assertEqual(calls[0]["arguments"]["query"], "remote work")

    def test_fenced_json(self):
        text = '```json\n{"tool": "read_policy_chunk", "arguments": {"chunk_id": 12}}\n```'
        calls = extract_tool_calls_from_text(text, self.NAMES)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "read_policy_chunk")
        self.assertEqual(calls[0]["arguments"]["chunk_id"], 12)

    def test_malformed_arguments_repaired(self):
        text = "<tool_call>{'name': 'search_policy_docs', 'arguments': {'query': 'leave',}}</tool_call>"
        calls = extract_tool_calls_from_text(text, self.NAMES)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["arguments"], {"query": "leave"})

    def test_no_tool_call(self):
        self.assertEqual(extract_tool_calls_from_text(
            "The remote work policy allows 3 days.", self.NAMES), [])

    def test_unknown_tool_ignored(self):
        text = '<tool_call>{"name": "delete_everything", "arguments": {}}</tool_call>'
        self.assertEqual(extract_tool_calls_from_text(text, self.NAMES), [])


if __name__ == "__main__":
    unittest.main()
