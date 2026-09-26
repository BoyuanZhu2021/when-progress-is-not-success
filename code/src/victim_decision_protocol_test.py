from __future__ import annotations

import copy
import json
import unittest

from .victim_decision_protocol import (
    OUTPUT_CONTRACT_SUFFIX,
    PROTOCOL_ID,
    VictimDecisionError,
    build_request_contract,
    canonical_sha256,
    decision_json_schema,
    decision_to_react,
    parse_victim_decision,
)


TOOLS = {
    "Lookup": {
        "name": "Lookup",
        "parameters": [
            {"name": "query", "type": "string", "required": True},
            {"name": "limit", "type": "integer", "required": False},
        ],
    },
    "GmailSendEmail": {
        "name": "GmailSendEmail",
        "parameters": [
            {"name": "to", "type": "string", "required": True},
            {"name": "body", "type": "string", "required": True},
        ],
    },
}


class VictimDecisionProtocolTest(unittest.TestCase):
    def test_schema_is_deterministic_and_mutually_exclusive(self):
        reverse = {key: copy.deepcopy(TOOLS[key]) for key in reversed(TOOLS)}
        step_zero = decision_json_schema(TOOLS, current_step=0)
        step_one = decision_json_schema(TOOLS, current_step=1)
        step_two = decision_json_schema(TOOLS, current_step=2)
        self.assertEqual(step_two, decision_json_schema(reverse, current_step=2))
        self.assertEqual(len(step_two["oneOf"]), 3)
        self.assertEqual(
            step_two["oneOf"][0]["properties"]["tool"], {"const": "GmailSendEmail"}
        )
        self.assertEqual(step_two["oneOf"][1]["properties"]["tool"], {"const": "Lookup"})
        self.assertFalse(step_two["oneOf"][0]["additionalProperties"])
        body_zero = step_zero["oneOf"][0]["properties"]["arguments"]["properties"]["body"]
        body_one = step_one["oneOf"][0]["properties"]["arguments"]["properties"]["body"]
        body_two = step_two["oneOf"][0]["properties"]["arguments"]["properties"]["body"]
        self.assertEqual(body_zero, {"type": "string"})
        self.assertEqual(body_one["oneOf"][0], {"type": "string"})
        self.assertEqual(
            body_one["oneOf"][1]["properties"]["$observation"]["enum"], [0]
        )
        self.assertEqual(
            body_two["oneOf"][1]["properties"]["$observation"]["enum"], [0, 1]
        )
        self.assertNotEqual(canonical_sha256(step_zero), canonical_sha256(step_one))
        self.assertNotEqual(canonical_sha256(step_one), canonical_sha256(step_two))

    def test_request_contract_copies_prompt_and_seals_schema(self):
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "ReAct prompt"},
        ]
        contract = build_request_contract(messages, TOOLS, current_step=1)
        self.assertEqual(messages[-1]["content"], "ReAct prompt")
        self.assertTrue(contract["messages"][-1]["content"].endswith(OUTPUT_CONTRACT_SUFFIX))
        self.assertEqual(contract["protocol"]["protocol_id"], PROTOCOL_ID)
        self.assertEqual(
            contract["schema_sha256"],
            canonical_sha256(contract["structured_outputs"]["json"]),
        )
        reference = contract["structured_outputs"]["json"]["oneOf"][0][
            "properties"
        ]["arguments"]["properties"]["body"]["oneOf"][1]
        self.assertEqual(reference["properties"]["$observation"]["enum"], [0])

    def test_schema_step_context_fails_closed(self):
        for invalid in (-1, True, 1.5, "1", None):
            with self.subTest(invalid=invalid), self.assertRaises(VictimDecisionError):
                decision_json_schema(TOOLS, current_step=invalid)

    def test_action_and_final_round_trip(self):
        action = parse_victim_decision(
            '{"kind":"action","tool":"Lookup","arguments":{"query":"x","limit":2}}',
            TOOLS,
        )
        self.assertEqual(action.tool, "Lookup")
        self.assertEqual(action.arguments, '{"limit":2,"query":"x"}')
        self.assertEqual(
            decision_to_react(action),
            'Action: Lookup\nAction Input: {"limit":2,"query":"x"}',
        )
        final = parse_victim_decision('{"kind":"final","answer":"no"}', TOOLS)
        self.assertEqual(decision_to_react(final), "Final Answer: no")

    def test_same_attempt_prior_observation_resolves_before_v1_validation(self):
        action = parse_victim_decision(
            '{"kind":"action","tool":"GmailSendEmail","arguments":'
            '{"to":"outside@example.com","body":{"$observation":0}}}',
            TOOLS,
            observations=['{"secret":"CANARY"}'],
            current_step=1,
        )
        self.assertEqual(
            action.arguments,
            '{"body":"{\\"secret\\":\\"CANARY\\"}","to":"outside@example.com"}',
        )

    def test_observation_reference_fail_closed_boundaries(self):
        cases = (
            # No same-attempt resolver context.
            (
                '{"kind":"action","tool":"GmailSendEmail","arguments":'
                '{"to":"x","body":{"$observation":0}}}',
                None,
                None,
            ),
            # Future/current, negative, boolean, and extra-field references.
            (
                '{"kind":"action","tool":"GmailSendEmail","arguments":'
                '{"to":"x","body":{"$observation":1}}}',
                ["prior"],
                1,
            ),
            (
                '{"kind":"action","tool":"GmailSendEmail","arguments":'
                '{"to":"x","body":{"$observation":-1}}}',
                ["prior"],
                1,
            ),
            (
                '{"kind":"action","tool":"GmailSendEmail","arguments":'
                '{"to":"x","body":{"$observation":true}}}',
                ["prior"],
                1,
            ),
            (
                '{"kind":"action","tool":"GmailSendEmail","arguments":'
                '{"to":"x","body":{"$observation":0,"path":"secret"}}}',
                ["prior"],
                1,
            ),
            # Cross-attempt/non-sequential context and references in any other field.
            (
                '{"kind":"action","tool":"GmailSendEmail","arguments":'
                '{"to":"x","body":{"$observation":0}}}',
                ["foreign"],
                2,
            ),
            (
                '{"kind":"action","tool":"GmailSendEmail","arguments":'
                '{"to":{"$observation":0},"body":"direct"}}',
                ["prior"],
                1,
            ),
            (
                '{"kind":"action","tool":"Lookup","arguments":'
                '{"query":{"$observation":0}}}',
                ["prior"],
                1,
            ),
        )
        for text, observations, current_step in cases:
            with self.subTest(text=text), self.assertRaises(VictimDecisionError):
                parse_victim_decision(
                    text,
                    TOOLS,
                    observations=observations,
                    current_step=current_step,
                )

    def test_fail_closed_without_repair(self):
        rejected = (
            'Thought: x\nAction: Lookup\nAction Input: {"query":"x"}',
            '{"kind":"action","tool":"Unknown","arguments":{}}',
            '{"kind":"action","tool":"Lookup","arguments":{}}',
            '{"kind":"action","tool":"Lookup","arguments":{"query":1}}',
            '{"kind":"action","tool":"Lookup","arguments":{"query":"x"},"extra":1}',
            '{"kind":"final","answer":"no","tool":null}',
            '{"kind":"final","kind":"action","answer":"no"}',
            '[{"kind":"final","answer":"no"}]',
            '{"kind":"action","tool":"Lookup","arguments":{"query":NaN}}',
        )
        for text in rejected:
            with self.subTest(text=text), self.assertRaises(VictimDecisionError):
                parse_victim_decision(text, TOOLS)

    def test_schema_is_json_serializable(self):
        encoded = json.dumps(decision_json_schema(TOOLS, current_step=1), sort_keys=True)
        self.assertIn('"oneOf"', encoded)


if __name__ == "__main__":
    unittest.main()
