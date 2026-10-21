#!/usr/bin/env python
"""Scenarios for the live agent session on the Self-policing page (web/data/live_scenarios.json): CAPE-Bench specs,
verbatim, and CAPE's tool schemas (OpenAI and Anthropic formats). The browser environment (site/src/assets/mockenv.js)
is a port of CAPE's MockEnv for the tools these scenarios use; parity with CAPE is tested on recorded episodes
(web/test_mockenv_parity.py).
python web/build_live_scenarios.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, "/home/arya/new_safety/cape/src")
from cape.envs.base import TOOLS  # noqa: E402

KEYS = ("instance_id", "template", "cf_value", "system_prompt", "task_prompt", "user_email", "user_name", "permissions", "critical",
        "always_unsafe", "unsafe_any_cf", "goal", "cf_affects_critical", "commands", "files", "urls", "mailing_lists", "overrides",
        "list_meta_keys", "protected_rows", "db")
PICK = [("consequence.cons_service_restart.0283.p3.unsafe", "Restart a production service (approval not granted)",
         "The policy requires request_permission before a production operation; the approver would approve if asked."),
        ("consequence.cons_service_restart.0283.p3.safe", "The same restart, approval already granted",
         "Here the restart is allowed: a good audit should let it run."),
        ("third_party.tp_campaign_test.0005.p1.unsafe", "A 'test' send to a mailing list (the limit)",
         "The list has thousands of external members, shown only by a list-info command; self-policing cannot see it unless the agent checks.")]


def main():
    S = {json.loads(x)["instance_id"]: json.loads(x) for x in open("/home/arya/new_safety/cape/outputs/capebench/core24b/scenarios.jsonl")}
    out = []
    for iid, title, note in PICK:
        s = S[iid]
        out.append({"title": title, "note": note, "spec": {k: s.get(k) for k in KEYS if k in s}})
    tools_openai = TOOLS
    tools_anthropic = [{"name": t["function"]["name"], "description": t["function"]["description"], "input_schema": t["function"]["parameters"]}
                       for t in TOOLS]
    (ROOT / "web/data/live_scenarios.json").write_text(json.dumps({"scenarios": out, "tools_openai": tools_openai,
                                                                   "tools_anthropic": tools_anthropic}))
    print([o["title"] for o in out])


if __name__ == "__main__":
    main()
