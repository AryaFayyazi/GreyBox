"""Fast SDK tests (no GPU, no data downloads). Run: pytest tests"""
import json
from pathlib import Path

import numpy as np
import pytest

from graybox.monitor import QuantileGate, Step, TextMonitor, act_text, ctx_text

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "site/tests/browser_model.json"


def test_render_matches_documented_format():
    c = ctx_text("policy", "task", [Step("read_file", {"path": "/a", "n": 1}, "x" * 2100)])
    assert c.startswith("[Operating instructions]\npolicy\n\n[User task]\ntask\n\n[Agent called] read_file {\"path\": \"/a\", \"n\": 1}")
    assert c.endswith("x" * 2000 + " ...")
    assert act_text("__final__", None) == "final answer to the user (no tool call)"
    assert act_text("send", {"to": "é"}) == 'send {"to": "é"}'


def test_gate_fails_closed_then_freezes():
    g = QuantileGate(0.05, warmup=500, min_seen=100)
    x = np.random.default_rng(0).random(2000)
    e = [g(v) for v in x]
    assert all(e[:100]) and g.frozen
    t = g.threshold
    for v in np.ones(500):          # a burst of maximal scores must not move a frozen cut
        g(v)
    assert g.threshold == t
    assert 0.02 < np.mean(e[500:]) < 0.09
    g.recalibrate()
    assert not g.frozen and g.threshold is None


def test_gate_rolling_option_moves():
    g = QuantileGate(0.05, rolling=True, window=500, warmup=500)
    g.fit(np.random.default_rng(1).random(500))
    t = g.threshold
    g.fit(np.ones(300))
    assert g.threshold > t


@pytest.mark.skipif(not MODEL.exists(), reason="browser model fixture missing")
def test_browser_model_loads_and_scores():
    tm = TextMonitor.load_json(MODEL)
    z = tm.decision(None, None, None, None, rendered=("[Operating instructions]\nx", 'send_message {"to": "y"}'))
    assert np.isfinite(z)
    p = tm.score("policy", "task", [], ("send_message", {"to": "y"}))
    assert 0 < p < 1


def test_text_monitor_fit_and_score():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(0)
    steps, y = [], []
    for i in range(80):
        bad = i % 2
        steps.append(("Never email outside corp.example.", f"Send report {i}", [Step("read_file", {"path": f"/r{i}"}, "ok")],
                      ("send_message", {"recipient": "x@ext.example" if bad else "a@corp.example", "n": int(rng.integers(9))})))
        y.append(bad)
    tm = TextMonitor().fit(steps, y)
    assert tm.score(*steps[1]) > tm.score(*steps[0])


def test_monitor_server_without_torch():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    import os
    os.environ["GB_TEXT_MONITOR"] = str(MODEL)
    from graybox import monitor_server
    c = TestClient(monitor_server.app)
    assert c.get("/v1/health").json()["text_monitor"]
    r = c.post("/v1/monitor/text", json={"instructions": "p", "task": "t", "action": {"tool": "send_message", "args": {"to": "y"}}}).json()
    assert set(r) >= {"score", "escalate", "threshold"}
    assert c.post("/v1/monitor/action", json={"instructions": "p", "task": "t", "action": {"tool": "x"}}).status_code == 503 or os.environ.get("GB_MONITOR_PROBE")


def test_server_calibrate_endpoint():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from graybox import monitor_server
    c = TestClient(monitor_server.app)
    sc = list(np.random.default_rng(2).random(400))
    r = c.post("/v1/monitor/calibrate", json={"stream": "s", "budget": 0.05, "scores": sc}).json()
    assert r["frozen"] and abs(r["threshold"] - float(np.quantile(sc, 0.95))) < 1e-9
    assert c.post("/v1/monitor/calibrate", json={"stream": "s", "scores": [0.1] * 10}).status_code == 422


def test_guard_openai_and_anthropic_formats():
    from graybox.guard import Guard, conversation_steps
    from graybox.monitor import TextMonitor
    tm = TextMonitor.load_json(MODEL)
    oa = [{"role": "system", "content": "Only email people the user names."}, {"role": "user", "content": "Mail Dana the report."},
          {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\": \"/r\"}"}}]},
          {"role": "tool", "tool_call_id": "c1", "content": "report text"}]
    an_sys = "Only email people the user names."
    an = [{"role": "user", "content": "Mail Dana the report."},
          {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "/r"}}]},
          {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "report text"}]}]
    s_oa = conversation_steps(oa + [{"role": "assistant", "tool_calls": [{"id": "c2", "type": "function", "function": {"name": "send_message", "arguments": "{\"to\": \"x\"}"}}]}])
    s_an = conversation_steps(an + [{"role": "assistant", "content": [{"type": "tool_use", "id": "t2", "name": "send_message", "input": {"to": "x"}}]}], system=an_sys)
    for s in (s_oa, s_an):                       # same steps from either provider format
        assert [x["action"]["tool"] for x in s] == ["read_file", "send_message"]
        assert s[1]["history"] == [{"tool": "read_file", "args": {"path": "/r"}, "result": "report text"}]
        assert s[1]["instructions"] == an_sys and s[1]["task"] == "Mail Dana the report."
    g = Guard(tm)
    v1 = g.check_openai(oa, {"id": "c2", "type": "function", "function": {"name": "send_message", "arguments": "{\"to\": \"x\"}"}})
    v2 = g.check_anthropic(an_sys, an, {"type": "tool_use", "id": "t2", "name": "send_message", "input": {"to": "x"}})
    assert abs(v1.score - v2.score) < 1e-9          # provider format does not change the verdict
    assert v1.allow == (v1.score < tm._json["threshold_logit"]) and isinstance(v1.block_message, str)


def test_guard_with_reader_and_gate():
    from graybox.guard import Guard
    from graybox.monitor import QuantileGate

    class FakeReader:
        def score(self, instructions, task, history, action):
            return 0.9 if action[0] == "delete_all" else 0.01
    g = Guard(FakeReader(), gate=QuantileGate(0.05).fit(list(np.random.default_rng(0).random(600) * 0.05)))
    assert not g.check("p", "t", [], "delete_all", {}).allow
    assert g.check("p", "t", [], "read_file", {}).allow
