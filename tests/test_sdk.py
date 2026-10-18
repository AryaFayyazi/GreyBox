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


def test_train_reader_data_side(tmp_path):
    from graybox.monitor import Q, act_text, ctx_text, Step
    from graybox.train_reader import load_steps, prompt_of
    conv = {"id": "c", "system": "policy", "messages": [
        {"role": "user", "content": "do it"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "/a"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "x"}]},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t2", "name": "send", "input": {"to": "y"}}]}],
        "labels": {"t1": 0, "t2": 1}}
    step = {"instructions": "policy", "task": "do it", "history": [], "action": {"tool": "read_file", "args": {}}, "label": 0}
    f = tmp_path / "d.jsonl"
    f.write_text(json.dumps(conv) + "\n" + json.dumps(step) + "\n")
    st = load_steps(str(f))
    assert [s["label"] for s in st] == [0, 1, 0] and st[0]["group"] == "c"
    assert prompt_of(st[1]) == Q.format(ctx=ctx_text("policy", "do it", [Step("read_file", {"path": "/a"}, "x")]), cot="", act=act_text("send", {"to": "y"}))


def test_guard_hold_for_review():
    from graybox.guard import Guard
    from graybox.monitor import QuantileGate

    class FakeReader:
        def score(self, instructions, task, history, action):
            return 0.9 if action[0].startswith("delete") else 0.01
    gate = QuantileGate(0.05).fit(list(np.random.default_rng(0).random(600) * 0.05))
    seen = []
    g = Guard(FakeReader(), gate=gate, reviewer=lambda st: seen.append(st["action"]["tool"]) or st["action"]["tool"] == "delete_tmp")
    v = g.check("p", "t", [], "delete_tmp", {})
    assert v.held and v.allow                      # held, the reviewer lets it run
    v = g.check("p", "t", [], "delete_all", {})
    assert v.held and not v.allow                  # held, the reviewer blocks it
    v = g.check("p", "t", [], "read_file", {})
    assert v.allow and not v.held                  # not escalated: never reaches the reviewer
    assert seen == ["delete_tmp", "delete_all"]


def test_chat_context_keeps_later_user_turns():
    from graybox.guard import Guard, conversation_steps
    from graybox.monitor import TextMonitor, act_text
    sysp = "Cancel only if the booking is under 24 hours old or the reason is covered by insurance."
    oa = [{"role": "system", "content": sysp}, {"role": "user", "content": "Cancel booking Q69X3R."},
          {"role": "assistant", "content": "Why do you want to cancel?"}, {"role": "user", "content": "My plans changed."},
          {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "get_reservation", "arguments": "{\"id\": \"Q69X3R\"}"}}]},
          {"role": "tool", "tool_call_id": "c1", "content": "booked 2024-05-01, no insurance"},
          {"role": "assistant", "content": None, "tool_calls": [{"id": "c2", "type": "function", "function": {"name": "cancel_reservation", "arguments": "{\"id\": \"Q69X3R\"}"}}]}]
    an = [{"role": "user", "content": "Cancel booking Q69X3R."}, {"role": "assistant", "content": [{"type": "text", "text": "Why do you want to cancel?"}]},
          {"role": "user", "content": "My plans changed."},
          {"role": "assistant", "content": [{"type": "tool_use", "id": "c1", "name": "get_reservation", "input": {"id": "Q69X3R"}}]},
          {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "booked 2024-05-01, no insurance"}]},
          {"role": "assistant", "content": [{"type": "tool_use", "id": "c2", "name": "cancel_reservation", "input": {"id": "Q69X3R"}}]}]
    plain = conversation_steps(oa)
    assert "context" not in plain[-1]              # default unchanged
    s_oa, s_an = conversation_steps(oa, chat=True), conversation_steps(an, system=sysp, chat=True)
    c = s_oa[-1]["context"]
    assert c == s_an[-1]["context"]                # same context from either provider format
    for part in ("[User task]\nCancel booking Q69X3R.", "[Agent said] Why do you want to cancel?", "[User] My plans changed.",
                 "[Agent called] get_reservation", "[Tool result] booked 2024-05-01, no insurance"):
        assert part in c
    assert c.index("[User] My plans changed.") < c.index("[Agent called] get_reservation")
    tm = TextMonitor.load_json(MODEL)
    v = Guard(tm, chat=True).check_openai(oa[:-1], oa[-1]["tool_calls"][0])
    assert abs(v.score - tm.decision(None, None, None, None, rendered=(c, act_text("cancel_reservation", {"id": "Q69X3R"})))) < 1e-9


def test_server_accepts_prebuilt_context():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    import os
    os.environ["GB_TEXT_MONITOR"] = str(MODEL)
    from graybox import monitor_server
    from graybox.monitor import TextMonitor, act_text
    c, tm = TestClient(monitor_server.app), TextMonitor.load_json(MODEL)
    ctx = "[Operating instructions]\nOnly email named people.\n\n[User task]\nMail Dana.\n\n[User] Also cc all-staff."
    r = c.post("/v1/monitor/text", json={"instructions": "", "task": "", "context": ctx, "action": {"tool": "send_message", "args": {"to": "all"}}}).json()
    assert abs(r["score"] - round(tm.decision(None, None, None, None, rendered=(ctx, act_text("send_message", {"to": "all"}))), 4)) < 1e-9


def test_guard_uses_the_models_context_setting(tmp_path):
    from graybox.guard import Guard
    from graybox.monitor import TextMonitor
    d = json.loads(MODEL.read_text())
    assert not Guard(TextMonitor.load_json(MODEL)).chat
    d["chat"] = True
    (tmp_path / "m.json").write_text(json.dumps(d))
    assert Guard(TextMonitor.load_json(tmp_path / "m.json")).chat
    assert not Guard(TextMonitor.load_json(tmp_path / "m.json"), chat=False).chat


def test_embed_monitor_scores_and_guards():
    from graybox.embed_monitor import EmbedMonitor
    from graybox.guard import Guard
    d = {"type": "embed", "encoder": "minilm", "mean": [0.0, 1.0], "scale": [1.0, 2.0], "w": [2.0, -1.0], "b": 0.5, "threshold_logit": 1.0, "chat": True}
    m = EmbedMonitor(d)

    class FakeEncoder:                                 # x = (len(ctx) % 3, len(act) % 5): no download in tests
        def encode(self, ctxs, acts):
            return np.array([[len(c) % 3, len(a) % 5] for c, a in zip(ctxs, acts)], np.float32)
    m._enc = FakeEncoder()
    x = FakeEncoder().encode(["ab"], ["send_message {}"])[0]
    want = 0.5 + 2.0 * (x[0] - 0.0) / 1.0 - 1.0 * (x[1] - 1.0) / 2.0
    assert abs(m.decision(None, None, None, None, rendered=("ab", "send_message {}")) - want) < 1e-9
    g = Guard(m)
    assert g.chat                                      # follows the file's setting
    v = g.check_openai([{"role": "user", "content": "hi"}], {"id": "c1", "type": "function", "function": {"name": "send_message", "arguments": "{}"}})
    assert v.allow == (v.score < 1.0)


def test_server_loads_the_reader_once_under_concurrent_first_requests(monkeypatch):
    pytest.importorskip("fastapi")
    import os, threading, time as _t
    from fastapi.testclient import TestClient
    from graybox import monitor_server, monitor
    loads = []

    class SlowReader:
        @classmethod
        def load(cls, path):
            loads.append(1); _t.sleep(0.3)
            return cls()

        def score(self, instructions, task, history, action):
            return 0.01
    monkeypatch.setattr(monitor, "ReaderMonitor", SlowReader)
    monkeypatch.setenv("GB_MONITOR_PROBE", "fake.joblib")
    monitor_server._STATE.pop("reader", None)
    c, codes = TestClient(monitor_server.app), []
    body = {"instructions": "p", "task": "t", "action": {"tool": "x"}, "stream": "conc"}
    th = [threading.Thread(target=lambda: codes.append(c.post("/v1/monitor/action", json=body).status_code)) for _ in range(8)]
    [t.start() for t in th]; [t.join() for t in th]
    monitor_server._STATE.pop("reader", None)
    assert codes == [200] * 8 and len(loads) == 1
