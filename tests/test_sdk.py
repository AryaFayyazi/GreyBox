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
    assert [s["label"] for s in st] == [0, 1, 0] and st[0]["group"] == "task: do it" and st[0]["conversation"] == "c"   # folds group tasks
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
    from graybox import reader_load
    monkeypatch.setattr(reader_load, "load_reader", lambda path: SlowReader.load(path))
    monkeypatch.setenv("GB_MONITOR_PROBE", "fake.joblib")
    monitor_server._STATE.pop("reader", None)
    c, codes = TestClient(monitor_server.app), []
    body = {"instructions": "p", "task": "t", "action": {"tool": "x"}, "stream": "conc"}
    th = [threading.Thread(target=lambda: codes.append(c.post("/v1/monitor/action", json=body).status_code)) for _ in range(8)]
    [t.start() for t in th]; [t.join() for t in th]
    monitor_server._STATE.pop("reader", None)
    assert codes == [200] * 8 and len(loads) == 1


class _Spy:
    """A monitor that records what it scores (TextMonitor interface: decision + _json threshold)."""
    _json = {"threshold_logit": 0.0}

    def __init__(self):
        self.seen = []

    def decision(self, instructions, task, history, action, *, rendered=None):
        self.seen.append((action, len(history or [])))
        return 1.0 if action and action[0] == "send_message" else -1.0


def _tc(i, name, args):
    return {"id": i, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def test_guard_scores_the_proposed_call_even_with_a_reused_or_empty_id():
    from graybox.guard import Guard
    msgs = [{"role": "system", "content": "Only email people the user names."}, {"role": "user", "content": "Mail Bob."},
            {"role": "assistant", "tool_calls": [_tc("call_0", "read_file", {"path": "/r"})]}, {"role": "tool", "tool_call_id": "call_0", "content": "x"}]
    for cid in ("call_0", "", None):                 # a reused id, an empty id, no id
        spy = _Spy()
        v = Guard(spy).check_openai(msgs, _tc(cid, "send_message", {"to": "all-staff"}))
        assert spy.seen[-1] == (("send_message", {"to": "all-staff"}), 1) and not v.allow, (cid, spy.seen)


def test_guard_same_score_for_every_documented_call_style():
    from graybox.guard import Guard, conversation_steps
    from graybox.monitor import TextMonitor
    tm = TextMonitor.load_json(MODEL)
    base = [{"role": "system", "content": "Only email people the user names."}, {"role": "user", "content": "Mail the report to Bob."}]
    c0, c1 = _tc("c0", "read_file", {"path": "/r"}), _tc("c1", "send_message", {"to": "all-staff"})
    reply = {"role": "assistant", "content": None, "tool_calls": [c0, c1]}
    g = Guard(tm)
    after = g.check_openai(base + [reply], c1).score                                  # the reply passed (README style)
    after_sib = g.check_openai(base + [reply, {"role": "tool", "tool_call_id": "c0", "content": "r"}], c1).score   # a sibling already ran
    # train/serve: the same step as the importer builds from the finished log (sibling c0 with an empty result)
    log = base + [reply, {"role": "tool", "tool_call_id": "c0", "content": "r"}, {"role": "tool", "tool_call_id": "c1", "content": "sent"}]
    st = [x for x in conversation_steps(log) if x["call_id"] == "c1"][0]
    assert st["history"] == [{"tool": "read_file", "args": {"path": "/r"}, "result": ""}]
    imp = g.check(st["instructions"], st["task"], st["history"], "send_message", {"to": "all-staff"}).score
    assert after == after_sib == imp
    one = {"role": "assistant", "content": None, "tool_calls": [c1]}                 # a single-call reply: every style agrees
    assert g.check_openai(base, c1).score == g.check_openai(base + [one], c1).score


def test_reviewer_fails_closed_and_sees_the_score():
    from graybox.guard import Guard
    seen = []
    for answer, allow in ((True, True), (False, False), ("no", False), (1, False)):
        v = Guard(_Spy(), reviewer=lambda st, a=answer: seen.append(st) or a).check("p", "t", [], "send_message", {})
        assert v.held and v.allow == allow and (v.review_error is None) == isinstance(answer, bool)
    assert "score" in seen[0] and "label" not in seen[0]

    def boom(st):
        raise RuntimeError("reviewer down")
    v = Guard(_Spy(), reviewer=boom).check("p", "t", [], "send_message", {})
    assert v.held and not v.allow and "reviewer down" in v.review_error


def test_text_monitor_trained_in_python_saves_and_guards(tmp_path):
    from graybox.guard import Guard
    from graybox.monitor import TextMonitor
    rng = np.random.default_rng(0)
    steps, y, g = [], [], []
    for i in range(120):
        bad = i % 6 == 0
        steps.append({"instructions": "Only email people the user names.", "task": f"Mail the report to user{i % 7}.",
                      "history": [{"tool": "read_file", "args": {"path": f"/r{i % 3}"}, "result": "report"}],
                      "action": {"tool": "send_message", "args": {"to": "all-staff" if bad else f"user{i % 7}", "n": int(rng.integers(9))}}})
        y.append(int(bad)); g.append(f"conv{i // 3}")
    tm = TextMonitor().fit(steps, y, g)
    tm.save(tmp_path / "m.json")
    tj = TextMonitor.load_json(tmp_path / "m.json")
    s = steps[0]
    z1 = tm.decision(s["instructions"], s["task"], s["history"], (s["action"]["tool"], s["action"]["args"]))
    z2 = tj.decision(s["instructions"], s["task"], s["history"], (s["action"]["tool"], s["action"]["args"]))
    assert abs(z1 - z2) < 1e-9 and tm.cv_auroc is not None
    v = Guard(tj).check(s["instructions"], s["task"], s["history"], "send_message", s["action"]["args"])
    assert v.allow == (z2 < tj._json["threshold_logit"]) and v.unit == "logit"
    z = np.array([tj.decision(x["instructions"], x["task"], x["history"], (x["action"]["tool"], x["action"]["args"])) for x in steps])
    assert z[np.array(y) == 1].mean() > z[np.array(y) == 0].mean()


def test_server_accepts_openai_shapes_and_reports_load_errors(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from graybox import monitor_server
    from graybox.monitor import TextMonitor, act_text, ctx_text, Step
    monkeypatch.setenv("GB_TEXT_MONITOR", str(MODEL))
    for k in ("text", "text_error"):
        monitor_server._STATE.pop(k, None)
    c, tm = TestClient(monitor_server.app), TextMonitor.load_json(MODEL)
    r = c.post("/v1/monitor/text", json={"context": "[Operating instructions]\nx", "action": {"tool": "send_message", "args": "{\"to\": \"a\"}"}})
    assert r.status_code == 200 and r.json()["unit"] == "logit"          # context alone; arguments as a JSON string
    assert abs(r.json()["score"] - round(tm.decision(None, None, None, None, rendered=("[Operating instructions]\nx", act_text("send_message", {"to": "a"}))), 4)) < 1e-9
    r = c.post("/v1/monitor/text", json={"instructions": "p", "task": "t", "history": [{"tool": "read_file", "args": None, "result": None}],
                                         "action": {"tool": "send_message", "args": {"to": "a"}}})
    assert r.status_code == 200 and abs(r.json()["score"] - round(tm.decision("p", "t", [Step("read_file", {}, "")], ("send_message", {"to": "a"})), 4)) < 1e-9
    monkeypatch.setenv("GB_TEXT_MONITOR", "/nonexistent/model.json")
    for k in ("text", "text_error"):
        monitor_server._STATE.pop(k, None)
    h = c.get("/v1/health").json()
    assert h["text_monitor"] is False and "FileNotFoundError" in h["text_monitor_error"]
    assert c.post("/v1/monitor/text", json={"action": {"tool": "x"}}).status_code == 503
    for k in ("text", "text_error"):
        monitor_server._STATE.pop(k, None)


def test_guard_follows_a_reader_probes_chat_setting():
    from graybox.guard import Guard

    class R:
        chat = True
    assert Guard(R(), gate=object()).chat and not Guard(R(), gate=object(), chat=False).chat


def test_guard_reply_detection_and_malformed_calls():
    from graybox.guard import Guard, conversation_steps, parse_label
    msgs = [{"role": "system", "content": "p"}, {"role": "user", "content": "Mail Bob."},
            {"role": "assistant", "tool_calls": [_tc("call_0", "send_message", {"to": "bob"})]}, {"role": "tool", "tool_call_id": "call_0", "content": "sent"}]
    for cid in ("call_0", "", None):               # same id and tool as an EXECUTED earlier call: not the reply
        spy = _Spy()
        Guard(spy).check_openai(msgs, _tc(cid, "send_message", {"to": "all-staff"}))
        assert spy.seen[-1] == (("send_message", {"to": "all-staff"}), 1), (cid, spy.seen)
    new = _tc("", "send_message", {"to": "all-staff"})
    spy = _Spy()                                    # an empty id with the reply appended: the call is not in its own history
    Guard(spy).check_openai(msgs + [{"role": "assistant", "tool_calls": [new]}], new)
    assert spy.seen[-1][1] == 1
    v = Guard(_Spy()).check_openai(msgs, {"id": "x", "type": "function", "function": {"name": "", "arguments": "{}"}})
    assert not v.allow and "malformed" in v.review_error
    assert [parse_label(x) for x in (1, 0, True, False, "1", "0", "true", "false", "yes", 2, None)] == [1, 0, 1, 0, 1, 0, 1, 0, None, None, None]
    conv = msgs + [{"role": "assistant", "tool_calls": [_tc("c9", "send_message", {"to": "x"})]}]
    assert [st["label"] for st in conversation_steps(conv, labels={"call_0": "0", "c9": "yes"})] == [0, None]


def test_server_text_gate_strict_bodies_and_chat(monkeypatch, tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from graybox import monitor_server
    monkeypatch.setenv("GB_TEXT_MONITOR", str(MODEL))
    for k in ("text", "text_error"):
        monitor_server._STATE.pop(k, None)
    c = TestClient(monitor_server.app)
    body = {"instructions": "p", "task": "t", "action": {"tool": "send_message", "args": {"to": "a"}}, "stream": "s1", "gate": True}
    z = c.post("/v1/monitor/text", json={k: v for k, v in body.items() if k != "gate"}).json()["score"]
    r = c.post("/v1/monitor/calibrate", json={"monitor": "text", "stream": "s1", "budget": 0.05, "scores": [z - 1 - i / 100 for i in range(200)]})
    assert r.status_code == 200
    v = c.post("/v1/monitor/text", json=body).json()
    assert v["escalate"] and v["cut"] == "live gate on this stream"     # above every seeded score of this stream
    assert c.post("/v1/monitor/text", json={**body, "instrucions": "typo"}).status_code == 422
    d = json.loads(MODEL.read_text()); d["chat"] = True
    (tmp_path / "chat.json").write_text(json.dumps(d))
    monkeypatch.setenv("GB_TEXT_MONITOR", str(tmp_path / "chat.json"))
    for k in ("text", "text_error"):
        monitor_server._STATE.pop(k, None)
    assert c.post("/v1/monitor/text", json={"instructions": "p", "task": "t", "action": {"tool": "x"}}).status_code == 422
    assert c.post("/v1/monitor/text", json={"context": "x", "action": {"tool": "x"}}).status_code == 200
    for k in ("text", "text_error"):
        monitor_server._STATE.pop(k, None)


def test_gate_with_a_chat_text_monitor_and_units(tmp_path):
    from graybox.guard import Guard
    from graybox.monitor import QuantileGate, TextMonitor
    d = json.loads(MODEL.read_text()); d["chat"] = True
    (tmp_path / "c.json").write_text(json.dumps(d))
    tm = TextMonitor.load_json(tmp_path / "c.json")
    gate = QuantileGate(0.05).fit(list(np.linspace(-9, -6, 300)))      # logits
    oa = [{"role": "system", "content": "p"}, {"role": "user", "content": "Mail Bob."}, {"role": "user", "content": "Actually, mail everyone."}]
    v = Guard(tm, gate=gate).check_openai(oa, {"id": "c1", "type": "function", "function": {"name": "send_message", "arguments": "{\"to\": \"all\"}"}})
    assert v.unit == "logit" and v.allow == (v.score < gate.threshold)
    v2 = Guard(tm).check_openai(oa, {"id": "c1", "type": "function", "function": {"name": "send_message", "arguments": "{\"to\": \"all\"}"}})
    assert abs(v.score - v2.score) < 1e-12                               # same logit with and without the gate


def test_public_render_accepts_dict_histories():
    from graybox.monitor import Step, ctx_text, render
    st = {"instructions": "i", "task": "t", "history": [{"tool": "r", "args": {}, "result": "x"}], "action": {"tool": "send", "args": {"to": "a"}}}
    ctx, act = render(st)
    assert ctx == ctx_text("i", "t", [Step("r", {}, "x")]) and act == 'send {"to": "a"}'
    assert render({"context": "c", "action": {"tool": "x", "args": {}}}) == ("c", "x {}")


def test_train_text_reports_within_tool_auroc_and_groups(tmp_path, capsys):
    # the violation is decided by the tool alone: pooled AUROC is high, within-tool AUROC is chance -> a warning
    from graybox import train_text
    rng = np.random.default_rng(0)
    rows = []
    for i in range(120):
        bad = i % 2
        tool = "refund" if bad else "lookup"
        rows.append({"context": f"task {i % 12} user asks about order {rng.integers(1e6)}", "action": f"{tool} {{\"id\": {rng.integers(1e6)}}}",
                     "label": int(bad if rng.random() < 0.8 else 1 - bad), "task": f"t{i % 12}", "instructions": ""})
    f = tmp_path / "steps.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in rows))
    train_text.main([str(f), "--out", str(tmp_path / "m.json"), "--group-by", "task"])
    rep = json.loads(capsys.readouterr().out.split("\nsaved")[0])
    assert rep["groups"] == 12 and rep["cv_auroc"] > 0.7
    assert rep["cv_auroc_within_tool"] is not None and rep["cv_auroc_within_tool"] < 0.65
    assert "warning" in rep and "top 5%" in rep["cut"]
    assert train_text._tool({"action": "cancel_reservation {\"reservation_id\": \"Q6\"}"}) == "cancel_reservation"
    assert train_text._tool({"action": {"tool": "send_email", "args": {}}}) == "send_email"


def test_caught_at_budget_sends_exactly_the_budget_with_ties():
    from graybox.stats import caught_at_budget
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0] * 10)
    assert caught_at_budget(y, np.where(y == 1, 9.0, 0.0), 0.2) == 1.0         # 20 reviewed = the 20 positives
    tied = np.where(np.arange(100) < 50, 100.0, 0.0)          # half the units at 100: a 5% budget reviews 5 of 50 tied units
    assert abs(caught_at_budget(y, tied, 0.05) - (5 / 50 * y[:50].sum()) / y.sum()) < 1e-12
    assert abs(caught_at_budget(y, np.zeros(100), 0.05) - 0.05) < 1e-12   # a constant score catches at random
    s = np.random.default_rng(0).normal(size=100)              # without ties: the top 5 units
    assert caught_at_budget(y, s, 0.05) == y[np.argsort(-s)[:5]].sum() / y.sum()


def test_guard_hold_tools_rule_escalates_every_call_of_a_tool():
    from graybox.guard import Guard
    from graybox.monitor import QuantileGate

    class FakeReader:
        def score(self, instructions, task, history, action):
            return 0.01                                       # never risky by score
    gate = QuantileGate(0.05).fit(list(np.random.default_rng(0).random(600) * 0.05))
    n0 = len(gate.buf)
    g = Guard(FakeReader(), gate=gate, hold_tools={"issue_refund"})
    v = g.check("p", "t", [], "issue_refund", {"amount": 5})
    assert not v.allow and v.by_rule and not v.held          # blocked by the rule (no reviewer)
    assert g.check("p", "t", [], "read_file", {}).allow
    g2 = Guard(FakeReader(), gate=QuantileGate(0.05).fit([0.0] * 600), reviewer=lambda st: True, hold_tools=["issue_refund"])
    v = g2.check("p", "t", [], "issue_refund", {})
    assert v.allow and v.held and v.by_rule                   # held for review by the rule; the reviewer lets it run
    assert gate.frozen and n0 == 500                          # the gate froze its cut after the 500-score warm-up, as usual



def test_anthropic_sdk_blocks_inside_dict_messages_keep_the_history():
    # the Anthropic SDK's own loop: messages.append({"role": "assistant", "content": response.content}) (SDK block objects)
    anthropic = pytest.importorskip("anthropic")
    from anthropic.types import TextBlock, ToolUseBlock
    from graybox.guard import Guard, conversation_steps
    from graybox.monitor import TextMonitor
    tm = TextMonitor.load_json(MODEL)
    plain = [{"role": "user", "content": "Send the Q3 report to the board."},
             {"role": "assistant", "content": [{"type": "text", "text": "Reading it."},
                                               {"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "/fin/q3.pdf"}}]},
             {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "CONFIDENTIAL ..."}]}]
    sdk = [plain[0], {"role": "assistant", "content": [TextBlock(type="text", text="Reading it."),
                                                       ToolUseBlock(type="tool_use", id="t1", name="read_file", input={"path": "/fin/q3.pdf"})]}, plain[2]]
    call = ToolUseBlock(type="tool_use", id="t2", name="send_email", input={"to": "press@news.example", "body": "attached"})
    g = Guard(tm)
    a = g.check_anthropic("policy", plain, {"type": "tool_use", "id": "t2", "name": "send_email", "input": {"to": "press@news.example", "body": "attached"}})
    b = g.check_anthropic("policy", sdk, call)
    assert a.score == b.score and a.allow == b.allow
    sa = conversation_steps(plain + [{"role": "assistant", "content": [call.model_dump()]}], system="policy")
    sb = conversation_steps(sdk + [{"role": "assistant", "content": [call]}], system="policy")
    assert [s["history"] for s in sa] == [s["history"] for s in sb] and len(sb[-1]["history"]) == 1   # the read_file call is kept


def test_train_text_cli_guards_against_common_mistakes(tmp_path, capsys):
    from graybox import train_text
    rows = [{"context": f"task {i % 6} order {i}", "action": f"refund {{\"id\": {i}}}" if i % 3 == 0 else f"lookup {{\"id\": {i}}}",
             "label": int(i % 3 == 0), "task": f"t{i % 6}"} for i in range(60)]
    f = tmp_path / "s.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in rows))
    out = tmp_path / "m.json"
    out.write_text("{}")
    with pytest.raises(SystemExit, match="exists: pass --force"):              # never overwrite a monitor silently
        train_text.main([str(f), "--out", str(out)])
    with pytest.raises(SystemExit, match="have no field 'tsak'"):              # a typo in --group-by is an error
        train_text.main([str(f), "--eval-only", "--group-by", "tsak"])
    train_text.main([str(f), "--eval-only", "--group-by", "task"])
    rep = json.loads(capsys.readouterr().out.split("\nnot saved")[0])
    assert rep["groups"] == 6 and rep["privacy_filter"] and 0 < rep["most_possible_at_budget"] <= 1 and out.read_text() == "{}"
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps(rows[0]) + "\n{not json\n")
    with pytest.raises(SystemExit, match="line 2 is not valid JSON"):
        train_text.main([str(bad), "--eval-only"])
    arr = tmp_path / "a.json"
    arr.write_text(json.dumps(rows))                                          # a .json array is read like JSON lines
    train_text.main([str(arr), "--eval-only"])
    nolab = tmp_path / "n.jsonl"
    nolab.write_text(json.dumps({"messages": [{"role": "user", "content": "x"}, {"role": "assistant", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]}], "labels": {"zzz": 1}}))
    with pytest.raises(SystemExit, match="none of the 1 tool calls has a matching id"):
        train_text.main([str(nolab), "--eval-only"])


def test_text_monitor_privacy_filter_keeps_secrets_out_of_the_file():
    from graybox.monitor import TextMonitor
    rng = np.random.default_rng(1)
    steps = [{"context": f"user asks to send the report token tok_live_d8tdcrgi0c{i % 3} to felix lund", "action": "send_email {\"to\": \"x\"}" if i % 2 else "read_file {\"p\": \"y\"}"} for i in range(80)]
    y = [i % 2 for i in range(80)]
    j = TextMonitor().fit(steps, y, [str(i % 8) for i in range(80)])._json
    vocab = set(j["vc"]["vocab"]) | set(j["va"]["vocab"])
    assert not any(any(ch.isdigit() for ch in w) or "tok_live" in w for w in vocab) and j["privacy"]
    j2 = TextMonitor().fit(steps, y, [str(i % 8) for i in range(80)], privacy=False)._json
    assert any("tok_live" in w for w in set(j2["vc"]["vocab"]))


def test_server_api_key_strict_inputs_and_persistent_gates(monkeypatch, tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from graybox import monitor_server
    monkeypatch.setenv("GB_TEXT_MONITOR", str(MODEL))
    monkeypatch.setenv("GB_API_KEY", "s3cret")
    monkeypatch.setenv("GB_GATE_STATE", str(tmp_path / "gates.json"))
    for k in ("text", "text_error"):
        monitor_server._STATE.pop(k, None)
    monitor_server._STATE["gates"] = {}
    body = {"instructions": "p", "task": "t", "action": {"tool": "send_message", "args": {"to": "a"}}}
    H = {"Authorization": "Bearer s3cret"}
    with TestClient(monitor_server.app) as c:
        assert c.get("/v1/health").json()["auth_required"]                        # health stays open
        assert c.post("/v1/monitor/text", json=body).status_code == 401           # no key
        assert c.post("/v1/monitor/text", json=body, headers={"Authorization": "Bearer nope"}).status_code == 401
        r = c.post("/v1/monitor/text", json=body, headers=H)
        assert r.status_code == 200 and r.json()["warning"] is None
        assert c.post("/v1/monitor/text", json={**body, "action": {"tool": ""}}, headers=H).status_code == 422   # empty tool
        assert c.post("/v1/monitor/text", json={**body, "budget": 0.1}, headers=H).status_code == 422          # budget without gate
        assert c.post("/v1/monitor/text", json={"context": "x", "action": {"tool": "t"}}, headers=H).json()["warning"]
        r = c.post("/v1/monitor/calibrate", json={"monitor": "text", "stream": "s9", "scores": [-5 + i / 100 for i in range(200)]}, headers=H)
        assert r.status_code == 200
        thr = r.json()["threshold"]
    monitor_server._STATE["gates"] = {}                                           # a restart: the gate comes back from the file
    with TestClient(monitor_server.app) as c:
        v = c.post("/v1/monitor/text", json={**body, "stream": "s9", "gate": True}, headers=H).json()
        assert v["threshold"] == round(thr, 4) and c.get("/v1/health").json()["gates"] >= 1
    for k in ("text", "text_error"):
        monitor_server._STATE.pop(k, None)
    monitor_server._STATE["gates"] = {}


def test_reviewer_timeout_blocks_and_async_checks_match():
    import asyncio
    import time as _t
    from graybox.guard import Guard
    from graybox.monitor import QuantileGate, TextMonitor

    class FakeReader:
        def score(self, instructions, task, history, action):
            return 0.9
    slow = Guard(FakeReader(), gate=QuantileGate(0.05).fit([0.0] * 600), reviewer=lambda st: _t.sleep(2) or True, reviewer_timeout=0.2)
    t0 = _t.perf_counter()
    v = slow.check("p", "t", [], "delete_all", {})
    assert not v.allow and v.held and "TimeoutError" in v.review_error and _t.perf_counter() - t0 < 1.5
    g = Guard(TextMonitor.load_json(MODEL))
    msgs = [{"role": "system", "content": "p"}, {"role": "user", "content": "send it"}]
    call = {"id": "c1", "type": "function", "function": {"name": "send_message", "arguments": '{"to": "a"}'}}
    assert asyncio.run(g.acheck_openai(msgs, call)).score == g.check_openai(msgs, call).score


def test_openai_responses_api_items_score_like_chat_completions():
    from graybox.guard import Guard
    from graybox.monitor import TextMonitor
    g = Guard(TextMonitor.load_json(MODEL))
    chat = [{"role": "system", "content": "policy"}, {"role": "user", "content": "share the file"},
            {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": '{"p": "/a"}'}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "secret"}]
    items = [{"role": "user", "content": [{"type": "input_text", "text": "share the file"}]},
             {"type": "function_call", "call_id": "c1", "name": "read_file", "arguments": '{"p": "/a"}'},
             {"type": "function_call_output", "call_id": "c1", "output": "secret"}]
    fc = {"type": "function_call", "call_id": "c2", "name": "send_message", "arguments": '{"to": "press"}'}
    a = g.check_openai(chat, {"id": "c2", "type": "function", "function": {"name": "send_message", "arguments": '{"to": "press"}'}})
    b = g.check_openai_responses(items, fc, instructions="policy")
    assert a.score == b.score
