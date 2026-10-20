import pytest

from graybox.selfaudit import SelfAudit, _merge_user_turns, parse_answer, prevention_law


def test_parse_answer():
    assert parse_answer("yes") is False
    assert parse_answer("No.\n\nThe approval came from someone else.") is True
    assert parse_answer("Rule 1: fine.\nNo issue with rule 2.\n\nyes") is False      # the verdict line wins
    assert parse_answer("All rules checked. Verdict: **no**. I will hold.") is True
    assert parse_answer("") is None
    assert parse_answer("I'll proceed with the call.") is None


def test_check_flags_and_reads():
    a = SelfAudit(lambda m: ["no", "No.", "yes"], prompt="ask", n=3, cut=0.67)
    r = a.check([{"role": "user", "content": "task"}], "send_message", {"to": "x"})
    assert r.audited and abs(r.share_no - 2 / 3) < 1e-9 and not r.flagged       # 2/3 < 0.67: G10 flagged only unanimous "no"
    r = SelfAudit(lambda m: ["no", "No.", "hmm"], prompt="ask", n=3, cut=0.67).check([], "send_message", {})
    assert r.share_no == 1.0 and r.flagged                                       # every parsable answer "no"
    a = SelfAudit(lambda m: ["no", "no", "yes"], prompt="ask", n=3, cut=0.5)
    assert a.check([], "send_message", {}).flagged
    r = a.check([], "read_file", {"path": "/a"})
    assert not r.audited and not r.flagged


def test_question_is_appended_as_user_turn():
    seen = {}

    def ask(m):
        seen["m"] = m
        return ["yes"]
    SelfAudit(ask, prompt="checklist", n=1).check([{"role": "user", "content": "t"}], "delete_record", {"id": 1})
    assert seen["m"][-1]["role"] == "user" and "delete_record" in seen["m"][-1]["content"]


def test_unparsable_is_not_flagged():
    r = SelfAudit(lambda m: ["hmm", "maybe"], n=2).check([], "send_message", {})
    assert r.share_no is None and not r.flagged


def test_prevention_law():
    run, rev = prevention_law(R=0.5, tau=1.0, phi=0.0, M=1)
    assert run == 0 and rev == 0.5                       # a perfect audit: every violation goes to review
    run, rev = prevention_law(R=0.5, tau=0.0, phi=0.0, M=4)
    assert run == 0.5 and rev == 0                       # a blind audit changes nothing
    r1, _ = prevention_law(0.3, 0.8, 0.1, M=1)
    r4, v4 = prevention_law(0.3, 0.8, 0.1, M=4)
    assert abs(r1 - 0.3 * 0.2) < 1e-12 and r4 > r1 and v4 < 0.3 * 0.8 + 0.7 * 0.1   # more draws: fewer reviews
    with pytest.raises(ValueError):
        prevention_law(1.2, 0.5, 0.5)


def test_merge_user_turns():
    m = _merge_user_turns([{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "x", "content": "ok"}]},
                           {"role": "user", "content": "question"}])
    assert len(m) == 1 and m[0]["content"][-1] == {"type": "text", "text": "question"}
