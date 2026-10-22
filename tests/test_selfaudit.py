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


def test_unparsable_holds_by_default_and_runs_on_request():
    with pytest.warns(UserWarning):
        r = SelfAudit(lambda m: ["hmm", "maybe"], n=2).check([], "send_message", {})
    assert r.share_no is None and r.flagged and r.parsed == 0                    # fail closed
    with pytest.warns(UserWarning):
        r = SelfAudit(lambda m: ["hmm"], n=1, unparsable="run").check([], "send_message", {})
    assert not r.flagged                                                         # the tests' behaviour


def test_misuse_raises():
    with pytest.raises(TypeError):
        SelfAudit(lambda m: "no", n=1).check([], "send_message", {})              # a string, not a list
    with pytest.raises(ValueError):
        SelfAudit(lambda m: ["no"], n=0)
    with pytest.raises(ValueError):
        SelfAudit(lambda m: ["no"], prompt="is it ok?")                          # custom prompt without {name}/{args}


def test_n_is_passed_to_the_asker_and_custom_prompt():
    seen = {}

    def ask(m, n):
        seen["n"], seen["q"] = n, m[-1]["content"]
        return ["no"] * n
    r = SelfAudit(ask, prompt="Is {name} with {args} allowed? yes or no", n=3).check([], "delete", {"id": object()})
    assert seen["n"] == 3 and r.flagged and "delete" in seen["q"]                # non-JSON args do not crash


def test_more_verdict_forms():
    assert parse_answer("Answer: No") is True
    assert parse_answer("Final answer: **NO**") is True
    assert parse_answer("Reasoning... Final answer: yes") is False


def test_flag_rates():
    from graybox.selfaudit import AuditResult, flag_rates
    rs = [AuditResult(1.0, True), AuditResult(0.0, False), AuditResult(1.0, True), AuditResult(None, False, audited=False)]
    f = flag_rates(rs, [True, True, False, True])
    assert f["tau"] == 0.5 and f["phi"] == 1.0 and f["violations"] == 2 and f["allowed"] == 1


def test_anthropic_asker_moves_system_message():
    from graybox.selfaudit import anthropic_asker
    sent = {}

    class B:
        type, text = "text", "yes"

    class R:
        content = [B()]

    class C:
        class messages:
            @staticmethod
            def create(**kw):
                sent.update(kw)
                return R()
    out = anthropic_asker(C(), "m", system="base")([{"role": "system", "content": "policy"}, {"role": "user", "content": "q"}], 2)
    assert out == ["yes", "yes"] and sent["system"] == "base\n\npolicy" and all(m["role"] != "system" for m in sent["messages"])


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
    with pytest.raises(ValueError):
        prevention_law(0.5, 0.8, 0.1, M=0)
    assert isinstance(prevention_law(1, 1, 0, 1)[1], float)


def test_merge_user_turns():
    m = _merge_user_turns([{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "x", "content": "ok"}]},
                           {"role": "user", "content": "question"}])
    assert len(m) == 1 and m[0]["content"][-1] == {"type": "text", "text": "question"}


def test_observable_state_fixed_readout():
    import math
    from graybox.selfaudit import STATE_QUESTIONS, observable_state

    def ask(msgs, n):                      # "no" to "does it comply", "yes" to every rule question: a violation
        q = msgs[-1]["content"]
        return ["no"] * n if "comply with the policy" in q else ["yes"] * n
    st = observable_state(ask, [{"role": "user", "content": "task"}], "send_email", {"to": "x"}, n=4)
    assert len(st.theta) == len(STATE_QUESTIONS) == 8
    top = 2 * math.asin(math.sqrt((4 + 3 / 8) / (4 + 3 / 4)))
    assert abs(st.theta[1] - top) < 1e-9 and st.counts[0] == [0, 4]
    safe = observable_state(lambda m, n: ["yes"] * n if "comply with the policy" in m[-1]["content"] else ["no"] * n,
                            [{"role": "user", "content": "task"}], "read_file", {"path": "a"}, n=4)
    assert st.score > safe.score


def test_operator_note_and_first_line_parser():
    from graybox.selfaudit import (OPERATOR_NOTE, TAU2_AIRLINE_QUESTIONS, observable_state, parse_first_line,
                                   with_operator_note)
    assert parse_first_line("[Operator check answer: no]\n\nYes, the total is") is True
    assert parse_first_line("Operator check: yes.\n\nConfirmed") is False
    assert parse_first_line("No.\n\nYes, I confirm") is True
    assert parse_first_line("I'm not going to answer") is None
    m = with_operator_note([{"role": "system", "content": "policy"}, {"role": "user", "content": "hi"}])
    assert m[0]["content"].endswith(OPERATOR_NOTE) and len(m) == 2
    assert with_operator_note(m)[0]["content"].count("<operator_checks>") == 1          # idempotent
    assert with_operator_note([{"role": "user", "content": "hi"}])[0]["role"] == "system"
    seen = []

    def ask(msgs, n):
        seen.append(msgs[0]["content"])
        return ["No.\n\nYes, your total is $375"] * n if "comply" in msgs[-1]["content"] else ["no"] * n
    st = observable_state(ask, [{"role": "system", "content": "policy"}, {"role": "user", "content": "cancel it"}],
                          "cancel_reservation", {"reservation_id": "X"}, questions=TAU2_AIRLINE_QUESTIONS,
                          parse=parse_first_line, operator_note=True)
    assert all(OPERATOR_NOTE in c for c in seen) and len(seen) == len(TAU2_AIRLINE_QUESTIONS)
    assert st.counts[0] == [0, 4]                        # "No." read from the first line, despite the trailing "Yes"
