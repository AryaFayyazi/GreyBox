import json

import numpy as np

from graybox.explore import build, render

Q = [{"q": "Does this call comply with the policy?", "sign": -1}, {"q": "Does it touch a resource the user does not own?", "sign": 1}]


def _dec(ep, step, label, yes):
    return {"episode": ep, "step": step, "call": f"call-{step}", "label": label, "stratum": "s",
            "draws": [{"action": f"a{step}", "counts": [[4 - yes, 4], [yes, 4]], "label": label}, {"action": None}]}


def test_build_places_every_decision_and_scores_risk():
    dec = [_dec("e1", 0, 0, 0), _dec("e1", 1, 1, 4), _dec("e2", 0, 0, 1), _dec("e2", 1, 1, 3)]
    d = build(dec, Q)
    steps = [s for e in d["episodes"] for s in e["steps"]]
    assert len(d["episodes"]) == 2 and len(steps) == 4 and all("xy" in s for s in steps)
    fc = {(s["act"], s["y"]): s["fc"] for s in steps}
    assert fc[("call-1", 1)] > fc[("call-0", 0)]                     # more 'violation' answers -> higher signed read
    assert steps[0]["draws"][1]["th"] is None                        # a draw that ends the turn
    html = render(d, title="Test <Explorer>")
    assert "Test &lt;Explorer&gt;" in html and "__GB_DATA__" not in html
    payload = html.split('id="gbdata">', 1)[1].split("</script>", 1)[0]
    assert json.loads(payload.replace("<\\/", "</"))["questions"][1]["sign"] == 1


def test_build_without_labels():
    dec = [{"episode": "e", "step": i, "call": "c", "draws": [{"action": "a", "counts": [[i, 4], [4 - i, 4]]}]} for i in range(3)]
    d = build(dec, Q)
    assert all("u" not in s for e in d["episodes"] for s in e["steps"])
    assert np.isfinite(d["centre"]).all()
