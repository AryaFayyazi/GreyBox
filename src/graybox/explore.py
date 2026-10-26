"""GreyBox Explorer: browse a closed agent's decisions in its estimated geometry, from its API alone.

The counterpart of a neuron explorer for a model without weights. A question is one axis of the estimated geometry
(its angle 2 arcsin sqrt P(yes)); a decision is placed by the angles of the agent's own sampled next actions. Nothing
is trained and no label is needed; labels, if given, only colour the map and score the axes.

Input: JSON lines, one per decision before a tool call,
  {"episode": "ep-1", "step": 3, "call": "restart-service web-1", "label": 1, "stratum": "restarts",
   "draws": [{"action": "restart-service web-1", "counts": [[yes, valid], ...one per question], "label": 1},
             {"action": null}]}                  # a draw that ends the turn without a tool call
  "label", "stratum" and each draw's "label" are optional; "counts" has one [yes, valid] pair per question.
Questions: JSON list of {"q": text, "sign": +1 or -1}, sign +1 when "yes" points toward a violation.

  graybox-explore decisions.jsonl --questions questions.json -o explorer.html [--title ...]
"""
import argparse
import json
from importlib import resources
from pathlib import Path

import numpy as np

from .geometry import fisher_angles_from_counts


def _raw(counts):
    c = np.asarray(counts, float)
    if c.ndim != 2 or c.shape[1] != 2:
        return None
    return c


def build(decisions, questions):
    """Explorer data from decision records (see the module docstring). Returns the dict the page reads.

    Geometry risk (the paper's G57 statistic): mean over the draws of the distance of each draw's angles from the
    agent's typical state (the median over its distinct read actions); a draw without answers on every question scores 0.
    Signed forecast (G46's read-out): mean over the draws of the mean signed angle; a draw that ends the turn, or has no
    answer at all, scores the read-out's minimum (every question answered on its safe side)."""
    sign = np.array([float(q["sign"]) for q in questions])
    m = len(questions)
    rows, distinct, n_max = [], {}, 1
    for d in decisions:
        cs = []
        for dr in d.get("draws") or []:
            c = _raw(dr["counts"]) if dr.get("action") is not None and dr.get("counts") is not None else None
            if c is not None and len(c) != m:
                raise ValueError(f"decision {d.get('episode')}/{d.get('step')}: {len(c)} answers, {m} questions")
            if c is not None:
                n_max = max(n_max, int(c[:, 1].max()))
                if c[:, 1].min() > 0:
                    distinct[(str(d.get("episode")), d.get("step"), str(dr.get("action")))] = fisher_angles_from_counts(c[:, 0], c[:, 1])
            cs.append(c)
        rows.append((d, cs))
    if not distinct:
        raise ValueError("no draw has answers on every question")
    centre = np.median(np.stack(list(distinct.values())), 0)
    safest = fisher_angles_from_counts(np.where(sign < 0, n_max, 0), np.full(m, n_max))
    lo = float(np.mean(sign * safest))
    episodes, order, pts = {}, [], []
    for d, cs in rows:
        ep = str(d.get("episode", "episode"))
        if ep not in episodes:
            episodes[ep] = {"id": ep, "steps": []}
            order.append(ep)
        s = {"t": int(d.get("step", len(episodes[ep]["steps"]))), "act": str(d.get("call", "")), "bad": bool(d.get("label"))}
        if d.get("label") is not None:
            s.update(u=1, y=int(d["label"]), st=str(d.get("stratum", "all")))
        if cs:
            full = [fisher_angles_from_counts(c[:, 0], c[:, 1]) if c is not None and c[:, 1].min() > 0 else None for c in cs]
            dist = [float(np.linalg.norm(a - centre)) if a is not None else 0.0 for a in full]
            read = [float(np.mean(sign * fisher_angles_from_counts(c[:, 0], c[:, 1]))) if c is not None and c[:, 1].sum() > 0 else lo for c in cs]
            s["geo"], s["fc"] = round(float(np.mean(dist)), 4), round(float(np.mean(read)), 4)
            s["draws"] = [{"a": str(dr.get("action")) if dr.get("action") is not None else "(ends turn)",
                           "th": [round(float(v), 3) for v in a] if a is not None else None, "bad": int(bool(dr.get("label")))}
                          for dr, a in zip(d["draws"], full)]
            ok = [a for a in full if a is not None]
            if ok:
                s["m"] = np.mean(ok, 0)
                pts.append((ep, len(episodes[ep]["steps"])))
        episodes[ep]["steps"].append(s)
    X = np.stack([episodes[e]["steps"][i]["m"] for e, i in pts])
    Xc = X - X.mean(0)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = Xc @ Vt[:2].T if Xc.shape[1] > 1 else np.hstack([Xc, np.zeros_like(Xc)])
    for (e, i), z in zip(pts, Z):
        st = episodes[e]["steps"][i]
        st["xy"] = [round(float(z[0]), 4), round(float(z[1]) if len(z) > 1 else 0.0, 4)]
        del st["m"]
    var = (S[:2] ** 2 / (S ** 2).sum()) if S.sum() > 0 else np.zeros(2)
    return {"questions": [{"q": q["q"], "sign": int(q["sign"])} for q in questions], "centre": [round(float(v), 4) for v in centre],
            "var": [round(float(v), 4) for v in var], "episodes": [episodes[e] for e in order]}


def render(data, title="GreyBox Explorer", subtitle=None) -> str:
    """Self-contained HTML page (no server, no network beyond optional web fonts)."""
    tpl = resources.files("graybox").joinpath("explorer_template.html").read_text(encoding="utf-8")
    sub = subtitle or ("Read through the agent's API only. Each point is one decision before a tool call, placed by the estimated "
                       "geometry of the agent's own sampled next actions. Pick a point to see what it would do and how each "
                       "action reads on the question-axes.")
    esc = lambda s: s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")  # noqa: E731
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    return tpl.replace("__GB_TITLE__", esc(title)).replace("__GB_SUB__", esc(sub)).replace("__GB_DATA__", payload)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="graybox-explore", description=__doc__.split("\n")[0])
    ap.add_argument("decisions", help="JSON lines, one decision per line (see graybox.explore)")
    ap.add_argument("--questions", required=True, help='JSON list of {"q": text, "sign": +1|-1}')
    ap.add_argument("-o", "--out", default="explorer.html")
    ap.add_argument("--title", default="GreyBox Explorer")
    ap.add_argument("--subtitle", default=None)
    a = ap.parse_args(argv)
    dec = [json.loads(x) for x in Path(a.decisions).read_text(encoding="utf-8").splitlines() if x.strip()]
    qs = json.loads(Path(a.questions).read_text(encoding="utf-8"))
    data = build(dec, qs)
    Path(a.out).write_text(render(data, a.title, a.subtitle), encoding="utf-8")
    n = sum(len(e["steps"]) for e in data["episodes"])
    print(f"saved {a.out}: {len(data['episodes'])} episodes, {n} decisions, {len(qs)} question-axes")


def cli():
    main()


if __name__ == "__main__":
    main()
