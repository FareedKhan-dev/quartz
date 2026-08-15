"""Scoring Errand: four numbers that a single accuracy would have hidden.

Published tool-calling benchmarks tend to report one number, and that number
conflates two very different abilities. Choosing among five declared tools is
classification over a few classes and saturates at ten thousand training
examples; copying `"kitchen"` out of "dim the kitchen to ten percent" is copying
over an unbounded space and is still improving at 1.2 million. A suite that
reports their average tells you which of those you are short of only by
accident, so this module reports them apart:

- `tool_selection`  did it call the right tools, in the right order?
- `exact_call`      and were all of the arguments right too?
- `argument_grounding` per argument, how many values were reproduced exactly?
- `refusal`         on off-topic input, did it correctly call nothing?
- `schema_valid`    could the call have been executed at all?

Schema validity is a tautology when Trellis is on -- the decoder makes an
invalid call unrepresentable, so the number is 100 percent by construction. It
is reported anyway because it is the number that tells you the constrained
decoder is actually running, and because the unconstrained column is the whole
argument for having built it.

The last two pieces are about the confidence, not the call. `expected
calibration error` bins by *predicted confidence*, never by perplexity: binning
by the training signal would only show that the head tracks its own label.
`threshold_sweep` then turns a calibrated score into the decision a product has
to make, which is how much to keep and how much to escalate.

numpy only. Scoring a benchmark must not require the framework that trained the
model, or the numbers can only be reproduced on the machine that made them.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

#: The thresholds the post sweeps. 0.6 keeps 83.4 percent of requests on the
#: device and is right 70.6 percent of the time; 0.8 keeps just over half and is
#: right 82.4. The failure mode to prefer is escalation, not wrong execution.
DEFAULT_THRESHOLDS: tuple[float, ...] = (0.30, 0.50, 0.60, 0.70, 0.80, 0.90)

#: Bins for the reliability diagram. Ten over 12,000 held-out calls leaves about
#: a thousand a bin, which is enough for the bin mean to mean something.
DEFAULT_BINS = 10


# --- comparing calls -------------------------------------------------------
def _calls(prediction: Any) -> list[dict[str, Any]]:
    """The call list out of whatever the caller passed.

    `Quartz.complete()` returns a dict with `function_calls` in it, but a raw
    list of calls is also accepted so a suite can be scored against a JSON dump
    from somewhere else. None and an empty list both mean refusal.
    """
    if prediction is None:
        return []
    if isinstance(prediction, Mapping):
        if "function_calls" in prediction:
            prediction = prediction.get("function_calls") or []
        else:                       # a single call, passed on its own
            prediction = [prediction] if "name" in prediction else []
    out = []
    for call in prediction or []:
        if isinstance(call, Mapping):
            out.append({"name": str(call.get("name", "")),
                        "arguments": dict(call.get("arguments") or {})})
    return out


def _value(value: Any, *, strict: bool) -> Any:
    """A value in the form two answers can be compared in.

    Non-strict is the default and forgives exactly two things: the case of a
    string, and a number that arrived as its own digits in quotes. Both are
    artefacts of JSON typing rather than of understanding, and a model that
    wrote `"10"` where the schema said integer has not misread the request. It
    is still a schema error, and `schema_errors` counts it there.
    """
    if strict:
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = " ".join(value.split()).casefold()
        try:
            number = float(text)
        except ValueError:
            return text
        return number if math.isfinite(number) else text
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, list):
        return [_value(v, strict=strict) for v in value]
    if isinstance(value, Mapping):
        return {k: _value(v, strict=strict) for k, v in sorted(value.items())}
    return value


def _arguments(call: Mapping[str, Any], *, strict: bool) -> dict[str, Any]:
    return {str(k): _value(v, strict=strict)
            for k, v in (call.get("arguments") or {}).items()}


def tool_names(calls: Any) -> list[str]:
    """The names called, in order. Order is part of the answer: "look up Priya
    and text her" is two calls whose second one depends on the first."""
    return [c["name"] for c in _calls(calls)]


def exact_call(predicted: Any, gold: Any, *, strict: bool = False) -> bool:
    """Is the predicted call list exactly the expected one?

    Every name, every argument key and every argument value, in order. This is
    the headline number of the suite and the hard one: the model picks the right
    tool 95.8 percent of the time and gets all of the arguments right 63.6.
    """
    pred, want = _calls(predicted), _calls(gold)
    if len(pred) != len(want):
        return False
    return all(
        p["name"] == g["name"]
        and _arguments(p, strict=strict) == _arguments(g, strict=strict)
        for p, g in zip(pred, want, strict=True))


def argument_grounding(predicted: Any, gold: Any, *,
                       strict: bool = False) -> tuple[int, int]:
    """(arguments reproduced exactly, arguments expected).

    Micro-averaged over arguments rather than over errands, so a robot errand
    carrying a pose, an object and a destination weighs three times a timer
    carrying one integer. Only calls whose tool name is right are compared: an
    argument filled into the wrong tool is not partly correct, it is a different
    action.
    """
    pred, want = _calls(predicted), _calls(gold)
    hit = total = 0
    for i, g in enumerate(want):
        expected = _arguments(g, strict=strict)
        total += len(expected)
        if i >= len(pred) or pred[i]["name"] != g["name"]:
            continue
        got = _arguments(pred[i], strict=strict)
        hit += sum(1 for k, v in expected.items() if k in got and got[k] == v)
    return hit, total


# --- validity against the declared schemas ---------------------------------
_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,), "integer": (int,), "number": (int, float),
    "boolean": (bool,), "array": (list,), "object": (dict,),
}


def schema_errors(predicted: Any, tools: Sequence[Mapping[str, Any]]) -> list[str]:
    """Every reason a predicted call could not be executed as written.

    The five doors the post closes, in the order it closes them: an undeclared
    tool name, an undeclared argument key, a missing required argument, a value
    outside an enum, and a number outside its range. The first two are shut by
    the trie, the last three by the schema automaton, and a run with constrained
    decoding on should return an empty list here for every errand.
    """
    schemas = {str(t.get("name", "")): t for t in tools or []}
    problems: list[str] = []
    for call in _calls(predicted):
        name = call["name"]
        spec = schemas.get(name)
        if spec is None:
            problems.append(f"undeclared tool {name!r}")
            continue
        params = spec.get("parameters") or {}
        props = params.get("properties") or {}
        arguments = call["arguments"]
        for key in arguments:
            if key not in props:
                problems.append(f"{name}: undeclared argument {key!r}")
        for key in params.get("required") or []:
            if key not in arguments:
                problems.append(f"{name}: missing required {key!r}")
        for key, value in arguments.items():
            field = props.get(key)
            if not isinstance(field, Mapping):
                continue
            wanted = _JSON_TYPES.get(str(field.get("type", "")))
            if wanted and not isinstance(value, wanted):
                problems.append(f"{name}.{key}: {type(value).__name__} is not "
                                f"{field.get('type')}")
            if "enum" in field and value not in field["enum"]:
                problems.append(f"{name}.{key}: {value!r} outside its choices")
            if isinstance(value, int | float) and not isinstance(value, bool):
                lo, hi = field.get("minimum"), field.get("maximum")
                if (lo is not None and value < lo) or (hi is not None and value > hi):
                    problems.append(f"{name}.{key}: {value} outside [{lo}, {hi}]")
    return problems


# --- calibration -----------------------------------------------------------
def expected_calibration_error(conf: Any, correct: Any,
                               bins: int = DEFAULT_BINS) -> float:
    """How far a stated confidence is from how often it is right.

    Binned by *predicted confidence*, weighted by how many predictions land in
    each bin. The raw Gauge head scores 0.1174 here, so a stated confidence is
    off by roughly a ninth, and one scalar temperature fitted on two thousand
    held-out examples drops it to 0.0212.

    Bins are half-open, `(lo, hi]`, except the first, which is closed at zero so
    a head that returns exactly 0.0 still appears in its own diagram instead of
    silently leaving the average.
    """
    conf = np.asarray(conf, dtype=np.float64).ravel()
    correct = np.asarray(correct, dtype=np.float64).ravel()
    if conf.size == 0 or conf.size != correct.size:
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        m = (conf >= lo) & (conf <= hi) if i == 0 else (conf > lo) & (conf <= hi)
        if m.sum():
            ece += m.sum() / conf.size * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def reliability_bins(conf: Any, correct: Any,
                     bins: int = DEFAULT_BINS) -> list[dict[str, float]]:
    """The reliability diagram as data: one row per bin.

    Drawing this is what found the miscalibration in the first place. The head
    runs hot, and the bins where it is most sure are the ones that overpromise,
    which is invisible in a mean absolute error against the training target.
    """
    conf = np.asarray(conf, dtype=np.float64).ravel()
    correct = np.asarray(correct, dtype=np.float64).ravel()
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        if conf.size:
            m = (conf >= lo) & (conf <= hi) if i == 0 else (conf > lo) & (conf <= hi)
        else:
            m = np.zeros(0, dtype=bool)
        count = int(m.sum())
        rows.append({
            "lo": float(lo), "hi": float(hi), "n": count,
            "confidence": float(conf[m].mean()) if count else 0.0,
            "accuracy": float(correct[m].mean()) if count else 0.0,
            "gap": float(correct[m].mean() - conf[m].mean()) if count else 0.0,
        })
    return rows


def maximum_calibration_error(conf: Any, correct: Any,
                              bins: int = DEFAULT_BINS) -> float:
    """The worst bin, not the average one. A product lives in its worst bin."""
    return max((abs(row["gap"]) for row in reliability_bins(conf, correct, bins)
                if row["n"]), default=0.0)


def threshold_sweep(confidence: Any, correct: Any,
                    thresholds: Sequence[float] = DEFAULT_THRESHOLDS
                    ) -> list[dict[str, float]]:
    """What each escalation threshold costs and buys.

    For each threshold: the share of requests the device answers itself, how
    often those are exact, and how often the escalated ones would have been.
    The last column is the one that matters. Escalated calls are right far less
    often than the kept ones, which is the evidence that the confidence is
    ordering requests by difficulty rather than by nothing.
    """
    conf = np.asarray(confidence, dtype=np.float64).ravel()
    ok = np.asarray(correct, dtype=np.float64).ravel()
    rows = []
    for t in thresholds:
        handled = conf >= t
        rows.append({
            "threshold": float(t),
            "handled": float(handled.mean()) if conf.size else 0.0,
            "exact_call_handled": float(ok[handled].mean()) if handled.any() else 0.0,
            "exact_call_escalated": (float(ok[~handled].mean())
                                     if (~handled).any() else 0.0),
        })
    return rows


# --- the report ------------------------------------------------------------
def _share(hits: int, total: int) -> float:
    return hits / total if total else 0.0


def score(suite: Sequence[Mapping[str, Any]], predictions: Sequence[Any], *,
          bins: int = DEFAULT_BINS,
          thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
          strict: bool = False) -> dict[str, Any]:
    """Score one run of the suite.

    Args:
        suite: the errands, as `build_suite` returns them. Each needs `answers`
            and, if the breakdowns are wanted, `family`, `held_out` and
            `off_topic`.
        predictions: one per errand, in order. Either a `Quartz.complete()`
            response or a bare list of calls.
        bins: bins for the reliability diagram and the calibration error.
        thresholds: the escalation thresholds to sweep.
        strict: compare argument values byte for byte, including the case of a
            string and the difference between 10 and "10".

    Returns:
        The headline rates first, then the breakdowns: `by_family`, `held_out`
        against `seen`, and the per-thousand error counts that say which doors
        the decoder closed. `reliability` and `threshold_sweep` are present only
        when the predictions carried a confidence, since there is nothing to
        calibrate without one. `format_report` prints the readable version.

    Two rules make the numbers comparable across runs. Exactness is scored over
    the errands only, because an off-topic request has no call to be exact
    about; refusal is scored over the off-topic ones, because refusing a real
    request is a different failure and is reported as `over_refusal`.
    """
    if len(suite) != len(predictions):
        raise ValueError(
            f"{len(suite)} errands and {len(predictions)} predictions: score() "
            f"pairs them by position, so the two must be the same length")

    totals = {"errands": 0, "exact": 0, "selected": 0, "valid": 0, "refused": 0,
              "off_topic": 0, "over_refused": 0, "grounded": 0, "arguments": 0,
              "calls": 0}
    errors = {"wrong_tool_name": 0, "wrong_argument_key": 0, "missing_required": 0,
              "value_outside_choices": 0, "number_outside_range": 0,
              "wrong_type": 0, "malformed_json": 0}
    families: dict[str, list[int]] = {}
    slices: dict[str, list[int]] = {"held_out": [0, 0], "seen": [0, 0],
                                    "single_call": [0, 0], "multi_call": [0, 0]}
    confidence: list[float] = []
    correct: list[float] = []

    for errand, prediction in zip(suite, predictions, strict=True):
        gold = errand.get("answers") or []
        calls = _calls(prediction)
        tools = errand.get("tools") or []
        totals["calls"] += len(calls)
        for call in calls:
            # One call at a time, so validity is a rate over calls rather than
            # over errands: an errand that emitted nothing is not evidence that
            # the decoder works, and counting it as a pass would flatter it.
            problems = schema_errors([call], tools)
            totals["valid"] += 0 if problems else 1
            for problem in problems:
                if "undeclared tool" in problem:
                    errors["wrong_tool_name"] += 1
                elif "undeclared argument" in problem:
                    errors["wrong_argument_key"] += 1
                elif "missing required" in problem:
                    errors["missing_required"] += 1
                elif "outside its choices" in problem:
                    errors["value_outside_choices"] += 1
                elif "outside [" in problem:
                    errors["number_outside_range"] += 1
                else:
                    errors["wrong_type"] += 1
        if isinstance(prediction, Mapping) and prediction.get("error"):
            errors["malformed_json"] += 1
            # A turn that could not be read was still an attempted call, so it
            # joins the denominator; otherwise the rate that matters most here
            # would have no attempts to be a rate over.
            totals["calls"] += 0 if calls else 1

        if errand.get("off_topic"):
            totals["off_topic"] += 1
            totals["refused"] += 1 if not calls else 0
            continue

        totals["errands"] += 1
        hit = exact_call(calls, gold, strict=strict)
        totals["exact"] += 1 if hit else 0
        totals["selected"] += 1 if tool_names(calls) == tool_names(gold) else 0
        totals["over_refused"] += 1 if not calls else 0
        grounded, expected = argument_grounding(calls, gold, strict=strict)
        totals["grounded"] += grounded
        totals["arguments"] += expected

        bucket = families.setdefault(str(errand.get("family", "?")), [0, 0])
        bucket[0] += 1
        bucket[1] += 1 if hit else 0
        for key in (("held_out" if errand.get("held_out") else "seen"),
                    ("multi_call" if len(gold) > 1 else "single_call")):
            slices[key][0] += 1
            slices[key][1] += 1 if hit else 0

        if isinstance(prediction, Mapping) and prediction.get("confidence") is not None:
            confidence.append(float(prediction["confidence"]))
            correct.append(1.0 if hit else 0.0)

    per_1000 = 1000.0 / max(1, totals["calls"])
    report: dict[str, Any] = {
        "n": len(suite),
        "errands": totals["errands"],
        "exact_call": _share(totals["exact"], totals["errands"]),
        "tool_selection": _share(totals["selected"], totals["errands"]),
        "argument_grounding": _share(totals["grounded"], totals["arguments"]),
        "refusal": _share(totals["refused"], totals["off_topic"]),
        "schema_valid": (_share(totals["valid"], totals["calls"])
                         if totals["calls"] else 1.0),
        "over_refusal": _share(totals["over_refused"], totals["errands"]),
        "calls": totals["calls"],
        "ece": expected_calibration_error(confidence, correct, bins),
        "mce": maximum_calibration_error(confidence, correct, bins),
        "scored_confidences": len(confidence),
        "by_family": {k: _share(v[1], v[0]) for k, v in sorted(families.items())},
        "by_slice": {k: _share(v[1], v[0]) for k, v in slices.items()},
        "errors_per_1000_calls": {k: round(v * per_1000, 1) for k, v in errors.items()},
    }
    if confidence:
        # Only when there is a Gauge head to be calibrated. Ten empty bins and a
        # sweep of zeros are not a result, they are noise in the artifact.
        report["reliability"] = reliability_bins(confidence, correct, bins)
        report["threshold_sweep"] = threshold_sweep(confidence, correct, thresholds)
    return report


def format_report(report: Mapping[str, Any]) -> str:
    """The report as the post prints it: the four headline rates, then the
    per-family split, then the sweep."""
    lines = [
        f"  {'tool selection':<24}{report['tool_selection']:>7.1%}",
        f"  {'argument exact':<24}{report['exact_call']:>7.1%}",
        f"  {'refusal on off-topic':<24}{report['refusal']:>7.1%}",
        f"  {'schema valid':<24}{report['schema_valid']:>7.1%}",
    ]
    if report.get("by_family"):
        lines.append("")
        lines.append(f"  {'by family':<24}{'exact call':>12}")
        for family, value in sorted(report["by_family"].items(),
                                    key=lambda kv: -kv[1]):
            lines.append(f"  {family:<24}{value:>12.1%}")
    if report.get("scored_confidences"):
        lines.append("")
        lines.append(f"  {'ECE':<24}{report['ece']:>7.4f}   "
                     f"MCE {report['mce']:.4f}  ({report['scored_confidences']:,} "
                     f"scored)")
        lines.append(f"  {'threshold':<12}{'handled':>10}{'here':>10}{'escalated':>12}")
        for row in report.get("threshold_sweep", []):
            lines.append(f"  {row['threshold']:<12.2f}{row['handled']:>10.1%}"
                         f"{row['exact_call_handled']:>10.1%}"
                         f"{row['exact_call_escalated']:>12.1%}")
    return "\n".join(lines)


def save_report(report: Mapping[str, Any], path: str) -> str:
    """Write the report as JSON, beside the per-errand predictions."""
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return str(file)
