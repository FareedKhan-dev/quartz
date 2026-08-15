"""Errand: the benchmark we wrote, and the scorer that refuses to average it.

    suite  1,200 device errands over phone, wearable, smart home and robot,
           plus 180 off-topic requests that must be refused. Held-out tools,
           near-miss twins and controlled distractors, all deterministic in a
           seed so a reported number names a suite anybody can rebuild.
    score  tool selection, exact call, argument grounding, refusal, schema
           validity, the expected calibration error of the confidence, and the
           escalation threshold sweep that turns it into a product decision.

We wrote the test we then passed, so its difficulty is our choice and the
comparisons against other models are the trustworthy half. Building and scoring
need numpy and nothing else -- no JAX, no weights, no network -- because a
benchmark that can only be rerun on the machine that produced it is not a
benchmark.
"""
from quartz.errand import score, suite

__all__ = ["score", "suite"]
