"""The import contract: `import quartz` must not cost a framework.

This is the test the whole lazy-import discipline exists for. JAX, Flax and
optax are optional extras, every module that needs them imports them inside the
function that calls them, and one convenience import at the top of any file
would silently undo that -- on the author's machine it would still work, and on
a phone it would fail at install time.

So the check runs in a *subprocess*. Asserting on `sys.modules` inside pytest
would pass or fail depending on whether some other test had already imported
JAX, which makes it a test of the collection order rather than of the package.
Three subprocesses, not one per module: each costs a fresh interpreter.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: What must not be resident after a plain import. jaxlib is named separately
#: because importing it without jax is just as expensive and just as wrong.
HEAVY = ("jax", "jaxlib", "flax", "optax")

#: Every module a device install has to be able to import, in dependency order
#: so the report names the first one that went heavy. `architecture` is on the
#: list on purpose: it defines Flax modules, and builds them on first attribute
#: access precisely so that importing it costs nothing.
#:
#: The last four are here because each one says so in its own docstring, and a
#: promise made only in prose is a promise that drifts: `run` claims a
#: checkpoint loads without JAX, Flax or ml_dtypes; `score` claims numpy only,
#: so a benchmark can be rescored on a machine that never trained anything;
#: `suite` claims it builds without a model at all; `winnow` claims nothing in
#: it imports JAX.
LIGHT = (
    "quartz",
    "quartz.model",
    "quartz.model.config",
    "quartz.model.scribe",
    "quartz.model.trellis",
    "quartz.model.grist",
    "quartz.model.ingot",
    "quartz.model.architecture",
    "quartz.agent",
    "quartz.agent.tools",
    "quartz.cli",
    "quartz.model.run",
    "quartz.errand.score",
    "quartz.errand.suite",
    "quartz.data.winnow",
)

_PROBE = """
import sys
first = {{}}
for name in {modules}:
    __import__(name)
    for heavy in {heavy}:
        if heavy in sys.modules and heavy not in first:
            first[heavy] = name
print(";".join(f"{{k}} came in with {{v}}" for k, v in first.items()))
"""


def _resident(*modules: str) -> list[str]:
    """Import each module in one fresh interpreter and report what went heavy."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), env.get("PYTHONPATH", "")])
    source = _PROBE.format(modules=modules, heavy=HEAVY)
    proc = subprocess.run([sys.executable, "-c", source], capture_output=True,
                          text=True, env=env, cwd=str(ROOT), check=False)
    assert proc.returncode == 0, f"importing {modules} failed:\n{proc.stderr}"
    return [line for line in proc.stdout.strip().split(";") if line]


def test_import_quartz_does_not_import_jax():
    """The contract, in one line: after `import quartz`, JAX is not resident."""
    assert _resident("quartz") == []


def test_every_light_module_stays_light():
    """The rest of the numpy-only surface, checked in one interpreter."""
    assert _resident(*LIGHT) == []


def test_the_probe_can_see_jax():
    """A test that can never fail is worse than no test.

    Where JAX is installed, importing it has to make the probe say so. Without
    this the two assertions above would pass just as happily on a machine where
    the subprocess was broken and the report always came back empty.
    """
    pytest.importorskip("jax")
    assert _resident("jax") != []


def test_public_surface():
    """What `from quartz import ...` is allowed to be."""
    import quartz

    assert quartz.__all__ == ["Field", "Quartz", "__version__", "extract", "tool"]
    for name in quartz.__all__:
        assert hasattr(quartz, name), f"{name} is exported and missing"


def test_version_is_a_release_string():
    import quartz

    parts = quartz.__version__.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)
