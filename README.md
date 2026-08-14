# Quartz

**A 45M parameter foundation tool-calling model for phones, wearables, smart home and robots.**

Every phone, watch and thermostat already contains a quartz crystal. It costs a few cents,
does one job forever, and does it on almost no power. This is a language model built to the
same brief: it reads a request, picks a tool, fills in the arguments, and does nothing else.

Text goes in. A JSON tool call comes out. There is no free-text mode, and a request no
declared tool can serve returns an empty call list, which is the refusal.

```python
import quartz

@quartz.tool
def set_lights(room: str, brightness: int):
    "Turn a room's lights on and set brightness."
    return {"room": room, "brightness": brightness}

agent = quartz.Quartz(tools=[set_lights])
print(agent.run("dim the kitchen to 10 percent")["results"])
# [{'room': 'kitchen', 'brightness': 10}]
```

## Why it is interesting

| | |
|---|---|
| Parameters | **45,211,383** total, 43,634,294 shipped |
| Weights on disk | **13.44 MB** at a mixed 2/3/4-bit scheme, 2.225 bits a weight |
| Whole artifact | **13.87 MB**, one file, tokenizer inline |
| Peak session memory | **27.6 MB**, and flat in conversation length |
| Errand exact-call | **63.6 %** |
| Phone-class latency | **89 ms** to a finished call, 41 mJ |
| Watch-class latency | **625 ms**, 96 mJ, about 3,000 calls per charge |

A 3B on-device model at four bits needs about 1.6 GB of weights. This is **one hundred and
twenty five times smaller** and scores six points higher on our suite.

## Install

```bash
pip install quartz-lm                 # inference only: numpy, sentencepiece, pyyaml
pip install "quartz-lm[train]"        # add JAX, Flax and optax to train
pip install "quartz-lm[gpu]"          # the CUDA 12 build
```

`import quartz` does **not** import JAX. There is a test that asserts it, so the inference
path stays light enough for a device or a CI job.

## Quickstart

**Call a tool.** Decorate a function. The signature gives the argument types, the docstring
is the description, and `run()` completes the loop.

```python
import quartz
from typing import Literal, Annotated

@quartz.tool
def set_thermostat(temperature: int, mode: Literal["heat", "cool", "auto"] = "auto"):
    """Set the thermostat.

    Args:
        temperature: target temperature in Celsius
    """
    return {"temperature": temperature, "mode": mode}

@quartz.tool
def send_money(amount: Annotated[float, quartz.Field(gt=0, le=10000)], to: str):
    "Send money to a handle."
    return {"sent": amount, "to": to}

agent = quartz.Quartz(tools=[set_thermostat, send_money])
agent.run("make it 21 and cool the room")
```

**Extract structured data.** Extraction is not a separate mode. It is tool calling with one
tool, so the same guarantees apply.

```python
from pydantic import BaseModel

class Invoice(BaseModel):
    vendor: str
    total: float
    due_date: str

invoice = quartz.extract("Invoice from Acme Corp, $1,200.00, due 2026-09-01", Invoice)
print(invoice.vendor, invoice.total)      # Acme Corp 1200.0
```

**Gate on confidence.** Every response carries a calibrated score. Act above your threshold,
escalate below it.

```python
r = agent.complete("dim the living room")
calls = r.get("function_calls") or []
if calls and r["confidence"] >= 0.6:
    execute(calls[0])
else:
    escalate()
```

## The architecture

One block, written once and scanned twenty seven times. Four residual lanes, mixed by a
doubly stochastic matrix, so the stream cannot grow from the mixing.

- **Loom** reads a weighted sum of four residual lanes, runs the block on the merged
  stream, and writes only the delta back through a Sinkhorn-normalised transport matrix.
  Because the transport is doubly stochastic, total residual mass is conserved.
- **Porthole** is grouped-query attention, eight query heads over four key/value heads, with
  QK-norm, a 256-token sliding window, and the declared tool schemas pinned as sinks so they
  can never fall out of the back of the window.
- **Spin** replaces the feed-forward network. Two fixed Walsh transforms and three learned
  diagonals, so a layer's MLP holds **1,536 parameters instead of 3,145,728**.
- **Imprint** is a hashed n-gram key/value memory at two layers, four independent tables of
  8,192 rows. Nine and a half million parameters that do no arithmetic at all.
- **Trellis** compiles the declared schemas into a byte-level constraint. A wrong tool name
  or a wrong argument key is not unlikely, it is unreachable.
- **Grist** quantises by rotating each group of 128 weights onto the unit sphere with a
  Walsh transform, keeping its length in one half-precision number, and storing each
  direction as an index into a codebook fitted offline to a Gaussian. No calibration data.

## Training from scratch

Four stages, about thirty nine hours on four H100s.

```bash
quartz train-tokenizer --corpus data/mixed.txt --vocab 8192
quartz winnow          --out data/packed --tokens 120e9
quartz pretrain        --config configs/base.yaml
quartz quarry          --schemas data/schemas.json --n 1200000
quartz sft             --data data/tools.jsonl --epochs 3
quartz qapt            --tokens 1.8e9 --bits-map "default=2,loom.phi=4,attn.out_proj=3"
quartz heads           --stage0 2500 --gauge --dowser
quartz build           ckpt/quartz_heads.pkl --out quartz.ingot
quartz errand          --weights quartz.ingot
```

Every stage writes a format-v2 checkpoint that records which stage produced it, and the
exporter refuses to ship a post-trained checkpoint that has not been stamped with the
quantisation scheme it was trained for.

## Layout

```
quartz/
├── model/
│   ├── config.py         the one source of geometry, everything derives from it
│   ├── scribe.py         tokenizer, the pre-tokenisation rule, the render template
│   ├── architecture.py   Loom, Spin, Porthole, Imprint, Dowser, Gauge
│   ├── decode.py         KV-cached decoder, sliding window, pinned sinks
│   ├── trellis.py        the constrained decoder
│   ├── grist.py          the 2-bit quantiser
│   ├── ingot.py          the single-file container
│   ├── graft.py          LoRA fine-tuning
│   └── run.py            checkpoints and reference inference
├── train/                pretrain, sft, qapt, heads, and Muon
├── data/                 winnow the corpus, quarry the teacher
├── errand/               1,200 device errands and the scorer
└── agent/                python callables into JSON tool schemas
```

## Tests

```bash
pip install -e ".[dev]"
pytest -m "not slow and not needs_jax"    # numpy only, seconds
pytest                                     # everything, needs the train extra
```

## What this does not do

- **The window is 256 tokens.** Nothing here handles a conversation that must recall
  something from twenty turns ago. Tool schemas are pinned. Nothing else is.
- **Tool retrieval is a ceiling.** With a 41,000-tool catalogue the right tool is in the
  rendered top five 89.4 % of the time, and a tool that is not rendered is unreachable
  rather than merely unlikely.
- **Errand is our own benchmark.** We wrote the test we then passed. Treat the comparisons
  against other models as the more trustworthy half.
- **Fine-tuning does not update the confidence head**, so a tuned model reports `confidence`
  as `None` rather than a confidently wrong number.

## Licence

MIT. See [LICENSE](LICENSE).

## Citation

```bibtex
@software{quartz2026,
  title  = {Quartz: a 45M parameter foundation tool-calling model for tiny devices},
  author = {Khan, Fareed},
  year   = {2026},
  url    = {https://github.com/FareedKhan-dev/quartz}
}
```
