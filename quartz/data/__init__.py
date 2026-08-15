"""The data the model is made of: the corpus, and the teacher's homework.

    winnow  120 billion tokens over four sources, one fifth of them structured
            on purpose, filtered, mixed and packed into fixed rows that carry
            the segment ids the attention mask needs to close the seam.
    quarry  1.2 million labelled tool calls across 41,000 schemas, written by a
            235 billion parameter open model behind a local OpenAI-compatible
            endpoint, deduplicated on request-and-call together.

Neither module needs JAX, and both are imported eagerly here: building a corpus
or generating training data are CPU jobs, and they are often the only thing
running on a machine that has no accelerator in it at all. numpy, sentencepiece
and the standard library carry both.
"""
from quartz.data import quarry, winnow

__all__ = ["quarry", "winnow"]
