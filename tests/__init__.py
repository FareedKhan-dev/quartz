"""The Quartz test suite.

Two suites in one directory, separated by marker rather than by folder::

    pytest -m "not slow and not needs_jax"   the fast suite, numpy only
    pytest -m needs_jax                      the model and the training stages

The fast suite is the deployment contract. The tokenizer rule, the constrained
decoder, the quantiser, the container and the schema builder all run on numpy
alone, because that is everything a device install has, and `test_import`
asserts that importing the package does not drag JAX in behind them.
"""
