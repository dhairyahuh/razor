"""The generative layer: cached LLM calls, and the agents built on them.

Four agents, all sharing one cache so that a run with no API key is byte-identical to the
committed one:

* :mod:`ideate` proposes new attack vectors, gated by the attack library's own validator.
* :mod:`payloads` writes indirect prompt-injection text, replacing template composition -
  which is what made the in-distribution injection AUC an uninformative 1.0000.
* :mod:`transcript_bank` writes the scam and legitimate conversations behind authorised push
  payments, for the same reason and with a sharper version of the same problem: a templated
  conversation is identifiable from its turn sequence without reading a word.
* :mod:`agents` is the red team that reads the per-vector recall table and proposes attack
  parameters, and the blue team that reads the weakest slices and proposes features. Both
  are auto-evaluated, and both publish their accept rate rather than their best result.

The design constraint throughout is that no agent is trusted. Every proposal passes through
a deterministic gate that already existed for other reasons - the library validator, the
injection classifier's held-out family split, a retrain-and-measure - and the rejections are
reported beside the acceptances.
"""

from .cache import CacheMiss, LLMCache, LLMCall, LLMResult  # noqa: F401
