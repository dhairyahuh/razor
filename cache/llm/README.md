# The LLM completion cache

This directory is where `redteam.genai.cache` reads and writes model completions. One JSON
file per completion, named by a hash of the prompt, model, temperature and task, storing the
prompt **and** the answer so that a reviewer can check what was asked as well as what came
back.

## It is empty, and that is a deliberate state rather than an oversight

Populating it needs an API key. The alternative — writing the completions by hand and
committing them as though a model produced them — would make every claim built on top of
them false, and it is exactly the failure this cache exists to prevent.

So the repository ships the cache empty, and every component that would use it degrades
loudly rather than quietly:

| component | with a populated cache | with this empty cache |
| --- | --- | --- |
| `genai/payloads.py` | model-written prompt-injection payloads mixed into the agent corpus | payloads are template-composed; the run log prints `payload bank: empty` |
| `genai/transcript_bank.py` | model-written scam and genuine conversations | transcripts are template-composed; the run log prints `transcript bank: empty` |
| `genai/judge.py` | an LLM tries to tell generated rows from described real ones, and its accuracy is a fidelity score | the judge is skipped and the report says the rating is unavailable |
| `genai/narrate.py` | plain-language analyst narratives for the top alerts | no narratives section in the report |
| `genai/ideate.py` | the agent proposes new attack vectors in the library's own schema | `redteam ideate` reports that it cannot run |
| `loop/agents.py` | LLM red- and blue-team proposals inside the co-evolution loop | the loop searches its programmatic genome space only |

Nothing above changes a detection metric silently. The tabular pipeline, the guards and the
closed loop all run to completion either way; what changes is how much of the *text* in the
system was written by a model rather than by a template, and the report states which.

## Populating it

```bash
export REDTEAM_LLM_API_KEY=sk-...
# Optional: any OpenAI-compatible endpoint, including a local server.
export REDTEAM_LLM_BASE_URL=https://api.openai.com/v1

PYTHONPATH=src python -m redteam run --quick --refresh-llm
```

Misses go to the provider and are written back here. Commit the result and every subsequent
run on any machine reproduces it offline, byte for byte, with no key.

## Why the prompt is stored beside the completion

A cache of answers without the questions cannot be audited, and an unauditable cache of
model output sitting in a red-team repository is indistinguishable from a file of invented
text. Each entry records the task, model, temperature, system prompt, user prompt,
completion and retrieval timestamp.
