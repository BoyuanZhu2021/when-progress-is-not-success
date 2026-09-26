"""Reproducible Hugging Face model identities for the active H1 tool-use run.

The repository names describe the experimental roles; immutable revisions ensure that cache
preflight, serving, training, and evaluation load identical weights.  The legacy H20/vLLM profile
keeps its short served-model alias, while Transformers Serve on the V100 profile uses the exact
``model@revision`` identity.  These names are intentionally separate and never substituted.
"""

ATTACKER_MODEL = "Qwen/Qwen3.5-4B"
ATTACKER_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
ATTACKER_VLLM_SERVED_NAME = "qwen3.5-4b-bnb4"

VICTIM_HF_MODEL = "Qwen/Qwen3.5-9B"
VICTIM_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
VICTIM_H20_SERVED_NAME = "qwen3.5-9b"
VICTIM_V100_SERVED_NAME = f"{VICTIM_HF_MODEL}@{VICTIM_REVISION}"

# Compatibility for the preserved H20 Gate/proof code.  New V100 code must import the explicit
# VICTIM_V100_SERVED_NAME constant instead of this legacy alias.
VICTIM_SERVED_NAME = VICTIM_H20_SERVED_NAME

import os

REMOTE_HF_HOME = os.environ.get("HF_HOME", str(os.path.expanduser("~/.cache/huggingface")))

# The data, tool schemas, and ReAct prompt template are deployed from this exact clean
# InjecAgent checkout.  Formal runtime verification checks this provenance field in addition to
# re-hashing every deployed byte.
INJECAGENT_COMMIT = "f19c9f2c79a41046eb13c03c51a24c567a8ffa07"
