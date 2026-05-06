"""Shared constants for the LLM subsystem."""

DEFAULT_MAX_TOKENS = 3000
DEFAULT_TEMPERATURE = 0.0
DEFAULT_RETRY_LIMIT = 2
DEFAULT_RETRY_DELAY = 120  # seconds
DEFAULT_SEED = 42
DEFAULT_LLM_TIMEOUT = 300  # seconds (5 minutes)
DEFAULT_LLM_PARALLEL_WORKERS = 30
NON_RETRYABLE_ERROR_MARKERS = (
    "contextwindowexceedederror",
    "context window exceeded",
)

DEEPSEEK_DEFAULT_FINGERPRINT = 'fp_eaab8d114b_prod0820_fp8_kvcache_new_kvcache'
