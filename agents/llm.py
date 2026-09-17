"""
LLM Client Factory for Multi-Agent RAG.

Initializes the unified Groq model (openai/gpt-oss-20b) with strict audit logging,
zero model substitution, and exponential backoff retry on the same model.
"""

import os
import sys
import time
import asyncio
import re
from typing import Optional, Any
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from langchain_groq import ChatGroq
except ImportError:
    ChatGroq = None

from config import GROQ_MODEL, GROQ_API_KEY

# Single standardized model across the entire agent workflow
MODEL_NAME = GROQ_MODEL

_GLOBAL_CALL_COUNT = 0
_MODELS_USED = set()


class AuditedChatGroq(ChatGroq):
    """
    Production Audited wrapper for ChatGroq that enforces:
    1. Zero model substitution: asserts model_name == MODEL_NAME (openai/gpt-oss-20b).
    2. Logging of model name and invocation count at the start of every call.
    3. Automatic retry with exponential backoff on the SAME model on rate limits or transient errors.
    4. Re-raises exceptions if retries are exhausted (never falls back to placeholder or another model).
    """

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        global _GLOBAL_CALL_COUNT, _MODELS_USED
        _GLOBAL_CALL_COUNT += 1
        _MODELS_USED.add(self.model_name)
        call_id = _GLOBAL_CALL_COUNT

        if self.model_name != MODEL_NAME:
            raise ValueError(f"CRITICAL: Model mismatch detected! Expected '{MODEL_NAME}', got '{self.model_name}'")

        print(f"[LLM INVOCATION] Model: {self.model_name} | Call #{call_id} (temp={self.temperature}, max_tokens={self.max_tokens})")

        max_retries = 8
        base_delays = [5, 10, 20, 30, 45, 60, 90, 120]

        for attempt in range(max_retries):
            try:
                return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
            except Exception as e:
                err_str = str(e)
                is_rate_limit = any(k in err_str.lower() for k in ["rate_limit", "429", "tpd", "tpm", "rpm", "too many requests", "resource_exhausted"])
                is_transient = any(k in err_str.lower() for k in ["timeout", "connection", "503", "502", "500", "overloaded"])
                if (is_rate_limit or is_transient) and attempt < max_retries - 1:
                    wait_time = base_delays[min(attempt, len(base_delays)-1)]
                    # Extract exact wait time if Groq provides it (e.g., 'Please try again in 5m32.64s')
                    match = re.search(r'try again in (?:(\d+)m)?(\d+(?:\.\d+)?s?)', err_str)
                    if match:
                        mins = float(match.group(1)) if match.group(1) else 0.0
                        sec_str = match.group(2).rstrip('s') if match.group(2) else "0"
                        wait_time = int(mins * 60 + float(sec_str)) + 3
                    print(f"  [!] Rate limit/Transient on {self.model_name}: {err_str[:110]}...")
                    print(f"  [!] Waiting {wait_time}s before retry (attempt {attempt+1}/{max_retries}) on SAME model ({self.model_name})...")
                    time.sleep(wait_time)
                else:
                    print(f"  [!] LLM Call #{call_id} failed on {self.model_name}: {err_str[:160]}")
                    raise e

        raise RuntimeError(f"Exhausted {max_retries} retries on {self.model_name} without success.")

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        global _GLOBAL_CALL_COUNT, _MODELS_USED
        _GLOBAL_CALL_COUNT += 1
        _MODELS_USED.add(self.model_name)
        call_id = _GLOBAL_CALL_COUNT

        if self.model_name != MODEL_NAME:
            raise ValueError(f"CRITICAL: Model mismatch detected! Expected '{MODEL_NAME}', got '{self.model_name}'")

        print(f"[LLM INVOCATION ASYNC] Model: {self.model_name} | Call #{call_id} (temp={self.temperature}, max_tokens={self.max_tokens})")

        max_retries = 5
        base_delays = [5, 10, 20, 30, 45]

        for attempt in range(max_retries):
            try:
                return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
            except Exception as e:
                err_str = str(e)
                is_rate_limit = any(k in err_str.lower() for k in ["rate_limit", "429", "tpm", "rpm", "too many requests", "resource_exhausted"])
                is_transient = any(k in err_str.lower() for k in ["timeout", "connection", "503", "502", "500", "overloaded"])
                if (is_rate_limit or is_transient) and attempt < max_retries - 1:
                    wait_time = base_delays[attempt]
                    print(f"  [!] Rate limit/Transient error on {self.model_name}: {err_str[:120]}...")
                    print(f"  [!] Retrying in {wait_time}s (attempt {attempt+1}/{max_retries}) on SAME model ({self.model_name})...")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"  [!] Async LLM Call #{call_id} failed on {self.model_name}: {err_str[:160]}")
                    raise e

        raise RuntimeError(f"Exhausted {max_retries} retries on {self.model_name} without success.")


def get_audit_summary() -> dict:
    """Returns invocation statistics to prove zero model switches."""
    return {
        "total_calls": _GLOBAL_CALL_COUNT,
        "models_used": list(_MODELS_USED),
        "zero_model_switches": len(_MODELS_USED) <= 1 and all(m == MODEL_NAME for m in _MODELS_USED)
    }


def get_llm(temperature: float = 0.1, max_tokens: int = 2048) -> Optional[Any]:
    """
    Returns an initialized AuditedChatGroq model instance.
    """
    api_key = GROQ_API_KEY
    if not api_key or api_key == "your_groq_api_key_here" or ChatGroq is None:
        if os.getenv("STRICT_BENCHMARK_MODE", "0") == "1":
            raise RuntimeError("CRITICAL: GROQ_API_KEY is missing in STRICT_BENCHMARK_MODE! Cannot run benchmark.")
        return None

    try:
        return AuditedChatGroq(
            groq_api_key=api_key,
            model_name=MODEL_NAME,
            temperature=temperature,
            max_tokens=max_tokens
        )
    except Exception as e:
        if os.getenv("STRICT_BENCHMARK_MODE", "0") == "1":
            raise e
        print(f"[!] Warning: ChatGroq initialization failed ({e}). Using dry-run mode.")
        return None

