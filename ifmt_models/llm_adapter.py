from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    enabled: bool
    base_url: str = ""
    reason: str = ""


@dataclass(frozen=True)
class LLMGeneration:
    text: str
    metrics: dict[str, Any]


class LLMAdapter:
    """Small adapter for optional real LLM calls.

    The evaluation pipeline never fabricates a direct-LLM result. A caller must
    explicitly enable LLM execution with environment configuration.
    """

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or load_llm_config()

    @property
    def is_available(self) -> bool:
        return self.config.enabled

    def generate(self, prompt: str) -> str:
        return self.generate_with_metrics(prompt).text

    def generate_with_metrics(self, prompt: str) -> LLMGeneration:
        if not self.config.enabled:
            raise RuntimeError(self.config.reason or "LLM execution is not configured.")
        if self.config.provider not in {"openai", "ollama"}:
            raise RuntimeError(f"Unsupported LLM provider: {self.config.provider}")

        retries = _optional_nonnegative_int_env(
            "VIRTUAL_REPORTER_LLM_RETRIES", default=2
        )
        for attempt in range(retries + 1):
            try:
                if self.config.provider == "openai":
                    return _call_openai_responses_api(prompt, self.config.model)
                return _call_ollama_chat_api(prompt, self.config)
            except (
                urllib.error.HTTPError,
                urllib.error.URLError,
                TimeoutError,
            ) as exc:
                if attempt == retries:
                    raise RuntimeError(
                        f"LLM request failed after {retries + 1} attempts: {exc}"
                    ) from exc
                time.sleep(min(2**attempt, 4))
        raise RuntimeError("LLM retry loop exited unexpectedly")


def load_llm_config() -> LLMConfig:
    provider = os.environ.get("VIRTUAL_REPORTER_LLM_PROVIDER", "").strip().lower()
    model = os.environ.get("VIRTUAL_REPORTER_LLM_MODEL", "").strip()
    base_url = os.environ.get("VIRTUAL_REPORTER_LLM_BASE_URL", "").strip()
    run_llm = os.environ.get("VIRTUAL_REPORTER_RUN_LLM", "").strip().lower() in {"1", "true", "yes", "sim"}

    if not run_llm:
        return LLMConfig(
            provider=provider or "none",
            model=model,
            enabled=False,
            reason="Set VIRTUAL_REPORTER_RUN_LLM=1 and provider/model/API credentials to run a real Direct LLM baseline.",
        )
    if provider not in {"openai", "ollama"}:
        return LLMConfig(
            provider=provider or "none",
            model=model,
            enabled=False,
            base_url=base_url,
            reason="Only provider=openai and provider=ollama are implemented for real LLM execution in this adapter.",
        )
    if not model:
        return LLMConfig(
            provider=provider,
            model=model,
            enabled=False,
            base_url=base_url,
            reason="VIRTUAL_REPORTER_LLM_MODEL is required for real LLM execution.",
        )
    if provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        return LLMConfig(
            provider=provider,
            model=model,
            enabled=False,
            base_url=base_url,
            reason="OPENAI_API_KEY is required for provider=openai.",
        )
    if provider == "ollama" and not base_url:
        base_url = "http://localhost:11434"
    return LLMConfig(provider=provider, model=model, enabled=True, base_url=base_url)


def _call_openai_responses_api(prompt: str, model: str) -> LLMGeneration:
    api_key = os.environ["OPENAI_API_KEY"]
    payload: dict[str, Any] = {
        "model": model,
        "input": prompt,
        "temperature": 0,
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        data = json.loads(response.read().decode("utf-8"))

    text_parts: list[str] = []
    for item in data.get("output", []) or []:
        for content in item.get("content", []) or []:
            if content.get("type") in {"output_text", "text"} and content.get("text"):
                text_parts.append(str(content["text"]))
    if text_parts:
        text = "\n".join(text_parts).strip()
    elif data.get("output_text"):
        text = str(data["output_text"]).strip()
    else:
        text = json.dumps(data, ensure_ascii=False)
    return LLMGeneration(
        text=text,
        metrics={
            "provider": "openai",
            "model": model,
            "usage": data.get("usage") or {},
            "response_id": data.get("id", ""),
        },
    )


def _ollama_payload(prompt: str, config: LLMConfig) -> dict[str, Any]:
    options: dict[str, Any] = {"temperature": 0}
    num_gpu = _optional_int_env("VIRTUAL_REPORTER_OLLAMA_NUM_GPU")
    if num_gpu is not None:
        options["num_gpu"] = num_gpu
    num_ctx = _optional_int_env("VIRTUAL_REPORTER_OLLAMA_NUM_CTX")
    if num_ctx is not None:
        options["num_ctx"] = num_ctx
    num_thread = _optional_int_env("VIRTUAL_REPORTER_OLLAMA_NUM_THREAD")
    if num_thread is not None:
        options["num_thread"] = num_thread
    num_predict = _optional_int_env("VIRTUAL_REPORTER_OLLAMA_NUM_PREDICT")
    if num_predict is not None:
        options["num_predict"] = num_predict

    payload: dict[str, Any] = {
        "model": config.model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": options,
        "keep_alive": -1,
    }
    response_format = os.environ.get("VIRTUAL_REPORTER_OLLAMA_FORMAT", "").strip()
    if response_format:
        payload["format"] = response_format
    think = _optional_bool_env("VIRTUAL_REPORTER_OLLAMA_THINK")
    if think is not None:
        payload["think"] = think
    return payload


def _call_ollama_chat_api(prompt: str, config: LLMConfig) -> LLMGeneration:
    endpoint = f"{config.base_url.rstrip('/')}/api/chat"
    payload = _ollama_payload(prompt, config)
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        data = json.loads(response.read().decode("utf-8"))

    return _parse_ollama_response(data)


def _parse_ollama_response(data: dict[str, Any]) -> LLMGeneration:

    message = data.get("message") if isinstance(data, dict) else None
    if isinstance(message, dict) and message.get("content"):
        text = str(message["content"]).strip()
    elif isinstance(data, dict) and data.get("response"):
        text = str(data["response"]).strip()
    else:
        text = json.dumps(data, ensure_ascii=False)
    return LLMGeneration(
        text=text,
        metrics={
            "provider": "ollama",
            "model": str(data.get("model", "")),
            "done": bool(data.get("done", False)),
            "done_reason": str(data.get("done_reason", "")),
            "total_duration_ns": int(data.get("total_duration") or 0),
            "load_duration_ns": int(data.get("load_duration") or 0),
            "prompt_eval_count": int(data.get("prompt_eval_count") or 0),
            "prompt_eval_duration_ns": int(data.get("prompt_eval_duration") or 0),
            "eval_count": int(data.get("eval_count") or 0),
            "eval_duration_ns": int(data.get("eval_duration") or 0),
        },
    )


def _optional_int_env(name: str) -> int | None:
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {value!r}.") from exc


def _optional_nonnegative_int_env(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {value!r}.") from exc
    if parsed < 0:
        raise RuntimeError(f"{name} must be non-negative, got {parsed}.")
    return parsed


def _optional_bool_env(name: str) -> bool | None:
    value = os.environ.get(name, "").strip().lower()
    if not value:
        return None
    if value in {"1", "true", "yes", "sim"}:
        return True
    if value in {"0", "false", "no", "nao"}:
        return False
    raise RuntimeError(f"{name} must be a boolean-like value, got {value!r}.")
