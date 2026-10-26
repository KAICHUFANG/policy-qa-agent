"""调用大模型的唯一出口。

为什么必须有这一层：
    调用大模型 API 的坑（超时、限流 429、返回空、token 统计）如果不集中处理，
    就会散落在每个业务函数里，各写一遍、各漏一遍。面试里这也是常被问的点：
    「你的模型调用失败了怎么办？」

这里统一做四件事：
    1. 超时     —— 不让一个卡住的请求拖死整个服务
    2. 重试     —— 只重试「可能自己好」的错误（超时、429、5xx）
    3. 用量统计 —— 每次调用都记 prompt/completion tokens，成本能算清
    4. 错误归档 —— 抛出带原文的异常，排障时不用猜
"""

import time
from typing import Any, Dict

from openai import (APIConnectionError, APIStatusError, APITimeoutError,
                    OpenAI, RateLimitError)

import config

_client = OpenAI(
    api_key=config.API_KEY,
    base_url=config.BASE_URL,
    timeout=config.TIMEOUT,
    max_retries=0,          # 重试自己写，才能把「重试了几次」记进结果里
)


class LLMError(RuntimeError):
    """模型调用失败。带可读原文，不要只抛一个空异常。"""


def _retryable(err: Exception) -> bool:
    return isinstance(err, (APITimeoutError, APIConnectionError, RateLimitError)) or (
        isinstance(err, APIStatusError) and 500 <= err.status_code < 600
    )


def chat(system: str, user: str, temperature: float = None) -> Dict[str, Any]:
    """一次问答调用。返回 {text, usage, attempts, latency_ms}。"""
    temperature = config.TEMPERATURE if temperature is None else temperature
    last: Exception | None = None
    started = time.perf_counter()

    for attempt in range(1, config.MAX_RETRIES + 2):     # 首次 + N 次重试
        try:
            resp = _client.chat.completions.create(
                model=config.MODEL,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                temperature=temperature,
            )
            text = (resp.choices[0].message.content or "").strip()
            usage = resp.usage
            return {
                "text": text,
                "usage": {
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(usage, "completion_tokens", 0),
                    "total_tokens": getattr(usage, "total_tokens", 0),
                },
                "attempts": attempt,
                "latency_ms": int((time.perf_counter() - started) * 1000),
            }
        except Exception as err:                        # noqa: BLE001
            last = err
            if attempt > config.MAX_RETRIES or not _retryable(err):
                break
            time.sleep(1.5 * attempt)                   # 退避：1.5s、3s

    raise LLMError(f"调用模型失败（{type(last).__name__}）：{last}")
