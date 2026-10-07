"""Thin Anthropic client: image + prompt in, parsed JSON/text out, with stage timing."""

import base64
import io
import json
import logging
import os
import re
import threading
import time
from contextlib import contextmanager
from typing import Iterator

import anthropic
from dotenv import load_dotenv
from PIL import Image

from .config import API_KEY_ENV, CONFIG, REPO_ROOT, Config

log = logging.getLogger("vision_automation")


class MissingApiKeyError(RuntimeError):
    pass


class LLMError(RuntimeError):
    pass


_stage_lock = threading.Lock()
_stage_totals: dict[str, list[float]] = {}  # stage -> [summed seconds, call count]


def reset_stage_times() -> None:
    with _stage_lock:
        _stage_totals.clear()


def stage_times() -> dict[str, dict[str, float]]:
    """Per-stage totals since the last reset. Parallel calls add up, so a stage's sum can exceed wall time."""
    with _stage_lock:
        return {k: {"seconds": round(v[0], 3), "calls": int(v[1])} for k, v in _stage_totals.items()}


@contextmanager
def timed(stage: str) -> Iterator[None]:
    """Log wall-clock time for a pipeline stage (capture, popup, plan, ground, verify, ...) and add it to the totals."""
    start = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - start
        log.info("[timing] %-8s %.2fs", stage, dt)
        with _stage_lock:
            t = _stage_totals.setdefault(stage, [0.0, 0])
            t[0] += dt
            t[1] += 1


def make_client(cfg: Config = CONFIG) -> anthropic.Anthropic:
    load_dotenv(REPO_ROOT / ".env", override=False)  # real env vars win over .env
    key = os.environ.get(API_KEY_ENV)
    if not key:
        raise MissingApiKeyError(
            f"{API_KEY_ENV} is not set. Put it in the gitignored .env file "
            f"({API_KEY_ENV}=...) or set it in your environment. Never commit it."
        )
    return anthropic.Anthropic(
        api_key=key, timeout=cfg.request_timeout_s, max_retries=cfg.sdk_max_retries
    )


def image_block(img: Image.Image) -> dict:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    data = base64.standard_b64encode(buf.getvalue()).decode("ascii")
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}


class LLM:
    """One place that builds requests, so model/effort are always sent the same way."""

    def __init__(self, client: anthropic.Anthropic | None = None, cfg: Config = CONFIG):
        self.cfg = cfg
        self.client = client or make_client(cfg)
        self.calls = 0
        self.parse_failures: list[dict] = []  # model replies we could not use (stage, detail)
        self.call_log: list[dict] = []  # one entry per API call: seconds, retries, tokens, request id
        self._lock = threading.Lock()

    def note_parse_failure(self, stage: str, detail: str) -> None:
        with self._lock:
            self.parse_failures.append({"stage": stage, "detail": detail[:300]})
        log.warning("[%s] unusable model reply: %s", stage, detail[:200])

    def request_params(self, **overrides) -> dict:
        params = {
            "model": self.cfg.model,
            "max_tokens": self.cfg.max_tokens,
            "output_config": {"effort": self.cfg.effort},
        }
        params.update(overrides)
        return params

    def ask(self, stage: str, prompt: str, images: list[Image.Image] | None = None,
            system: str | None = None) -> str:
        content: list[dict] = [image_block(i) for i in images or []]
        content.append({"type": "text", "text": prompt})
        params = self.request_params(messages=[{"role": "user", "content": content}])
        if system:
            params["system"] = system
        with self._lock:
            self.calls += 1
        record: dict = {"stage": stage, "images": [f"{i.width}x{i.height}" for i in images or []],
                        "seconds": None, "retries": None, "in_tokens": None, "out_tokens": None,
                        "request_id": None, "status": "error"}
        t0 = time.perf_counter()
        try:
            with timed(stage):
                try:
                    raw = self.client.messages.with_raw_response.create(**params)
                    resp = raw.parse()
                except anthropic.APIStatusError as e:
                    record["status"] = f"http {e.status_code}"
                    raise LLMError(f"{stage}: API error {e.status_code}: {e.message}") from e
                except anthropic.APIConnectionError as e:
                    record["status"] = "connection error"
                    raise LLMError(f"{stage}: network error: {e}") from e
            record.update(retries=raw.retries_taken, request_id=raw.request_id, status=resp.stop_reason,
                          in_tokens=resp.usage.input_tokens, out_tokens=resp.usage.output_tokens)
        finally:
            record["seconds"] = round(time.perf_counter() - t0, 2)
            with self._lock:
                self.call_log.append(record)
            # Latency diagnostics: a slow call with retries>0 was SDK backoff (429/5xx/timeout), not the model.
            log.info("[%s] %.1fs retries=%s tokens in=%s out=%s status=%s request_id=%s", stage,
                     record["seconds"], record["retries"], record["in_tokens"], record["out_tokens"],
                     record["status"], record["request_id"])
        if resp.stop_reason == "refusal":
            raise LLMError(f"{stage}: model refused ({resp.stop_details})")
        return "".join(b.text for b in resp.content if b.type == "text")

    def ask_json(self, stage: str, prompt: str, images: list[Image.Image] | None = None,
                 system: str | None = None, retries: int = 1) -> dict:
        last: Exception | None = None
        for _ in range(retries + 1):
            text = self.ask(stage, prompt, images, system)
            try:
                return extract_json(text)
            except ValueError as e:
                last = e
                self.note_parse_failure(stage, f"unparsable JSON: {e}")
        raise LLMError(f"{stage}: could not parse JSON: {last}")


def extract_json(text: str) -> dict:
    """Tolerantly pull the LAST top-level JSON object out of a model reply.

    Last, because some prompts (e.g. the verifier's) ask for analysis first and the JSON after.
    """
    decoder = json.JSONDecoder()
    found: dict | None = None
    # Models occasionally emit a trailing comma before } or ] (invalid JSON); drop it.
    text = re.sub(r",(\s*[}\]])", r"\1", text)
    i = 0
    while True:
        i = text.find("{", i)
        if i == -1:
            break
        try:
            obj, end = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            i += 1
            continue
        if isinstance(obj, dict):
            found = obj
        i = end  # skip past it so nested objects aren't mistaken for top-level ones
    if found is None:
        raise ValueError(f"no JSON object found in: {text[:200]!r}")
    return found
