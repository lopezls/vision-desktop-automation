"""Fetch the posts from JSONPlaceholder, with retries, and format them for typing."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable

import requests

from .config import POSTS_URL

log = logging.getLogger("vision_automation")

ATTEMPTS = 3
BACKOFF_S = (1.0, 2.0)  # waits before attempts 2 and 3
TIMEOUT_S = 10.0


class PostsError(RuntimeError):
    pass


@dataclass(frozen=True)
class Post:
    id: int
    title: str
    body: str

    @property
    def filename(self) -> str:
        return f"post_{self.id}.txt"

    @property
    def content(self) -> str:
        """The exact text typed into Notepad: "Title: {title}\\n\\n{body}"."""
        return f"Title: {self.title}\n\n{self.body}"


def untypeable_chars(text: str) -> list[str]:
    """Characters pyautogui cannot type reliably (anything outside printable ASCII except newline)."""
    return sorted({c for c in text if c != "\n" and not (c.isascii() and c.isprintable())})


def _parse(data: object, count: int) -> list[Post]:
    if not isinstance(data, list):
        raise PostsError(f"expected a JSON list of posts, got {type(data).__name__}")
    posts = []
    for item in data[:count]:
        if not (isinstance(item, dict) and isinstance(item.get("id"), int)
                and isinstance(item.get("title"), str) and isinstance(item.get("body"), str)):
            raise PostsError(f"malformed post: {str(item)[:120]!r}")
        posts.append(Post(item["id"], item["title"], item["body"]))
    if len(posts) < count:
        raise PostsError(f"API returned only {len(posts)} posts, wanted {count}")
    return posts


def fetch_posts(count: int, url: str = POSTS_URL, get: Callable[..., requests.Response] = requests.get,
                sleep: Callable[[float], None] = time.sleep) -> list[Post]:
    """First `count` posts, in API order. Retries connection errors, timeouts and 5xx; fails clearly otherwise."""
    last: Exception | None = None
    for attempt in range(1, ATTEMPTS + 1):
        try:
            resp = get(url, timeout=TIMEOUT_S)
            if resp.status_code >= 500:
                raise requests.HTTPError(f"HTTP {resp.status_code}")
            resp.raise_for_status()
            posts = _parse(resp.json(), count)
            break
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as e:
            last = e
            log.warning("[posts] attempt %d/%d failed: %s", attempt, ATTEMPTS, e)
            if attempt < ATTEMPTS:
                sleep(BACKOFF_S[attempt - 1])
        except ValueError as e:  # includes bad JSON (requests' JSONDecodeError)
            raise PostsError(f"could not read the posts response: {e}") from e
    else:
        raise PostsError(f"could not fetch posts from {url} after {ATTEMPTS} attempts: {last}")

    for p in posts:
        bad = untypeable_chars(p.content)
        if bad:
            raise PostsError(f"post {p.id} contains characters that cannot be typed reliably: {bad!r}")
    return posts
