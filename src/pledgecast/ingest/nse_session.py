"""NSE HTTP session - PLAN.md sec.1.2, sec.10 ("Network failure mid-ingest").

Three corrections to sec.1.2, all measured live on 2026-08-13:

  1. **The homepage bootstrap no longer works.** sec.1.2 says to GET
     ``https://www.nseindia.com/`` first to set cookies. That now returns
     **403 (AkamaiGHost "Access Denied")** for non-browser TLS clients. The
     ``/api/*`` paths sit behind a different rule and still return 200 + cookies,
     so the bootstrap hits ``/api/marketStatus`` instead.

  2. **Never advertise Brotli.** Sending ``Accept-Encoding: gzip, deflate, br``
     returns bodies ``requests`` cannot decode - they arrive as binary garbage
     that looks like a corrupt XML file. Enforced by a validator in config.py.

  3. **``Referer`` is mandatory on ``nsearchives.nseindia.com``**, not just on
     the JSON APIs. Without it XBRL downloads 404.

Retry policy (sec.10): exponential backoff on transient failures, and a full
session re-bootstrap on 401/403 - NSE expires cookies rather than returning a
clean auth error.
"""

from __future__ import annotations

import hashlib
import random
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import requests
from requests.adapters import HTTPAdapter

from pledgecast.exceptions import DataIngestionError
from pledgecast.logging_config import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from config import Settings

logger = get_logger(__name__)

# Retried with backoff; anything else fails fast.
_TRANSIENT_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


class NSESession:
    """A cookie-bootstrapped ``requests.Session`` with backoff and refresh.

    Safe to share across the ingest thread pool: the underlying session is
    reused for concurrent GETs, and the only mutating operation
    (re-bootstrapping) is serialised behind a lock.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        if settings is None:
            from config import get_settings

            settings = get_settings()
        self.cfg = settings.ingest
        self._lock = threading.Lock()
        self._session: requests.Session | None = None
        self._bootstrapped_at: float = 0.0

    # ------------------------------------------------------------------ setup
    def _build_session(self) -> requests.Session:
        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": self.cfg.user_agent,
                "Accept": "*/*",
                "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
                # NO 'br' - see module docstring.
                "Accept-Encoding": self.cfg.accept_encoding,
                "Connection": "keep-alive",
            }
        )
        # One pooled connection per worker, or the pool thrashes at 4 threads.
        adapter = HTTPAdapter(
            pool_connections=self.cfg.max_workers,
            pool_maxsize=self.cfg.max_workers * 2,
            max_retries=0,  # retries are handled here, so we can refresh cookies
        )
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def bootstrap(self, *, force: bool = False) -> requests.Session:
        """Establish cookies. Idempotent; call freely."""
        with self._lock:
            if self._session is not None and not force:
                return self._session

            session = self._build_session()
            url = f"{self.cfg.base_url}{self.cfg.bootstrap_path}"
            try:
                response = session.get(url, timeout=self.cfg.timeout_seconds)
                response.raise_for_status()
            except requests.RequestException as exc:
                raise DataIngestionError(
                    f"NSE session bootstrap failed against {url}: {exc}"
                ) from exc

            self._session = session
            self._bootstrapped_at = time.monotonic()
            logger.info(
                "NSE session bootstrapped via %s (%d cookies)",
                self.cfg.bootstrap_path,
                len(session.cookies),
            )
            return session

    @property
    def session(self) -> requests.Session:
        return self._session if self._session is not None else self.bootstrap()

    def close(self) -> None:
        with self._lock:
            if self._session is not None:
                self._session.close()
                self._session = None

    def __enter__(self) -> NSESession:
        self.bootstrap()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # ------------------------------------------------------------- requesting
    def _absolute(self, path_or_url: str) -> str:
        if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
            return path_or_url
        return f"{self.cfg.base_url}{path_or_url}"

    def get(
        self,
        path_or_url: str,
        *,
        params: dict[str, Any] | None = None,
        referer: str | None = None,
        timeout: int | None = None,
    ) -> requests.Response:
        """GET with backoff, session refresh on 401/403, and a polite delay.

        ``referer`` may be a full URL or a path relative to the NSE base; it
        defaults to the corporate-filings page, which both the JSON APIs and the
        archives host require.
        """
        url = self._absolute(path_or_url)
        ref = referer or self.cfg.default_referer
        headers = {
            "Referer": self._absolute(ref),
            "X-Requested-With": "XMLHttpRequest",
        }
        last_error: str = "no attempt made"

        for attempt in range(self.cfg.max_retries + 1):
            if self.cfg.request_delay_seconds:
                time.sleep(self.cfg.request_delay_seconds)

            try:
                response = self.session.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=timeout or self.cfg.timeout_seconds,
                )
            except requests.RequestException as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "attempt %d/%d %s -> %s", attempt + 1, self.cfg.max_retries + 1, url, last_error
                )
                self._sleep_backoff(attempt)
                continue

            if response.status_code == 200:
                return response

            if response.status_code in self.cfg.session_refresh_status_codes:
                # NSE expires cookies rather than returning a clean auth error.
                last_error = f"HTTP {response.status_code} (cookie expiry)"
                logger.warning("session expired on %s; re-bootstrapping", url)
                self.bootstrap(force=True)
                self._sleep_backoff(attempt)
                continue

            if response.status_code in _TRANSIENT_STATUS:
                last_error = f"HTTP {response.status_code}"
                logger.warning("transient %s on %s", last_error, url)
                self._sleep_backoff(attempt)
                continue

            # 404 and friends are permanent - do not burn retries on them.
            raise DataIngestionError(f"GET {url} failed permanently: HTTP {response.status_code}")

        raise DataIngestionError(
            f"GET {url} failed after {self.cfg.max_retries + 1} attempts: {last_error}"
        )

    def _sleep_backoff(self, attempt: int) -> None:
        """Exponential backoff with jitter, so 4 workers do not retry in lockstep."""
        delay = (self.cfg.backoff_factor**attempt) + random.uniform(0, 0.4)  # noqa: S311
        time.sleep(delay)

    def get_json(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        referer: str | None = None,
    ) -> Any:
        """GET and decode JSON, raising ``DataIngestionError`` on a bad body."""
        response = self.get(path, params=params, referer=referer)
        try:
            return response.json()
        except ValueError as exc:
            snippet = response.text[:200].replace("\n", " ")
            raise DataIngestionError(
                f"expected JSON from {response.url} but got "
                f"{response.headers.get('Content-Type')}: {snippet!r}"
            ) from exc

    def download(
        self,
        url: str,
        dest: Path,
        *,
        referer: str | None = None,
        skip_existing: bool | None = None,
    ) -> tuple[Path, str, int, bool]:
        """Download to ``dest``. Returns ``(path, sha256, n_bytes, was_skipped)``.

        Resumable by design (sec.10): a file already on disk is hashed and
        skipped, so re-running the ingest after a network failure costs nothing.
        Raw files are never overwritten - ``data/raw/`` is immutable research
        data (sec.5.2).
        """
        skip = self.cfg.skip_existing_files if skip_existing is None else skip_existing

        if skip and dest.exists() and dest.stat().st_size > 0:
            payload = dest.read_bytes()
            return dest, hashlib.sha256(payload).hexdigest(), len(payload), True

        response = self.get(url, referer=referer)
        payload = response.content
        if not payload:
            raise DataIngestionError(f"empty body downloading {url}")

        dest.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temp name then rename, so an interrupted run never leaves a
        # truncated file that the resume logic would then trust.
        tmp = dest.with_suffix(dest.suffix + ".part")
        tmp.write_bytes(payload)
        tmp.replace(dest)

        return dest, hashlib.sha256(payload).hexdigest(), len(payload), False


def fetch_json_url(url: str, settings: Settings | None = None, **params: Any) -> Any:
    """Plain GET+JSON for non-NSE hosts (Yahoo), which need no cookie dance."""
    if settings is None:
        from config import get_settings

        settings = get_settings()

    try:
        response = requests.get(
            url,
            params=params or None,
            headers={
                "User-Agent": settings.ingest.user_agent,
                "Accept": "*/*",
                "Accept-Encoding": settings.ingest.accept_encoding,
            },
            timeout=settings.ingest.timeout_seconds,
        )
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError) as exc:
        raise DataIngestionError(f"GET {url} failed: {type(exc).__name__}: {exc}") from exc


__all__ = ["NSESession", "fetch_json_url"]
