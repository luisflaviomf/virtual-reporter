from __future__ import annotations

import time
import warnings
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import urlparse, urlunparse

import requests
from requests import Response
from urllib3.exceptions import InsecureRequestWarning

from .config import DEFAULT_HEADERS, LEGACY_IP_HOST, OLD_SUBDOMAIN_MAIN_HOST, OLD_SUBDOMAIN_SUFFIX


RETRY_STATUS_CODES = {429, 500, 502, 503, 504}


@dataclass
class FetchAttempt:
    request_url: str
    final_url: str | None
    status_code: int | None
    success: bool
    redirected: bool
    used_ip_access: bool
    ssl_verify_disabled: bool
    verify: bool
    elapsed_seconds: float
    retry_count: int
    html: str
    error_type: str | None = None
    error_message: str | None = None

    @property
    def html_length(self) -> int:
        return len(self.html or "")

    def to_dict(self, include_html: bool = False) -> dict[str, Any]:
        data = asdict(self)
        data["html_length"] = self.html_length
        if not include_html:
            data.pop("html", None)
        return data


@dataclass
class FetchResult:
    requested_url: str
    attempts: list[FetchAttempt]
    fallback_used: bool = False

    @property
    def selected(self) -> FetchAttempt:
        for attempt in self.attempts:
            if attempt.success:
                return attempt
        return self.attempts[-1]

    def to_dict(self, include_html: bool = False) -> dict[str, Any]:
        return {
            "requested_url": self.requested_url,
            "fallback_used": self.fallback_used,
            "selected": self.selected.to_dict(include_html=include_html),
            "attempts": [
                attempt.to_dict(include_html=include_html) for attempt in self.attempts
            ],
        }


class HttpClient:
    def __init__(
        self,
        timeout: float = 30.0,
        delay_seconds: float = 0.5,
        max_retries: int = 2,
        backoff_factor: float = 0.75,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.timeout = timeout
        self.delay_seconds = delay_seconds
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.headers = dict(DEFAULT_HEADERS)
        if headers:
            self.headers.update(headers)
        self.session = requests.Session()
        self._last_request_at = 0.0

    def fetch_url(self, url: str) -> FetchResult:
        attempts: list[FetchAttempt] = []
        parsed = urlparse(url)
        is_legacy_ip = parsed.hostname == LEGACY_IP_HOST
        is_legacy_ip_https = is_legacy_ip and parsed.scheme.lower() == "https"
        is_old_subdomain = _is_old_subdomain_host(parsed.hostname or "")

        attempts.append(
            self._request_with_retries(
                url=url,
                verify=not (is_legacy_ip or is_old_subdomain),
                ssl_verify_disabled=is_legacy_ip or is_old_subdomain,
            )
        )

        fallback_used = False
        if is_legacy_ip_https and not attempts[-1].success:
            fallback_used = True
            fallback_url = urlunparse(parsed._replace(scheme="http"))
            attempts.append(
                self._request_with_retries(
                    url=fallback_url,
                    verify=False,
                    ssl_verify_disabled=True,
                )
            )

        return FetchResult(
            requested_url=url,
            attempts=attempts,
            fallback_used=fallback_used,
        )

    def _request_with_retries(
        self,
        url: str,
        verify: bool,
        ssl_verify_disabled: bool,
    ) -> FetchAttempt:
        last_attempt: FetchAttempt | None = None

        for retry_count in range(self.max_retries + 1):
            self._respect_delay()
            started = time.perf_counter()
            try:
                response = self._get(url, verify=verify)
                elapsed = time.perf_counter() - started
                attempt = self._attempt_from_response(
                    response=response,
                    request_url=url,
                    verify=verify,
                    ssl_verify_disabled=ssl_verify_disabled,
                    elapsed_seconds=elapsed,
                    retry_count=retry_count,
                )
                last_attempt = attempt
                if (
                    retry_count < self.max_retries
                    and attempt.status_code in RETRY_STATUS_CODES
                ):
                    self._backoff_sleep(retry_count)
                    continue
                return attempt
            except requests.RequestException as exc:
                elapsed = time.perf_counter() - started
                last_attempt = FetchAttempt(
                    request_url=url,
                    final_url=None,
                    status_code=None,
                    success=False,
                    redirected=False,
                    used_ip_access=urlparse(url).hostname == LEGACY_IP_HOST,
                    ssl_verify_disabled=ssl_verify_disabled,
                    verify=verify,
                    elapsed_seconds=elapsed,
                    retry_count=retry_count,
                    html="",
                    error_type=exc.__class__.__name__,
                    error_message=str(exc),
                )
                if retry_count < self.max_retries:
                    self._backoff_sleep(retry_count)
                    continue
                return last_attempt

        if last_attempt is None:
            raise RuntimeError("request loop ended without an attempt")
        return last_attempt

    def _get(self, url: str, verify: bool) -> Response:
        if verify:
            return self.session.get(
                url,
                headers=self.headers,
                timeout=self.timeout,
                allow_redirects=True,
                verify=True,
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InsecureRequestWarning)
            return self.session.get(
                url,
                headers=self.headers,
                timeout=self.timeout,
                allow_redirects=True,
                verify=False,
            )

    def _attempt_from_response(
        self,
        response: Response,
        request_url: str,
        verify: bool,
        ssl_verify_disabled: bool,
        elapsed_seconds: float,
        retry_count: int,
    ) -> FetchAttempt:
        final_url = response.url
        return FetchAttempt(
            request_url=request_url,
            final_url=final_url,
            status_code=response.status_code,
            success=200 <= response.status_code < 400,
            redirected=bool(response.history) or final_url != request_url,
            used_ip_access=urlparse(request_url).hostname == LEGACY_IP_HOST,
            ssl_verify_disabled=ssl_verify_disabled,
            verify=verify,
            elapsed_seconds=elapsed_seconds,
            retry_count=retry_count,
            html=response.text or "",
        )

    def _respect_delay(self) -> None:
        if self.delay_seconds <= 0:
            return
        elapsed = time.perf_counter() - self._last_request_at
        wait_seconds = self.delay_seconds - elapsed
        if self._last_request_at and wait_seconds > 0:
            time.sleep(wait_seconds)
        self._last_request_at = time.perf_counter()

    def _backoff_sleep(self, retry_count: int) -> None:
        time.sleep(self.backoff_factor * (2**retry_count))


def _is_old_subdomain_host(host: str) -> bool:
    return host == OLD_SUBDOMAIN_MAIN_HOST or host.endswith(OLD_SUBDOMAIN_SUFFIX)
