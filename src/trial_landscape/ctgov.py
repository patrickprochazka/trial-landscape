"""Thin, cached, rate-limited client for the ClinicalTrials.gov API v2.

Docs: https://clinicaltrials.gov/data-api/api
No auth required. Documented soft limit is ~50 requests/minute.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

BASE_URL = "https://clinicaltrials.gov/api/v2"

# Condensed field set returned by search_trials — enough to identify and
# triage a trial without pulling the full record into the model's context.
SEARCH_FIELDS = [
    "NCTId",
    "BriefTitle",
    "OverallStatus",
    "Phase",
    "LeadSponsorName",
    "EnrollmentCount",
    "StartDate",
]

# Field set used internally by aggregate_trials — only what's needed to
# compute counts, never surfaced as full study records.
AGGREGATE_FIELDS = [
    "NCTId",
    "OverallStatus",
    "Phase",
    "LeadSponsorName",
    "StartDate",
]

VALID_STATUSES = {
    "ACTIVE_NOT_RECRUITING",
    "COMPLETED",
    "ENROLLING_BY_INVITATION",
    "NOT_YET_RECRUITING",
    "RECRUITING",
    "SUSPENDED",
    "TERMINATED",
    "WITHDRAWN",
    "AVAILABLE",
    "NO_LONGER_AVAILABLE",
    "TEMPORARILY_NOT_AVAILABLE",
    "APPROVED_FOR_MARKETING",
    "WITHHELD",
    "UNKNOWN",
}

VALID_PHASES = {"EARLY_PHASE1", "PHASE1", "PHASE2", "PHASE3", "PHASE4", "NA"}

# Hard ceiling on auto-pagination so a broad query can't accidentally pull
# the entire registry into memory/context.
DEFAULT_MAX_STUDIES = 1000
MIN_REQUEST_INTERVAL_SECONDS = 1.3  # keeps us safely under ~50 req/min


class CTGovError(Exception):
    """Raised for any ClinicalTrials.gov API failure (HTTP error, network error, bad params)."""


def _cache_key(path: str, params: dict[str, Any]) -> str:
    normalized = json.dumps({"path": path, "params": params}, sort_keys=True, default=str)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def normalize_status(status: str) -> str:
    """Accepts free-form status text and maps to the CT.gov enum, comma-joined for multiple."""
    parts = [p.strip().upper().replace(" ", "_").replace("-", "_") for p in status.split(",") if p.strip()]
    bad = [p for p in parts if p not in VALID_STATUSES]
    if bad:
        raise CTGovError(
            f"Unrecognized status value(s): {bad}. Valid values: {sorted(VALID_STATUSES)}"
        )
    return ",".join(parts)


def normalize_phases(phases: list[str]) -> str:
    """Builds an Essie filter.advanced expression like AREA[Phase](PHASE2 OR PHASE3)."""
    normalized = []
    for p in phases:
        key = p.strip().upper().replace(" ", "_").replace("-", "_")
        # allow shorthand like "1", "2/3", "early 1"
        key = {
            "1": "PHASE1",
            "2": "PHASE2",
            "3": "PHASE3",
            "4": "PHASE4",
            "EARLY_1": "EARLY_PHASE1",
            "N/A": "NA",
        }.get(key, key)
        if key not in VALID_PHASES:
            raise CTGovError(f"Unrecognized phase value: {p!r}. Valid values: {sorted(VALID_PHASES)}")
        normalized.append(key)
    if not normalized:
        return ""
    if len(normalized) == 1:
        return f"AREA[Phase]{normalized[0]}"
    return f"AREA[Phase]({' OR '.join(normalized)})"


@dataclass
class CTGovClient:
    """Session-scoped client: caches raw responses by query hash and rate-limits outbound calls."""

    _client: httpx.Client = field(default_factory=lambda: httpx.Client(timeout=20.0))
    _cache: dict[str, Any] = field(default_factory=dict)
    _last_request_at: float = field(default=0.0)
    cache_hits: int = field(default=0)
    cache_misses: int = field(default=0)

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < MIN_REQUEST_INTERVAL_SECONDS:
            time.sleep(MIN_REQUEST_INTERVAL_SECONDS - elapsed)

    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        params = {k: v for k, v in params.items() if v is not None and v != ""}
        key = _cache_key(path, params)
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]

        self.cache_misses += 1
        self._throttle()
        self._last_request_at = time.monotonic()
        try:
            resp = self._client.get(f"{BASE_URL}{path}", params=params)
        except httpx.HTTPError as exc:
            raise CTGovError(f"Network error calling ClinicalTrials.gov: {exc}") from exc

        if resp.status_code == 404:
            raise CTGovError(resp.text.strip() or "Not found")
        if resp.status_code >= 400:
            raise CTGovError(f"ClinicalTrials.gov API error ({resp.status_code}): {resp.text.strip()}")

        try:
            data = resp.json()
        except ValueError as exc:
            raise CTGovError(f"Could not parse ClinicalTrials.gov response as JSON: {exc}") from exc

        self._cache[key] = data
        return data

    def search(
        self,
        *,
        condition: str | None = None,
        intervention: str | None = None,
        sponsor: str | None = None,
        status: str | None = None,
        phases: list[str] | None = None,
        location: str | None = None,
        sort: str | None = None,
        fields: list[str],
        page_size: int = 50,
        max_studies: int = DEFAULT_MAX_STUDIES,
        count_total_only: bool = False,
    ) -> tuple[list[dict[str, Any]], int | None]:
        """Auto-paginates GET /studies via pageToken up to max_studies.

        Returns (studies, total_count). If count_total_only is True, does a
        single pageSize=1 request and returns ([], total_count) — used by
        aggregate_trials to report true match counts without pulling records.
        """
        base_params: dict[str, Any] = {
            "query.cond": condition,
            "query.intr": intervention,
            "query.spons": sponsor,
            "query.locn": location,
            "filter.overallStatus": normalize_status(status) if status else None,
            "filter.advanced": normalize_phases(phases) if phases else None,
            "sort": sort,
            "fields": ",".join(fields),
            "countTotal": "true",
        }

        if count_total_only:
            data = self._get("/studies", {**base_params, "pageSize": 1})
            return [], data.get("totalCount")

        studies: list[dict[str, Any]] = []
        total_count: int | None = None
        page_token: str | None = None
        while len(studies) < max_studies:
            remaining = max_studies - len(studies)
            params = {**base_params, "pageSize": min(page_size, remaining)}
            if page_token:
                params["pageToken"] = page_token
            data = self._get("/studies", params)
            if total_count is None:
                total_count = data.get("totalCount")
            page = data.get("studies", [])
            studies.extend(page)
            page_token = data.get("nextPageToken")
            if not page_token or not page:
                break
        return studies, total_count

    def get_study(self, nct_id: str) -> dict[str, Any]:
        nct_id = nct_id.strip().upper()
        return self._get(f"/studies/{nct_id}", {})
