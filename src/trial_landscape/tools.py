"""Function declaration schemas (for Gemini) and their implementations (against ClinicalTrials.gov)."""

from __future__ import annotations

from collections import Counter
from typing import Any

from trial_landscape.ctgov import (
    AGGREGATE_FIELDS,
    DEFAULT_MAX_STUDIES,
    SEARCH_FIELDS,
    VALID_PHASES,
    VALID_STATUSES,
    CTGovClient,
    CTGovError,
)

# ---------------------------------------------------------------------------
# Tool schemas — each becomes a google.genai.types.FunctionDeclaration
# (see agent.py), keyed here by "parameters" holding a plain JSON schema.
# ---------------------------------------------------------------------------

_FILTER_PROPERTIES = {
    "condition": {
        "type": "string",
        "description": "Disease/condition to search for, e.g. 'non-small cell lung cancer' or 'NSCLC'.",
    },
    "intervention": {
        "type": "string",
        "description": "Drug, device, or intervention name, e.g. 'sotorasib' or 'KRAS G12C inhibitor'.",
    },
    "sponsor": {
        "type": "string",
        "description": "Lead sponsor name (organization or company), e.g. 'Amgen'.",
    },
    "status": {
        "type": "string",
        "description": (
            "Comma-separated overall trial status filter. Valid values: "
            + ", ".join(sorted(VALID_STATUSES))
            + ". Common ones: RECRUITING, ACTIVE_NOT_RECRUITING, COMPLETED, TERMINATED, NOT_YET_RECRUITING."
        ),
    },
    "phase": {
        "type": "array",
        "items": {"type": "string", "enum": sorted(VALID_PHASES)},
        "description": (
            "Trial phase(s) to filter on. Pass multiple values for a range, "
            "e.g. ['PHASE2','PHASE3'] for a 'phase 2/3' query."
        ),
    },
    "location": {
        "type": "string",
        "description": "Geographic location filter, e.g. 'Boston' or 'Germany'.",
    },
}

SEARCH_TRIALS_TOOL = {
    "name": "search_trials",
    "description": (
        "Search ClinicalTrials.gov for trials matching the given filters and return a "
        "condensed list (NCT ID, title, phase, status, sponsor, enrollment, start date) — "
        "not the full raw study record. Use this to browse or scan a landscape of trials. "
        "For deep detail on one specific trial (eligibility criteria, arms, outcomes), call "
        "get_study_details with its NCT ID afterward. For counts/distributions across many "
        "trials (e.g. 'how crowded is this space'), use aggregate_trials instead — it's "
        "cheaper and won't flood your context with a long trial list."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            **_FILTER_PROPERTIES,
            "sort": {
                "type": "string",
                "description": (
                    "Sort order, field:direction. Common fields: LastUpdatePostDate, "
                    "StudyFirstPostDate, EnrollmentCount, PrimaryCompletionDate. "
                    "E.g. 'LastUpdatePostDate:desc' for most-recently-updated first."
                ),
            },
            "max_results": {
                "type": "integer",
                "description": (
                    f"Max number of trials to return, auto-paginating as needed. "
                    f"Default 30, hard cap {DEFAULT_MAX_STUDIES}."
                ),
            },
        },
    },
}

GET_STUDY_DETAILS_TOOL = {
    "name": "get_study_details",
    "description": (
        "Get full detail on a single trial by NCT ID — eligibility criteria, arms/interventions, "
        "outcome measures, detailed description, and locations. Use this to drill into a specific "
        "trial that search_trials or aggregate_trials flagged as notable."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "nct_id": {
                "type": "string",
                "description": "The NCT ID of the trial, e.g. 'NCT03600883'.",
            },
        },
        "required": ["nct_id"],
    },
}

AGGREGATE_TRIALS_TOOL = {
    "name": "aggregate_trials",
    "description": (
        "Get aggregate statistics for trials matching the given filters — counts by phase, "
        "by status, top sponsors, and trial count by start year (for recency/momentum signal) "
        "— instead of a list of individual trials. Use this to answer 'how crowded is this "
        "space', compare activity across drugs/conditions, or spot white space and recency "
        "trends, without dumping every matching trial into context."
    ),
    "parameters": {
        "type": "object",
        "properties": _FILTER_PROPERTIES,
    },
}

ALL_TOOLS = [SEARCH_TRIALS_TOOL, GET_STUDY_DETAILS_TOOL, AGGREGATE_TRIALS_TOOL]


# ---------------------------------------------------------------------------
# Study record condensing helpers
# ---------------------------------------------------------------------------


def _dig(d: dict[str, Any], *path: str) -> Any:
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _truncate(text: str | None, n: int) -> str | None:
    if not text:
        return text
    return text if len(text) <= n else text[:n].rstrip() + "…"


def condense_study(study: dict[str, Any]) -> dict[str, Any]:
    ps = study.get("protocolSection", {})
    return {
        "nct_id": _dig(ps, "identificationModule", "nctId"),
        "title": _dig(ps, "identificationModule", "briefTitle"),
        "status": _dig(ps, "statusModule", "overallStatus"),
        "phase": _dig(ps, "designModule", "phases") or [],
        "sponsor": _dig(ps, "sponsorCollaboratorsModule", "leadSponsor", "name"),
        "enrollment": _dig(ps, "designModule", "enrollmentInfo", "count"),
        "start_date": _dig(ps, "statusModule", "startDateStruct", "date"),
    }


def condense_study_details(study: dict[str, Any]) -> dict[str, Any]:
    ps = study.get("protocolSection", {})
    locations = _dig(ps, "contactsLocationsModule", "locations") or []
    location_summary = [
        {
            "facility": loc.get("facility"),
            "city": loc.get("city"),
            "state": loc.get("state"),
            "country": loc.get("country"),
            "status": loc.get("status"),
        }
        for loc in locations[:10]
    ]
    interventions = _dig(ps, "armsInterventionsModule", "interventions") or []
    arms = _dig(ps, "armsInterventionsModule", "armGroups") or []
    outcomes_primary = _dig(ps, "outcomesModule", "primaryOutcomes") or []
    outcomes_secondary = _dig(ps, "outcomesModule", "secondaryOutcomes") or []

    return {
        "nct_id": _dig(ps, "identificationModule", "nctId"),
        "title": _dig(ps, "identificationModule", "briefTitle"),
        "official_title": _dig(ps, "identificationModule", "officialTitle"),
        "status": _dig(ps, "statusModule", "overallStatus"),
        "why_stopped": _dig(ps, "statusModule", "whyStopped"),
        "phase": _dig(ps, "designModule", "phases") or [],
        "study_type": _dig(ps, "designModule", "studyType"),
        "sponsor": _dig(ps, "sponsorCollaboratorsModule", "leadSponsor", "name"),
        "collaborators": [
            c.get("name") for c in (_dig(ps, "sponsorCollaboratorsModule", "collaborators") or [])
        ],
        "enrollment": _dig(ps, "designModule", "enrollmentInfo", "count"),
        "start_date": _dig(ps, "statusModule", "startDateStruct", "date"),
        "primary_completion_date": _dig(ps, "statusModule", "primaryCompletionDateStruct", "date"),
        "conditions": _dig(ps, "conditionsModule", "conditions") or [],
        "brief_summary": _truncate(_dig(ps, "descriptionModule", "briefSummary"), 1500),
        "interventions": [
            {"type": iv.get("type"), "name": iv.get("name")} for iv in interventions
        ],
        "arms": [
            {"label": a.get("label"), "type": a.get("type"), "description": _truncate(a.get("description"), 300)}
            for a in arms
        ],
        "primary_outcomes": [
            {"measure": o.get("measure"), "time_frame": o.get("timeFrame")} for o in outcomes_primary
        ],
        "secondary_outcomes": [
            {"measure": o.get("measure"), "time_frame": o.get("timeFrame")} for o in outcomes_secondary
        ],
        "eligibility_criteria": _truncate(_dig(ps, "eligibilityModule", "eligibilityCriteria"), 4000),
        "min_age": _dig(ps, "eligibilityModule", "minimumAge"),
        "max_age": _dig(ps, "eligibilityModule", "maximumAge"),
        "sex": _dig(ps, "eligibilityModule", "sex"),
        "location_count": len(locations),
        "locations_sample": location_summary,
    }


def _phase_bucket(phases: list[str]) -> str:
    if not phases:
        return "NOT_SPECIFIED"
    return "/".join(sorted(phases))


def _start_year(date_str: str | None) -> str | None:
    if not date_str or len(date_str) < 4:
        return None
    return date_str[:4]


# ---------------------------------------------------------------------------
# Tool dispatch — called from the orchestration loop with Claude's tool_use.input
# ---------------------------------------------------------------------------


def run_search_trials(client: CTGovClient, args: dict[str, Any]) -> dict[str, Any]:
    max_results = min(int(args.get("max_results") or 30), DEFAULT_MAX_STUDIES)
    studies, total_count = client.search(
        condition=args.get("condition"),
        intervention=args.get("intervention"),
        sponsor=args.get("sponsor"),
        status=args.get("status"),
        phases=args.get("phase"),
        location=args.get("location"),
        sort=args.get("sort"),
        fields=SEARCH_FIELDS,
        page_size=min(50, max_results),
        max_studies=max_results,
    )
    return {
        "total_matching": total_count,
        "returned": len(studies),
        "trials": [condense_study(s) for s in studies],
    }


def run_get_study_details(client: CTGovClient, args: dict[str, Any]) -> dict[str, Any]:
    nct_id = args["nct_id"]
    study = client.get_study(nct_id)
    return condense_study_details(study)


def run_aggregate_trials(client: CTGovClient, args: dict[str, Any]) -> dict[str, Any]:
    studies, total_count = client.search(
        condition=args.get("condition"),
        intervention=args.get("intervention"),
        sponsor=args.get("sponsor"),
        status=args.get("status"),
        phases=args.get("phase"),
        location=args.get("location"),
        fields=AGGREGATE_FIELDS,
        page_size=100,
        max_studies=DEFAULT_MAX_STUDIES,
    )

    phase_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    sponsor_counts: Counter[str] = Counter()
    year_counts: Counter[str] = Counter()

    for s in studies:
        c = condense_study(s)
        phase_counts[_phase_bucket(c["phase"])] += 1
        if c["status"]:
            status_counts[c["status"]] += 1
        if c["sponsor"]:
            sponsor_counts[c["sponsor"]] += 1
        year = _start_year(c["start_date"])
        if year:
            year_counts[year] += 1

    sample_size = len(studies)
    result: dict[str, Any] = {
        "total_matching": total_count,
        "sample_analyzed": sample_size,
        "by_phase": dict(phase_counts.most_common()),
        "by_status": dict(status_counts.most_common()),
        "top_sponsors": dict(sponsor_counts.most_common(15)),
        "trials_by_start_year": dict(sorted(year_counts.items())),
    }
    if total_count is not None and sample_size < total_count:
        result["note"] = (
            f"Aggregates computed from a sample of {sample_size} trials "
            f"(the {DEFAULT_MAX_STUDIES}-trial analysis cap was hit); "
            f"true total matching the filters is {total_count}."
        )
    return result


DISPATCH = {
    "search_trials": run_search_trials,
    "get_study_details": run_get_study_details,
    "aggregate_trials": run_aggregate_trials,
}


def execute_tool(client: CTGovClient, name: str, args: dict[str, Any]) -> tuple[Any, bool]:
    """Runs a tool call; returns (result, is_error). Never raises — CTGovError becomes an error result."""
    handler = DISPATCH.get(name)
    if handler is None:
        return {"error": f"Unknown tool '{name}'"}, True
    try:
        return handler(client, args), False
    except CTGovError as exc:
        return {"error": str(exc)}, True
    except Exception as exc:  # noqa: BLE001 - surfaced to Claude as a tool error, not a crash
        return {"error": f"Unexpected error running {name}: {exc}"}, True
