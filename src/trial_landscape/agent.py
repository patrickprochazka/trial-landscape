"""Gemini function-calling orchestration loop.

Manual loop (automatic_function_calling disabled) so we have full control over:
printing each function call live as it happens, feeding real API errors back to
Gemini as a function response instead of crashing, and backing off on 429s
ourselves so the retry behavior is visible and explainable, not hidden in the SDK.
"""

from __future__ import annotations

import json
import random
import re
import time
from datetime import datetime, timezone
from io import StringIO

import httpx
from google import genai
from google.genai import errors, types
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from trial_landscape.ctgov import CTGovClient
from trial_landscape.tools import ALL_TOOLS, execute_tool

# Model candidates are discovered live from client.models.list() (see
# discover_flash_candidates), never hardcoded — so a future release like
# gemini-3.7-flash shows up automatically with no code change. We only fall
# back to this fixed pair if the catalog listing itself fails (e.g. network
# issue): gemini-flash-latest is Google's own rolling alias for "whatever they
# currently recommend", the closest thing to a version-proof default.
FALLBACK_CANDIDATES = ["gemini-flash-latest", "gemini-2.5-flash"]
FALLBACK_MODEL = "gemini-flash-latest"

# Model names to exclude from the general text/function-calling menu even
# though they match "flash" — different modality (image/tts/audio) or
# streaming-only (no generateContent support, checked separately below).
_NON_TEXT_MODEL_TAGS = ("image", "tts", "audio", "live")
_VERSION_RE = re.compile(r"gemini-(\d+)(?:\.(\d+))?")

MAX_OUTPUT_TOKENS = 4096
MAX_TOOL_ITERATIONS = 8
MAX_RETRIES_ON_429 = 5
BASE_BACKOFF_SECONDS = 2.0

SYSTEM_PROMPT = """\
You are a clinical trial landscape research assistant built on the ClinicalTrials.gov API v2.

You have three functions:
- search_trials: a condensed list of individual trials matching filters.
- get_study_details: full detail on one specific trial by NCT ID (eligibility, arms, outcomes).
- aggregate_trials: counts by phase/status/sponsor/year for a set of filters — use this for
  "how crowded is this space" style questions instead of listing every trial.

Guidelines:
- Prefer aggregate_trials when the user wants a landscape view, momentum/recency signal, or a
  comparison between two drugs/conditions/sponsors. Call it once per side of a comparison.
- Use search_trials when the user wants to see or browse actual trials.
- Use get_study_details when a specific trial is worth citing in depth (e.g. the largest,
  newest, or only trial at a given phase) — don't call it for every trial in a list.
- Chain calls as needed: e.g. call aggregate_trials or search_trials for two different drugs to
  compare them, or call search_trials then drill into a standout NCT ID.
- Your final answer must be a natural-language synthesis, not raw JSON or a data dump. Cite
  specific NCT IDs, sponsor names, and trial titles when they support your point. Call out
  notable white space (phases/statuses with few or no trials), recency trends, or a dominant
  sponsor when the data shows one.
- If a function call returns an error, don't give up — adjust the parameters and retry once if
  the fix is obvious, otherwise tell the user plainly what went wrong.
"""


def create_client() -> genai.Client:
    try:
        return genai.Client()
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"Could not initialize the Gemini client: {exc}\n"
            "Set GEMINI_API_KEY in your environment and try again."
        ) from exc


# Per-model verification results for this session: model name -> None (works)
# or a short error string (doesn't). Shared across startup resolution and every
# /model invocation so a model already checked is never pinged twice.
ModelCache = dict[str, str | None]


def verify_model(client: genai.Client, model: str, cache: ModelCache | None = None) -> str | None:
    """Cheap ping to confirm `model` actually accepts generate_content calls for
    this account (being in client.models.list() doesn't guarantee that — see
    module docstring above). Returns None if it works, else a short error string.
    Reads/writes `cache` if given, so repeated checks of the same model are free.
    """
    if cache is not None and model in cache:
        return cache[model]
    try:
        client.models.generate_content(
            model=model,
            contents="ping",
            config=types.GenerateContentConfig(max_output_tokens=5),
        )
        result = None
    except Exception as exc:  # noqa: BLE001 - any failure means "don't use this model"
        result = str(exc).splitlines()[0][:180]
    if cache is not None:
        cache[model] = result
    return result


def _model_sort_key(name: str) -> tuple:
    """Newest/best-first ordering without hardcoding version numbers. Google's own
    rolling "-latest" alias (e.g. gemini-flash-latest) always tracks whatever they
    currently recommend, so it outranks everything, including pinned versions
    ahead of it in time — that's the real fix for "what about 3.7": no code change
    needed, the alias already points at it the day it ships. Pinned versions sort
    by parsed major.minor next, stable variants ahead of preview/lite at the same
    version."""
    is_rolling_alias = name.endswith("-latest") and "lite" not in name
    match = _VERSION_RE.search(name)
    major = int(match.group(1)) if match else 0
    minor = int(match.group(2)) if match and match.group(2) else 0
    is_stable = not any(tag in name for tag in ("preview", "-lite", "lite-", "exp"))
    return (is_rolling_alias, major, minor, is_stable)


def discover_flash_candidates(client: genai.Client) -> list[str]:
    """Lists every Flash-family, text/function-calling-capable model this
    account's catalog exposes, best-first. Unlike a hardcoded list, this picks
    up new releases (e.g. a future gemini-3.7-flash) the moment Google adds
    them to the catalog — no code change needed. Empty if listing fails."""
    try:
        models = list(client.models.list())
    except Exception:  # noqa: BLE001 - listing is a nice-to-have, never fatal
        return []

    names: set[str] = set()
    for m in models:
        name = m.name.removeprefix("models/")
        if "flash" not in name.lower():
            continue
        if any(tag in name.lower() for tag in _NON_TEXT_MODEL_TAGS):
            continue
        actions = getattr(m, "supported_actions", None)
        if actions and "generateContent" not in actions:
            continue
        names.add(name)
    return sorted(names, key=_model_sort_key, reverse=True)


def discover_working_models(client: genai.Client, console: Console, cache: ModelCache | None = None) -> list[str]:
    """Discovers the live Flash-family candidate pool (see discover_flash_candidates)
    and pings each one, returning only those that actually work for this account,
    best-first — the accurate alternative to trusting client.models.list() alone,
    which can list a model (e.g. gemini-2.5-flash) that then 404s at call time."""
    candidates = discover_flash_candidates(client) or list(FALLBACK_CANDIDATES)
    working = []
    with console.status("[dim]loading models…[/]"):
        for model in candidates:
            error = cache[model] if cache is not None and model in cache else verify_model(client, model, cache=cache)
            if error is None:
                working.append(model)
    return working


def resolve_startup_model(
    client: genai.Client, console: Console, cache: ModelCache | None = None, override: str | None = None
) -> tuple[str, list[str]]:
    """Runs discovery once, verifies `override` (e.g. from GEMINI_MODEL) if given,
    and returns (chosen_model, working_models) — the working list is what /model
    reuses afterward without re-discovering."""
    working = discover_working_models(client, console, cache=cache)

    if override:
        error = verify_model(client, override, cache=cache)
        if error is None:
            if override not in working:
                working = [override] + working
            return override, working
        console.print(f"[yellow]{override} isn't usable with this API key ({error}) — using the default instead[/]")

    if not working:
        raise SystemExit(
            "None of the known Gemini models could be reached with this API key.\n"
            "Check GEMINI_API_KEY and your account's model access at https://aistudio.google.com/apikey."
        )
    return working[0], working


def prompt_for_model(candidates: list[str], console: Console) -> str:
    """Interactively asks the user which model to use; Enter accepts the (first,
    best-available) default. `candidates` should already be verified-working —
    see discover_working_models."""
    if not candidates:
        console.print(f"[dim]no working models found, defaulting to {FALLBACK_MODEL}[/]")
        return FALLBACK_MODEL

    default_index = 1
    console.print("[bold]Choose a Gemini model:[/]")
    for i, name in enumerate(candidates, start=1):
        tag = " [green](default)[/]" if i == default_index else ""
        console.print(f"  {i}. {name}{tag}")

    try:
        raw = console.input(f"[bold cyan]model [{default_index}]>[/] ").strip()
    except (EOFError, KeyboardInterrupt):
        raw = ""

    if not raw:
        choice = candidates[default_index - 1]
    elif raw.isdigit() and 1 <= int(raw) <= len(candidates):
        choice = candidates[int(raw) - 1]
    elif raw in candidates:
        choice = raw
    else:
        console.print(f"[yellow]unrecognized choice '{raw}', using {candidates[default_index - 1]}[/]")
        choice = candidates[default_index - 1]

    return choice


def _to_function_declaration(tool: dict) -> types.FunctionDeclaration:
    return types.FunctionDeclaration(
        name=tool["name"],
        description=tool["description"],
        parameters_json_schema=tool["parameters"],
    )


def markdown_to_plain_text(markdown_text: str, width: int = 88) -> str:
    """Renders markdown — including GFM tables — into clean plain text suitable for
    pasting into an email or doc: no literal '**'/'#'/'|' syntax, tables become
    aligned columns with a separator line. Reuses the same Rich Markdown renderer
    as the terminal display (see Agent.render_answer), just captured to a string
    with color/terminal styling off instead of printed."""
    buf = StringIO()
    capture_console = Console(file=buf, width=width, no_color=True, force_terminal=False)
    capture_console.print(Markdown(markdown_text))
    lines = [line.rstrip() for line in buf.getvalue().splitlines()]
    return "\n".join(lines).strip("\n")


def export_conversation_markdown(
    contents: list[types.Content], model: str, include_tool_calls: bool = False
) -> str:
    """Renders the conversation (since the last /reset) as a Markdown transcript.
    By default, just questions and final answers (in original Markdown, so
    tables/formatting survive intact — contrast with markdown_to_plain_text).
    With include_tool_calls=True, also interleaves the tool calls Gemini made
    and their raw JSON results — the full reasoning trace, not just the Q&A."""
    mode_note = (
        "full trace: questions, tool calls, tool results, and answers"
        if include_tool_calls
        else "Q&A only — rerun with `/export --complete` for the full tool-call trace"
    )
    lines = [
        "# trial-landscape conversation export",
        "",
        f"_Exported {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} · model: {model} · {mode_note}_",
    ]
    for content in contents:
        parts = content.parts or []
        if content.role == "user":
            question = "\n".join(p.text for p in parts if p.text)
            if question:
                lines += ["", "---", "", "## You", "", question]
            if include_tool_calls:
                for part in parts:
                    if part.function_response is not None:
                        fr = part.function_response
                        lines += [
                            "",
                            f"**Result** (`{fr.name}`)",
                            "```json",
                            json.dumps(fr.response, indent=2, default=str),
                            "```",
                        ]
        elif content.role == "model":
            if include_tool_calls:
                for part in parts:
                    if part.function_call is not None:
                        fc = part.function_call
                        lines += [
                            "",
                            f"**Tool call:** `{fc.name}`",
                            "```json",
                            json.dumps(dict(fc.args or {}), indent=2, default=str),
                            "```",
                        ]
            answer = "\n".join(p.text for p in parts if p.text)
            if answer:
                lines += ["", "### Answer", "", answer]
    return "\n".join(lines).strip("\n") + "\n"


class Agent:
    def __init__(self, model: str, client: genai.Client | None = None, console: Console | None = None) -> None:
        self.console = console or Console()
        self.client = client or create_client()
        self.model = model
        self.tool = types.Tool(function_declarations=[_to_function_declaration(t) for t in ALL_TOOLS])
        self.config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=[self.tool],
            max_output_tokens=MAX_OUTPUT_TOKENS,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        self.ctgov = CTGovClient()
        self.contents: list[types.Content] = []
        self.last_answer: str | None = None

    def _print_tool_call(self, name: str, args: dict) -> None:
        pretty_args = json.dumps(args, indent=2)
        self.console.print(
            Panel(
                pretty_args,
                title=f"[bold cyan]function call[/] → [bold]{name}[/]",
                border_style="cyan",
                expand=False,
            )
        )

    def _print_tool_result(self, name: str, result, is_error: bool) -> None:
        style = "red" if is_error else "dim"
        label = "error" if is_error else "result"
        preview = json.dumps(result, indent=2)
        if len(preview) > 1200:
            preview = preview[:1200].rstrip() + "\n… (truncated for display; full result sent to Gemini)"
        self.console.print(
            Panel(
                preview,
                title=f"[{style}]function {label}[/] ← [bold]{name}[/]",
                border_style=style,
                expand=False,
            )
        )

    def _generate(self):
        """Calls generate_content with exponential backoff on 429 RESOURCE_EXHAUSTED
        and on transient network errors (dropped/reset connections, timeouts) —
        the SDK's own retry wrapper doesn't cover the latter, so without this a
        momentary network blip raises a raw httpx exception and crashes the REPL."""
        delay = BASE_BACKOFF_SECONDS
        for attempt in range(MAX_RETRIES_ON_429 + 1):
            try:
                return self.client.models.generate_content(
                    model=self.model,
                    contents=self.contents,
                    config=self.config,
                )
            except errors.ClientError as exc:
                if getattr(exc, "code", None) != 429 or attempt == MAX_RETRIES_ON_429:
                    raise
                reason = "rate limited (429)"
            except httpx.TransportError as exc:
                if attempt == MAX_RETRIES_ON_429:
                    raise
                reason = f"network error ({exc})"

            sleep_for = delay + random.uniform(0, 0.5 * delay)
            self.console.print(
                f"[yellow]{reason} — backing off {sleep_for:.1f}s "
                f"(attempt {attempt + 1}/{MAX_RETRIES_ON_429})[/]"
            )
            time.sleep(sleep_for)
            delay = min(delay * 2, 60.0)
        raise RuntimeError("unreachable")  # loop always returns or raises

    def ask(self, user_query: str) -> str | None:
        """Returns the final answer text, or None if the user interrupted (Ctrl+C)
        mid-query — the only way to "stop" a step in a synchronous REPL that's
        blocked on the network, since there's no input loop to type a command into
        until the current call returns. On interrupt, any partial turn (a tool call
        sent but not yet answered, etc.) is rolled back so the next question starts
        from clean, valid history instead of a half-finished exchange."""
        history_checkpoint = len(self.contents)
        self.contents.append(types.Content(role="user", parts=[types.Part(text=user_query)]))

        try:
            for _ in range(MAX_TOOL_ITERATIONS):
                try:
                    response = self._generate()
                except errors.ClientError as exc:
                    return f"[Gemini API error {getattr(exc, 'code', '?')}] {exc}"
                except errors.ServerError as exc:
                    return f"[Gemini server error] {exc}"
                except httpx.TransportError as exc:
                    return f"[Network error talking to Gemini] {exc}"
                except Exception as exc:  # noqa: BLE001 - final safety net; never let an unexpected error crash the REPL
                    return f"[Unexpected error calling Gemini] {exc}"

                model_content = response.candidates[0].content if response.candidates else None
                if model_content is not None:
                    self.contents.append(model_content)

                function_calls = response.function_calls or []
                if not function_calls:
                    text = response.text
                    return text or "(no response text)"

                response_parts = []
                for fc in function_calls:
                    self._print_tool_call(fc.name, fc.args or {})
                    result, is_error = execute_tool(self.ctgov, fc.name, fc.args or {})
                    self._print_tool_result(fc.name, result, is_error)
                    response_payload = {"error": result.get("error", str(result))} if is_error else result
                    response_parts.append(
                        types.Part.from_function_response(name=fc.name, response=response_payload)
                    )
                # Note: despite some docs suggesting role="tool", the live API rejects it
                # ("Role 'tool' is not supported... USER, ASSISTANT, ... MODEL, USER.") —
                # function responses go back as a "user" turn.
                self.contents.append(types.Content(role="user", parts=response_parts))

            return "(stopped after too many function-call rounds — try narrowing your question)"
        except KeyboardInterrupt:
            del self.contents[history_checkpoint:]
            return None

    def render_answer(self, text: str) -> None:
        self.last_answer = text
        self.console.print(Panel(Markdown(text), title="[bold green]answer[/]", border_style="green"))
