# trial-landscape

A natural-language research assistant over the [ClinicalTrials.gov API v2](https://clinicaltrials.gov/data-api/api), powered by [Gemini function calling](https://ai.google.dev/gemini-api/docs/function-calling).

Type a plain-English question — "what's the phase 3 landscape for KRAS G12C inhibitors, recruiting only" — and Gemini decides which function(s) to call, in what order, and with what parameters, then synthesizes the raw API results into a natural-language answer with cited NCT IDs, sponsors, and trial titles.

## Architecture

```
you → REPL → Gemini (function calling) → ClinicalTrials.gov API v2
              ↑                                  ↓
              └──────── function response ───────┘ (condensed, cached, rate-limited)
```

- **Interactive REPL** (`repl.py`) — reads your question, hands it to the agent, prints each function call live so you can see (and explain) Gemini's reasoning, then prints the final synthesis.
- **Orchestration loop** (`agent.py`) — a manual function-calling loop against the `google-genai` SDK, with `automatic_function_calling` disabled so every call is visible before it executes and API errors are fed back to Gemini as a function response instead of crashing the process. Backs off with jittered exponential retry on `429 RESOURCE_EXHAUSTED`. Also owns model discovery — live catalog scanning, verification pings, and the best-model sort key (see Model selection below).
- **Tools** (`tools.py`) — three functions mapped onto the API:
  - `search_trials` — condensed trial list (NCT ID, title, phase, status, sponsor, enrollment, start date), auto-paginated up to a 1000-study cap.
  - `get_study_details` — full detail on one trial (eligibility criteria, arms, outcomes) by NCT ID.
  - `aggregate_trials` — counts by phase/status/sponsor/year, computed client-side from a capped, field-limited pull — for "how crowded is this space" questions without dumping every trial into context.
- **API client** (`ctgov.py`) — thin `httpx` wrapper: rate-limited to stay under the ~50 req/min soft limit, and caches raw responses by query hash for the life of the process so repeated/chained function calls don't re-hit the network.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and a free Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey).

Either export the key for the session:

```bash
export GEMINI_API_KEY=...
uv run trial-landscape
```

or drop it in a `.env` file in the project root (auto-loaded on startup, and git-ignored):

```
GEMINI_API_KEY=...
```

```bash
uv run trial-landscape
```

That's it — `uv run` resolves dependencies into an ephemeral environment on first run, no separate install step.

## Model selection

There's no hardcoded model list. On startup, the app discovers every Flash-family, function-calling-capable model your account's catalog exposes, then pings each one with a real `generate_content` call to confirm it actually works — a catalog listing alone isn't proof of that; some models appear in `client.models.list()` but return a live 404 at call time. The default is Google's `gemini-flash-latest` alias if it works (it always tracks their current recommendation), otherwise the newest verified pinned version. Because discovery is dynamic, a future release (e.g. `gemini-3.7-flash`) shows up automatically once it's in the catalog — no code change needed.

```
loading models…
using model: gemini-flash-latest (switch anytime with /model)
```

The scan is a one-time cost at startup (a few seconds). The verified list is cached for the session, so `/model` picks from it instantly with no re-scanning.

Override the startup default with `GEMINI_MODEL` (verified the same way):

```bash
GEMINI_MODEL=gemini-3.5-flash uv run trial-landscape
```

Switch models at any point with `/model`:

```
you> /model
current model: gemini-flash-latest
Choose a Gemini model:
  1. gemini-flash-latest (default)
  2. gemini-3.6-flash
  3. gemini-3.5-flash
  ...
model [1]> 2
using model: gemini-3.6-flash
```

- `/model <name>` — jump straight to a model by name, e.g. `/model gemini-3.5-flash`. Verified before committing; reverts with a clear error if it doesn't work.
- `/model refresh` — re-scans the catalog, e.g. to pick up a model that's appeared since startup.

## Usage

```
you> what's the phase 3 landscape for KRAS G12C inhibitors, recruiting only
```

Gemini will typically call `aggregate_trials` (or `search_trials`) with `intervention` set to something like "KRAS G12C" and `phase=["PHASE3"]`, `status="RECRUITING"`, print that function call and its result live, then write a synthesis.

```
you> compare trial activity for sotorasib vs adagrasib over the last 2 years
```

Gemini typically calls `aggregate_trials` once per drug and compares the results — chaining two function calls in one turn.

REPL commands:

- `/help` — show commands and capabilities (no API call)
- `/copy` — copy the latest answer to the clipboard as plain text; tables become aligned columns, no literal `**`/`#`/`|` markdown syntax
- `/export [--complete] [path]` — save the conversation since the last `/reset` as a Markdown file. By default it's just questions and final answers (in original Markdown, so tables/formatting survive); add `--complete` to also interleave the tool calls Gemini made and their raw JSON results — the full reasoning trace, not just the Q&A. `path` is optional and flexible: omit it entirely for an auto-named file (`trial-landscape-export-<timestamp>.md`) in the current directory, give a folder (e.g. `/export ~/Desktop`) to auto-name inside it, or give a full filename to use it exactly. `~` is expanded.
- `/reset` — clear conversation history (start a fresh topic)
- `/model` — switch models, interactively or via `/model <name>`; `/model refresh` re-scans the catalog (see Model selection above)
- `/stats` — show ClinicalTrials.gov cache hit/miss counts for the session
- `/exit` — quit
- **Ctrl+C mid-answer** — stops the query in progress and returns to the prompt, without exiting the app. Any partial tool-call exchange from the interrupted step is discarded, so the next question starts from clean history.

Input is handled by [`prompt_toolkit`](https://python-prompt-toolkit.readthedocs.io/) rather than a bare `input()`, which matters for one thing specifically: **pasting a multi-line paragraph works correctly.** A terminal paste is captured as one atomic block (newlines included) — it does not submit early on each line break the way a naive `input()` loop would. Enter submits; **Alt+Enter** (or Escape then Enter) inserts a literal line break if you want to compose a multi-line question by hand. Arrow-up history recall comes along for free too.

## Design notes

- **Condensed, not raw, function results.** `search_trials` and `aggregate_trials` never return full study records — only the fields needed to identify or count trials. This keeps function-response payloads small so multi-call chains (e.g. comparing two drugs) stay within a reasonable context and token budget.
- **Session-scoped cache.** Every `GET` to ClinicalTrials.gov is cached in-memory by a hash of its path + params. If Gemini calls `search_trials` and then `aggregate_trials` with overlapping filters in the same session, or you ask a follow-up that revisits the same query, no extra network call is made.
- **Errors are recoverable, not fatal.** A bad NCT ID, an unrecognized phase/status value, or a network hiccup comes back to Gemini as a function response containing the real API error message, so it can retry with corrected parameters or explain the failure to you — the process never crashes on a bad function call.
- **Rate-limit resilient.** A `429` triggers jittered exponential backoff (printed live) rather than an immediate crash, so a burst of tool-call turns during a demo doesn't take the whole session down. If you want to conserve quota, `/model` to a smaller/lighter model from the verified list (see Model selection above).
