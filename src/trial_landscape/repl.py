"""Interactive REPL: type a natural-language question, watch the tool calls, get a synthesis."""

from __future__ import annotations

import os
from datetime import datetime

import pyperclip
from dotenv import load_dotenv
from rich.console import Console

from trial_landscape.agent import (
    Agent,
    ModelCache,
    create_client,
    discover_working_models,
    export_conversation_markdown,
    markdown_to_plain_text,
    prompt_for_model,
    resolve_startup_model,
    verify_model,
)

BANNER = """[bold]trial-landscape[/] — natural-language research over ClinicalTrials.gov
Ask things like:
  [dim]"what's the phase 3 landscape for KRAS G12C inhibitors, recruiting only"[/]
  [dim]"compare trial activity for sotorasib vs adagrasib over the last 2 years"[/]
Commands: [bold]/help[/] full guide · [bold]/copy[/] copy last answer · [bold]/export[/] save conversation · [bold]/reset[/] clear conversation · [bold]/model[/] switch model · [bold]/stats[/] cache stats · [bold]/exit[/] quit
Press [bold]Ctrl+C[/] mid-answer to stop that query and return to the prompt.
"""

HELP_TEXT = """[bold]What this does[/]
Ask a plain-English question about clinical trials — Gemini decides which ClinicalTrials.gov \
calls to make on your behalf, then writes a natural-language answer citing specific trials.

[bold]Capabilities[/]
  • [bold]Landscape / momentum[/] — counts by phase, status, sponsor, and year for a condition or \
drug (e.g. "how crowded is the KRAS G12C phase 3 space")
  • [bold]Search & browse[/] — list matching trials by condition, intervention, sponsor, phase, \
status, or location
  • [bold]Deep-dive[/] — full eligibility criteria, arms, and outcomes for one trial by NCT ID

[bold]Commands[/]
  [bold]/help[/]               show this message
  [bold]/copy[/]               copy the latest answer to the clipboard as plain text (tables included)
  [bold]/export[/] [dim][path][/]      save the conversation since the last /reset as a Markdown file
  [bold]/reset[/]              clear conversation history
  [bold]/model[/]              switch models (menu) · [bold]/model <name-or-number>[/] · [bold]/model refresh[/]
  [bold]/stats[/]              ClinicalTrials.gov cache hit/miss counts for this session
  [bold]/exit[/], [bold]/quit[/]        quit
  [bold]Ctrl+C[/]              stop the current query, return to the prompt (doesn't exit)
"""


def _switch_model(agent: Agent, requested: str, console: Console, cache: ModelCache) -> None:
    """Pings `requested` before committing to it (cache-backed, so an already-checked
    model is free), so a model that's broken for this account fails fast and reverts
    instead of silently eating the user's next real question."""
    if requested not in cache:
        console.print(f"[dim]checking {requested}…[/]")
    error = verify_model(agent.client, requested, cache=cache)
    if error is not None:
        console.print(f"[red]{requested} is not usable with this API key: {error}[/]")
        console.print(f"[dim]staying on {agent.model}[/]")
        return
    agent.model = requested
    console.print(f"[dim]using model: {requested}[/]")


def main() -> None:
    load_dotenv()  # picks up GEMINI_API_KEY from a .env file in the cwd, if present
    console = Console()
    console.print(BANNER)

    try:
        client = create_client()
    except SystemExit as exc:
        console.print(f"[bold red]{exc}[/]")
        raise

    # Discovered once, live, from the account's actual model catalog — not a
    # hardcoded version list, so new releases (e.g. a future gemini-3.7-flash)
    # show up automatically. /model reuses this list rather than re-discovering
    # on every invocation; /model refresh re-runs discovery on demand.
    model_cache: ModelCache = {}
    try:
        model, available_models = resolve_startup_model(
            client, console, cache=model_cache, override=os.environ.get("GEMINI_MODEL")
        )
    except SystemExit as exc:
        console.print(f"[bold red]{exc}[/]")
        raise
    console.print(f"[dim]using model: {model} (switch anytime with /model)[/]")

    agent = Agent(model=model, client=client, console=console)
    agent.available_models = available_models

    while True:
        try:
            query = console.input("[bold cyan]you>[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break

        if not query:
            continue
        if query in {"/exit", "/quit"}:
            break
        if query in {"/help", "/?"}:
            console.print(HELP_TEXT)
            continue
        if query == "/copy":
            if agent.last_answer is None:
                console.print("[dim]nothing to copy yet — ask a question first[/]")
                continue
            plain_text = markdown_to_plain_text(agent.last_answer)
            try:
                pyperclip.copy(plain_text)
                console.print("[dim]copied latest answer to clipboard (plain text)[/]")
            except pyperclip.PyperclipException as exc:
                console.print(f"[red]couldn't copy to clipboard: {exc}[/]")
            continue
        if query == "/export" or query.startswith("/export "):
            if not agent.contents:
                console.print("[dim]nothing to export yet — ask a question first[/]")
                continue
            path = query[len("/export"):].strip()
            if not path:
                path = f"trial-landscape-export-{datetime.now().strftime('%Y%m%d-%H%M%S')}.md"
            markdown = export_conversation_markdown(agent.contents, agent.model)
            try:
                with open(path, "w") as f:
                    f.write(markdown)
                console.print(f"[dim]exported conversation to {os.path.abspath(path)}[/]")
            except OSError as exc:
                console.print(f"[red]couldn't write export file: {exc}[/]")
            continue
        if query == "/reset":
            agent.contents.clear()
            console.print("[dim]conversation cleared[/]")
            continue
        if query == "/model":
            console.print(f"[dim]current model: {agent.model}[/]")
            choice = prompt_for_model(agent.available_models, console)
            _switch_model(agent, choice, console, model_cache)
            continue
        if query == "/model refresh":
            console.print("[dim]re-checking available models…[/]")
            agent.available_models = discover_working_models(client, console, cache=model_cache)
            continue
        if query.startswith("/model "):
            arg = query[len("/model "):].strip()
            if arg.isdigit() and 1 <= int(arg) <= len(agent.available_models):
                arg = agent.available_models[int(arg) - 1]  # e.g. "/model 2" = menu item 2
            _switch_model(agent, arg, console, model_cache)
            continue
        if query == "/stats":
            c = agent.ctgov
            console.print(
                f"[dim]cache hits: {c.cache_hits} · cache misses (real API calls): {c.cache_misses}[/]"
            )
            continue

        answer = agent.ask(query)
        if answer is None:
            console.print("[yellow]interrupted — back to prompt[/]")
            continue
        agent.render_answer(answer)

    console.print("[dim]goodbye[/]")


if __name__ == "__main__":
    main()
