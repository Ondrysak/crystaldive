#!/usr/bin/env python3
"""Discord progress notifier for Claude Code hooks.

Invoked from .claude/settings.json as:
    python .claude/hooks/discord-notify.py <event-name>

Reads the hook event JSON from stdin, the webhook URL from project .env, and
POSTs a short, human-readable line to the channel. Always exits 0 — a Discord
blip or missing webhook never breaks the Claude Code session. All hooks should
be configured with "async": true so this script never blocks Claude.

Throttle: silently drops messages if >25 have been sent in the last 60s, to
stay under Discord's webhook rate limit (30/min) and keep room for the bursts
that matter (e.g. dangerous-command alerts).
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time
import urllib.request


# ── helpers ──────────────────────────────────────────────────────────────────


def project_dir() -> pathlib.Path:
    return pathlib.Path(os.environ.get("CLAUDE_PROJECT_DIR", os.getcwd()))


def load_webhook() -> str | None:
    env = project_dir() / ".env"
    if not env.exists():
        return None
    try:
        for raw in env.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line.startswith("DISCORD_AGENT_WEBHOOK="):
                val = line.split("=", 1)[1].strip().strip('"').strip("'")
                return val or None
    except OSError:
        return None
    return None


def should_send() -> bool:
    """Client-side throttle: ≤25 messages per 60-second sliding window."""
    rate_file = project_dir() / ".claude" / ".discord-rate.json"
    now = time.time()
    try:
        history = json.loads(rate_file.read_text(encoding="utf-8"))
        if not isinstance(history, list):
            history = []
    except (FileNotFoundError, ValueError, OSError):
        history = []
    history = [float(t) for t in history if isinstance(t, (int, float)) and now - float(t) < 60.0]
    if len(history) >= 25:
        return False
    history.append(now)
    try:
        rate_file.parent.mkdir(parents=True, exist_ok=True)
        rate_file.write_text(json.dumps(history), encoding="utf-8")
    except OSError:
        pass  # best-effort; throttle still works in-memory for this invocation
    return True


def send(content: str) -> None:
    if not content:
        return
    if len(content) > 1800:
        content = content[:1799] + "…"
    webhook = load_webhook()
    if not webhook:
        return
    if not should_send():
        return
    body = json.dumps({"content": content}).encode("utf-8")
    req = urllib.request.Request(
        webhook,
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": "crystaldive-hook/1"},
    )
    try:
        urllib.request.urlopen(req, timeout=5).read()
    except Exception:
        pass  # swallow; Discord blips can't break the session


def truncate(s, n=140):
    """Word-boundary truncate. Returns at most n chars; ends with '…' if cut."""
    if s is None:
        return ""
    s = str(s).strip().replace("\r", " ").replace("\n", " ")
    while "  " in s:
        s = s.replace("  ", " ")
    if len(s) <= n:
        return s
    cut = s.rfind(" ", 0, n - 1)
    if cut < n // 2:
        cut = n - 1
    return s[:cut].rstrip(" ,.;:—-") + "…"


_HEADER_PREFIXES = ("# ", "## ", "### ", "#### ", "---", "===")


def first_sentence(s, n=120):
    """Drop leading markdown headers/code fences, then take first sentence up
    to n chars, word-truncated. Keeps Discord lines scannable.
    """
    if not s:
        return ""
    s = str(s).strip()
    # Strip leading markdown noise — header lines, hr rules, code fences.
    lines = s.splitlines()
    while lines:
        ln = lines[0].strip()
        if not ln:
            lines.pop(0)
            continue
        if ln.startswith("```") or ln.startswith(_HEADER_PREFIXES):
            lines.pop(0)
            continue
        break
    s = " ".join(ln.strip() for ln in lines if ln.strip())
    # First sentence: split on the first sentence-ender.
    end = -1
    for i, ch in enumerate(s):
        if ch in ".!?" and (i + 1 == len(s) or s[i + 1] in " \n\t"):
            end = i + 1
            break
    if 0 < end <= n + 40:
        s = s[:end]
    return truncate(s, n)


def summarize_turn(transcript_path: str) -> str:
    """Walk the transcript JSONL backwards from the end, count tool_use blocks
    in the assistant messages of the current turn (i.e. since the most recent
    real user prompt — a user message containing text rather than only
    tool_result blocks). Returns a short string like "6 Edit, 4 Bash, 2 Read"
    or empty string if nothing useful is available.
    """
    if not transcript_path:
        return ""
    try:
        lines = pathlib.Path(transcript_path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""

    counts: dict[str, int] = {}
    for raw in reversed(lines):
        if not raw.strip():
            continue
        try:
            entry = json.loads(raw)
        except (ValueError, json.JSONDecodeError):
            continue
        etype = entry.get("type")
        msg = entry.get("message") if isinstance(entry.get("message"), dict) else entry
        content = msg.get("content") if isinstance(msg, dict) else None

        if etype == "user" and isinstance(content, list):
            # If this user entry has any text block, it's a real prompt that
            # opens a new turn — stop walking. Tool-result wrappers (content
            # is only tool_result blocks) are part of the current turn.
            has_text = any(
                isinstance(b, dict) and b.get("type") == "text" for b in content
            )
            if has_text:
                break
            continue

        if etype == "assistant" and isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    name = block.get("name") or "?"
                    counts[name] = counts.get(name, 0) + 1

    if not counts:
        return ""
    # Sort by count desc, then name asc for stability.
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    top = ranked[:4]
    rest = ranked[4:]
    parts = [f"{c} {n}" for n, c in top]
    if rest:
        parts.append(f"+{sum(c for _, c in rest)} more")
    return ", ".join(parts)


# ── per-event formatters ─────────────────────────────────────────────────────


def format_message(event: str, d: dict) -> str | None:
    tool = d.get("tool_name", "")
    inp = d.get("tool_input") or {}

    if event == "SessionStart":
        return f"🟢 session started ({d.get('source', 'startup')})"

    if event == "SessionEnd":
        return f"⚫ session ended ({d.get('reason', 'other')})"

    if event == "UserPromptSubmit":
        return f"👤 {truncate(d.get('prompt', ''), 220)}"

    if event == "Stop":
        msg = first_sentence(d.get("last_assistant_message", ""), 120)
        tools = summarize_turn(d.get("transcript_path", ""))
        head = "✅ turn done"
        if tools:
            head = f"{head} · {tools}"
        return f"{head} — {msg}" if msg else head

    if event == "StopFailure":
        return f"🔴 turn failed: {truncate(d.get('error', 'unknown'), 80)}"

    if event == "Notification":
        kind = d.get("notification_type", "notif")
        return f"🔔 {kind}: {truncate(d.get('message', ''), 180)}"

    if event == "PreToolUse":
        if tool == "Bash":
            cmd = truncate(inp.get("command", ""), 200)
            return f"🚨 PRE-Bash: {cmd}"
        return f"🚨 PRE-{tool}"

    if event == "PostToolUse":
        if tool == "Bash":
            cmd = truncate(inp.get("command", ""), 150)
            desc = (inp.get("description") or "").strip()
            return f"🔧 bash · {desc}: {cmd}" if desc else f"🔧 bash: {cmd}"
        if tool in ("Edit", "Write"):
            return f"📝 {tool.lower()}: {inp.get('file_path', '?')}"
        if tool == "Agent":
            desc = truncate(inp.get("description", ""), 100)
            stype = inp.get("subagent_type", "")
            return f"👥 agent done ({stype}): {desc}" if stype else f"👥 agent done: {desc}"
        return f"✓ {tool}"

    if event == "PostToolUseFailure":
        return f"💥 {tool or 'tool'} failed: {truncate(d.get('error', ''), 160)}"

    if event == "SubagentStart":
        return f"🚀 subagent spawn: {d.get('agent_type', '?')}"

    if event == "SubagentStop":
        atype = d.get("agent_type", "?")
        msg = first_sentence(d.get("last_assistant_message", ""), 120)
        return f"🏁 subagent done ({atype}) — {msg}" if msg else f"🏁 subagent done ({atype})"

    if event == "TaskCreated":
        return f"📋 task: {truncate(d.get('task_subject', ''), 160)}"

    if event == "TaskCompleted":
        return f"✔️ task done: {truncate(d.get('task_subject', ''), 160)}"

    if event == "PreCompact":
        return f"🗜️ compacting ({d.get('trigger', '?')})"

    if event == "PostCompact":
        return "🗜️ compact done"

    return None


# ── entry point ──────────────────────────────────────────────────────────────


def main() -> None:
    if len(sys.argv) < 2:
        return
    event = sys.argv[1]
    # Read stdin as UTF-8 explicitly — on Windows, sys.stdin defaults to
    # cp1252 which mojibakes em-dashes and other multi-byte UTF-8 chars
    # (the `â€"` artifact in Discord).
    try:
        raw = sys.stdin.buffer.read().decode("utf-8", errors="replace")
        data = json.loads(raw) if raw.strip() else {}
        if not isinstance(data, dict):
            data = {}
    except (json.JSONDecodeError, ValueError, OSError):
        data = {}
    msg = format_message(event, data)
    if msg:
        send(msg)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # absolutely never propagate — hooks must exit 0
        pass
