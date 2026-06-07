#!/usr/bin/env python3
"""
Commitment Tracker: declared-intention → outcome monitor.

Scans Hermes session transcripts for future-tense declarations,
stores them in a local SQLite DB, and checks on due dates whether
they were fulfilled.

MK built PRI/ACE to detect when language models break their commitments.
This is the dogfood: commitment-architecture applied to personal infra.

Usage:
    commitment_tracker.py scan [--days N]    Scan last N days of sessions
    commitment_tracker.py check              Check pending commitments (output ping)
    commitment_tracker.py add "text" --check YYYY-MM-DD   Manually add a commitment
    commitment_tracker.py resolve <id> --kept|--broken|--acknowledged
    commitment_tracker.py status             Show all commitments
    commitment_tracker.py reconcile <id> --kept|--broken   Resolve without interactive
"""

import argparse
import datetime
import json
import os
import re
import sqlite3
import sys
import textwrap

# ─── Paths ────────────────────────────────────────────────────────────────────

SESSION_DB = os.path.expanduser("~/.hermes/state.db")
COMMIT_DB = os.path.expanduser("~/.hermes/commitments.db")
HERMES_SCRIPTS = os.path.expanduser("~/.hermes/scripts")
PRI_SCRIPTS = os.path.expanduser("~/Documents/PRI_at_commitment/scripts")


# ─── DB Setup ─────────────────────────────────────────────────────────────────

def get_conn(db_path=COMMIT_DB):
    """Get a connection to the commitment DB, creating tables if needed."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS commitments (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            commitment_text TEXT NOT NULL,
            check_date      TEXT NOT NULL,  -- ISO date YYYY-MM-DD
            session_id      TEXT,
            message_id      INTEGER,
            session_title   TEXT,
            status          TEXT NOT NULL DEFAULT 'pending'
                            CHECK(status IN ('pending','kept','broken','acknowledged','expired')),
            outcome         TEXT,
            checked_at      TEXT,
            created_at      TEXT NOT NULL DEFAULT (datetime('now','localtime'))
        );
        CREATE TABLE IF NOT EXISTS extraction_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_date   TEXT NOT NULL DEFAULT (date('now')),
            found       INTEGER NOT NULL DEFAULT 0,
            new         INTEGER NOT NULL DEFAULT 0,
            detail      TEXT
        );
    """)
    return conn


# ─── Session Scanning ─────────────────────────────────────────────────────────

# Future-tense time references mapped to relative day offsets
TIME_PATTERNS = {
    # Absolute day-of-week references — resolved dynamically
    r'\btoday\b': 0,
    r'\btonight\b': 1,       # check tomorrow — did it happen?
    r'\btomorrow\b': 1,
    r'\bthis weekend\b': lambda: (6 - datetime.date.today().weekday()) % 7 + 1,  # next Monday
    r'\bthis week\b': lambda: (6 - datetime.date.today().weekday()) % 7 + 1,    # next Monday
    r'\bin the next few days\b': 3,
}

DAY_NAMES = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']


def _days_until_weekday(name):
    """Days from today until the next occurrence of the given weekday name."""
    today = datetime.date.today().weekday()
    target = DAY_NAMES.index(name.lower())
    return (target - today) % 7 or 7  # next week, not today


COMMITMENT_PATTERNS = [
    # "I'll <action> <time>" — e.g. "I'll push that tomorrow"
    re.compile(r"(?i)\bI\'(?:ll|m going to|m gonna)\s+(.+?)\s+(tomorrow|tonight|today|this\s+(?:weekend|week|month)|next\s+(?:week|month|year|\w+)|on\s+\w+day|by\s+\w+|in\s+\d+\s+\w+)"),
    # "I will <action> <time>"
    re.compile(r"(?i)\bI\s+will\s+(.+?)\s+(tomorrow|tonight|today|this\s+(?:weekend|week|month)|next\s+(?:week|month|year|\w+)|on\s+\w+day|by\s+\w+|in\s+\d+\s+\w+)"),
    # "Let's <action> <time>" — e.g. "Let's do that Friday"
    re.compile(r"(?i)\blet'?s\s+(.+?)\s+(tomorrow|tonight|this\s+(?:weekend|week|month)|next\s+(?:week|month|year|\w+)|on\s+\w+day|by\s+\w+|in\s+\d+\s+\w+)"),
    # "I need to <action> <time>" — e.g. "I need to finish by Friday"
    re.compile(r"(?i)\bI\s+need\s+to\s+(.+?)\s+(by\s+\w+|before\s+\w+|tomorrow|tonight|this\s+(?:weekend|week|month)|next\s+(?:week|month|year|\w+)|on\s+\w+day)"),
    # "I plan to / I'm planning to <action> <time>"
    re.compile(r"(?i)\bI\s*(?:'m\s+)?(?:plan|planning)\s+to\s+(.+?)\s+(tomorrow|tonight|this\s+(?:weekend|week|month)|next\s+(?:week|month|year|\w+)|on\s+\w+day|by\s+\w+)"),
    # "remind me to <action> <time>" / "remind me about <thing> <time>"
    re.compile(r"(?i)\b(?:remind|ping)\s+me\s+(?:to|about)\s+(.+?)\s+(tomorrow|tonight|this\s+(?:weekend|week|month)|next\s+(?:week|month|year|\w+)|on\s+\w+day|by\s+\w+)"),
    # "remind me <time>" — e.g. "remind me next week"
    re.compile(r"(?i)\b(?:remind|ping)\s+me\s+(tomorrow|tonight|this\s+(?:weekend|week|month)|next\s+(?:week|month|year|\w+)|on\s+\w+day|by\s+\w+)"),
]


def _resolve_check_date(time_ref: str) -> str | None:
    """Try to determine the check date from a time reference string."""
    today = datetime.date.today()
    tr = time_ref.strip().lower()

    # Direct day names
    for name in DAY_NAMES:
        if tr == f"on {name}":
            d = _days_until_weekday(name)
            return (today + datetime.timedelta(days=d)).isoformat()
        if tr == f"next {name}":
            d = _days_until_weekday(name)
            return (today + datetime.timedelta(days=d)).isoformat()

    # Known patterns
    for pat, offset in TIME_PATTERNS.items():
        if re.search(pat, tr):
            if callable(offset):
                return (today + datetime.timedelta(days=offset())).isoformat()
            return (today + datetime.timedelta(days=offset)).isoformat()

    # "next week" — assume next Monday
    if "next week" in tr or "next month" in tr:
        d = _days_until_weekday("monday")
        return (today + datetime.timedelta(days=d)).isoformat()

    # "by <dayname>" — e.g. "by friday"
    for name in DAY_NAMES:
        if tr.endswith(name) or tr.startswith(name):
            d = _days_until_weekday(name)
            return (today + datetime.timedelta(days=d)).isoformat()

    # "on <date>" — best-effort date parse
    m = re.search(r"(\w+)\s+(\d{1,2})(?:st|nd|rd|th)?", tr)
    if m:
        try:
            month_str, day_str = m.groups()
            # Parse month name
            dt = datetime.datetime.strptime(f"{month_str} {day_str}", "%B %d")
            dt = dt.replace(year=today.year)
            if dt.date() < today:
                dt = dt.replace(year=today.year + 1)
            return dt.date().isoformat()
        except ValueError:
            pass

    # "in N days/weeks"
    m = re.search(r"in\s+(\d+)\s+(day|days|week|weeks)", tr)
    if m:
        num = int(m.group(1))
        unit = m.group(2)
        if unit in ("week", "weeks"):
            num *= 7
        return (today + datetime.timedelta(days=num)).isoformat()

    # Fallback: vague future — check in 3 days
    return (today + datetime.timedelta(days=3)).isoformat()


def _extract_commitments(text: str) -> list[dict]:
    """Extract potential commitments from a line of text.

    Returns list of {text, check_date} dicts.
    """
    results = []
    for pattern in COMMITMENT_PATTERNS:
        for m in pattern.finditer(text):
            full = m.group(0).strip()
            try:
                action = m.group(1).strip()
                time_ref = m.group(2).strip()
            except IndexError:
                action = full
                time_ref = ""
            check_date = _resolve_check_date(time_ref) if time_ref else None
            check_date = check_date or (datetime.date.today() + datetime.timedelta(days=1)).isoformat()
            results.append({
                "text": full.strip('.,;:!?'),
                "check_date": check_date,
                "time_ref": time_ref,
            })
    return results


def cmd_scan(args):
    """Scan recent session messages for commitments."""
    days = args.days
    cutoff = datetime.datetime.now() - datetime.timedelta(days=days)
    cutoff_ts = cutoff.timestamp()

    conn = get_conn()
    session_conn = sqlite3.connect(SESSION_DB)
    session_conn.row_factory = sqlite3.Row

    # Get recent sessions
    sessions = session_conn.execute(
        "SELECT id, title, started_at FROM sessions WHERE started_at >= ? ORDER BY started_at DESC",
        (cutoff_ts,)
    ).fetchall()

    found_total = 0
    new_total = 0
    details = []

    for sess in sessions:
        msgs = session_conn.execute(
            "SELECT id, role, content, timestamp FROM messages WHERE session_id = ? AND role = 'user' ORDER BY timestamp",
            (sess["id"],)
        ).fetchall()

        for msg in msgs:
            if not msg["content"]:
                continue
            commitments = _extract_commitments(msg["content"])
            if not commitments:
                continue

            found_total += len(commitments)

            for cmt in commitments:
                # Dedup: check if similar commitment already exists from same session
                existing = conn.execute(
                    "SELECT id FROM commitments WHERE session_id = ? AND commitment_text = ? AND status = 'pending'",
                    (sess["id"], cmt["text"])
                ).fetchone()
                if existing:
                    continue

                conn.execute(
                    """INSERT INTO commitments (commitment_text, check_date, session_id, message_id, session_title)
                       VALUES (?, ?, ?, ?, ?)""",
                    (cmt["text"], cmt["check_date"], sess["id"], msg["id"], sess["title"])
                )
                new_total += 1
                details.append({
                    "text": cmt["text"],
                    "check": cmt["check_date"],
                    "session": sess["title"] or sess["id"][:12],
                })

    conn.commit()

    # Log the scan
    conn.execute(
        "INSERT INTO extraction_log (found, new, detail) VALUES (?, ?, ?)",
        (found_total, new_total, json.dumps(details, indent=2) if details else "")
    )
    conn.commit()

    if details:
        print(f"🔍 Scanned {len(sessions)} sessions (last {days}d): {found_total} hits, {new_total} new commitments")
        for d in details:
            print(f"   • [{d['check']}] \"{d['text']}\"  ← {d['session']}")
    else:
        print(f"🔍 Scanned {len(sessions)} sessions (last {days}d): no new commitments found")

    conn.close()
    session_conn.close()


# ─── Checking ─────────────────────────────────────────────────────────────────

def cmd_check(args):
    """Check pending commitments whose check_date has passed.

    This is the 'ping' mode — intended to be run by a daily cron job.
    Outputs a human-readable message for delivery.
    """
    today = datetime.date.today().isoformat()
    conn = get_conn()

    due = conn.execute(
        "SELECT * FROM commitments WHERE status = 'pending' AND check_date <= ? ORDER BY check_date",
        (today,)
    ).fetchall()

    if not due:
        print("✅ No commitments due today. All clear.")
        conn.close()
        return

    # Check if this is an interactive run (user at terminal) or cron (no stdin)
    interactive = sys.stdin.isatty() and not args.report

    print(f"⏰ Commitment check — {len(due)} due:")
    print()

    for row in due:
        delay = (datetime.date.today() - datetime.datetime.strptime(row["check_date"], "%Y-%m-%d").date()).days
        overdue = f" ({delay}d overdue)" if delay > 0 else ""

        print(f"  [#{row['id']}]{overdue}")
        print(f"  You said: \"{row['commitment_text']}\"")
        if row["session_title"]:
            print(f"  Context: {row['session_title']}")
        print(f"  Check date: {row['check_date']}")
        print()

        if interactive:
            print("  Did it happen?")
            print("    [k] Kept / Done ✅")
            print("    [b] Broken / Didn't happen ❌")
            print("    [a] Acknowledged (saw this, moving on) 👀")
            print("    [s] Skip (check again later)")
            try:
                choice = input("  → ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                choice = "s"

            if choice in ("k", "kept"):
                conn.execute(
                    "UPDATE commitments SET status = 'kept', outcome = 'kept', checked_at = datetime('now','localtime') WHERE id = ?",
                    (row["id"],)
                )
                print("  ✅ Marked as kept.\n")
            elif choice in ("b", "broken"):
                conn.execute(
                    "UPDATE commitments SET status = 'broken', outcome = 'broken', checked_at = datetime('now','localtime') WHERE id = ?",
                    (row["id"],)
                )
                print("  ❌ Marked as broken.\n")
            elif choice in ("a", "acknowledged"):
                conn.execute(
                    "UPDATE commitments SET status = 'acknowledged', outcome = 'acknowledged', checked_at = datetime('now','localtime') WHERE id = ?",
                    (row["id"],)
                )
                print("  👀 Acknowledged.\n")
            else:
                print("  ⏭️  Skipped.\n")
        else:
            # Non-interactive — print a ping message
            session_ref = f" (from: {row['session_title']})" if row["session_title"] else ""
            print(f"  ⏰ You said: \"{row['commitment_text']}\"{session_ref}")
            print(f"     Check date was {row['check_date']}{overdue}")
            print()

    if interactive is False:
        print(f"  ────")
        print(f"  Resolve with: commitment_tracker.py resolve <id> --kept|--broken|--acknowledged")
        print(f"  Or: commitment_tracker.py check (interactive)")
    elif args.report:
        # Deliberately empty — just the commitment list above is the report
        pass

    conn.commit()
    conn.close()


# ─── Manual Add ───────────────────────────────────────────────────────────────

def cmd_add(args):
    """Manually add a commitment."""
    conn = get_conn()
    conn.execute(
        "INSERT INTO commitments (commitment_text, check_date) VALUES (?, ?)",
        (args.text, args.check)
    )
    conn.commit()
    print(f"✅ Added commitment: \"{args.text}\" → check {args.check}")
    conn.close()


# ─── Resolve ──────────────────────────────────────────────────────────────────

def cmd_resolve(args):
    """Resolve a commitment non-interactively (for cron post-processing)."""
    status = None
    if args.kept:
        status = "kept"
    elif args.broken:
        status = "broken"
    elif args.acknowledged:
        status = "acknowledged"

    conn = get_conn()
    row = conn.execute("SELECT * FROM commitments WHERE id = ?", (args.id,)).fetchone()
    if not row:
        print(f"❌ No commitment with id {args.id}")
        conn.close()
        return

    conn.execute(
        "UPDATE commitments SET status = ?, outcome = ?, checked_at = datetime('now','localtime') WHERE id = ?",
        (status, status, args.id)
    )
    conn.commit()
    emoji = {"kept": "✅", "broken": "❌", "acknowledged": "👀", "expired": "⏳"}.get(status, "➡️")
    print(f"{emoji} Commitment #{args.id} marked as {status}: \"{row['commitment_text']}\"")
    conn.close()


# ─── Status ───────────────────────────────────────────────────────────────────

def cmd_status(args):
    """Show all commitments."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM commitments ORDER BY check_date DESC"
    ).fetchall()

    if not rows:
        print("📭 No commitments yet.")
        conn.close()
        return

    # Group by status
    groups = {"pending": [], "kept": [], "broken": [], "acknowledged": [], "expired": []}
    for r in rows:
        groups.setdefault(r["status"], []).append(r)

    print(f"📋 Commitment Tracker — {len(rows)} total\n")

    for status, label, emoji in [
        ("pending", "Pending", "⏰"),
        ("kept", "Kept", "✅"),
        ("broken", "Broken", "❌"),
        ("acknowledged", "Acknowledged", "👀"),
        ("expired", "Expired", "⏳"),
    ]:
        items = groups.get(status, [])
        if not items:
            continue
        print(f"  {emoji} {label} ({len(items)}):")
        for r in items:
            ctx = f"  ← {r['session_title']}" if r["session_title"] else ""
            chk = f" [due: {r['check_date']}]" if status == "pending" else ""
            print(f"     #{r['id']} \"{r['commitment_text']}\"{chk}{ctx}")
        print()

    # Summary
    pending = len(groups.get("pending", []))
    kept = len(groups.get("kept", []))
    broken = len(groups.get("broken", []))
    total_resolved = kept + broken + len(groups.get("acknowledged", []))
    print(f"  📊 Pending: {pending}  |  Kept: {kept}  |  Broken: {broken}  |  Resolved: {total_resolved}")

    conn.close()


# ─── Reconcile (non-interactive check for cron) ───────────────────────────────

def cmd_reconcile(args):
    """Non-interactive resolve for cron use. Marks kept or broken based on arg."""
    status = "kept" if args.kept else "broken"
    conn = get_conn()
    row = conn.execute("SELECT * FROM commitments WHERE id = ?", (args.id,)).fetchone()
    if not row:
        print(f"❌ No commitment with id {args.id}")
        conn.close()
        return
    conn.execute(
        "UPDATE commitments SET status = ?, outcome = ?, checked_at = datetime('now','localtime') WHERE id = ?",
        (status, status, args.id)
    )
    conn.commit()
    print(f"{'✅' if status == 'kept' else '❌'} Commitment #{args.id}: \"{row['commitment_text']}\" → {status}")
    conn.close()


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Commitment Tracker — declared-intention → outcome monitor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              commitment_tracker.py scan --days 7
              commitment_tracker.py check
              commitment_tracker.py add "I'll submit the paper by Friday" --check 2026-06-05
              commitment_tracker.py resolve 3 --kept
              commitment_tracker.py status
        """),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # scan
    p = sub.add_parser("scan", help="Scan session transcripts for commitments")
    p.add_argument("--days", type=int, default=7, help="Days of history to scan (default: 7)")

    # check
    p = sub.add_parser("check", help="Check pending commitments due today")
    p.add_argument("--report", action="store_true", help="Non-interactive report mode for cron delivery")

    # add
    p = sub.add_parser("add", help="Manually add a commitment")
    p.add_argument("text", help="Commitment text")
    p.add_argument("--check", "-c", required=True, help="Check date (YYYY-MM-DD)")

    # resolve
    p = sub.add_parser("resolve", help="Resolve a commitment non-interactively")
    p.add_argument("id", type=int, help="Commitment ID")
    p.add_argument("--kept", action="store_true", help="Mark as kept/done")
    p.add_argument("--broken", action="store_true", help="Mark as broken/not done")
    p.add_argument("--acknowledged", action="store_true", help="Mark as acknowledged (seen, moving on)")

    # status
    p = sub.add_parser("status", help="Show all commitments")

    # reconcile (legacy, same as resolve)
    p = sub.add_parser("reconcile", help="Non-interactive resolve for cron")
    p.add_argument("id", type=int, help="Commitment ID")
    p.add_argument("--kept", action="store_true")
    p.add_argument("--broken", action="store_true")

    args = parser.parse_args()

    # Route
    if args.command == "scan":
        cmd_scan(args)
    elif args.command == "check":
        cmd_check(args)
    elif args.command == "add":
        cmd_add(args)
    elif args.command == "resolve":
        if not (args.kept or args.broken or args.acknowledged):
            print("❌ Specify --kept, --broken, or --acknowledged")
            sys.exit(1)
        cmd_resolve(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "reconcile":
        if not (args.kept or args.broken):
            print("❌ Specify --kept or --broken")
            sys.exit(1)
        cmd_reconcile(args)


if __name__ == "__main__":
    main()
