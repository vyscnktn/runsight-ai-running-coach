#!/usr/bin/env python3
"""Sanitise an n8n workflow export so it can be published.

Usage:
    python scripts/sanitize_workflow.py raw_export.json n8n/runsight_workflow.json

What it removes / replaces (the workflow structure and all logic stay intact):
  * credential IDs            -> "REPLACE_ME" (credential *names* are kept)
  * webhookId of trigger nodes -> removed (n8n regenerates it on import)
  * Google Sheet / Drive IDs and URLs -> placeholders
  * e-mail addresses          -> you@example.com
  * the Lambda Function URL   -> placeholder
  * raw.githubusercontent prompt URLs -> this repository
  * pinData (execution samples with real GPS/health data) and meta.instanceId

After running it, the script scans the result and exits non-zero if anything
that still looks like an ID, e-mail address or key remains.
"""
import json
import re
import sys

REPO_RAW = "https://raw.githubusercontent.com/vyscnktn/runsight-ai-running-coach/main/prompts/"
LAMBDA_PLACEHOLDER = "https://<your-lambda-function-url>.lambda-url.eu-central-1.on.aws/"


def sanitize(text: str) -> str:
    d = json.loads(text)
    d.pop("pinData", None)
    d.get("meta", {}).pop("instanceId", None)
    for node in d["nodes"]:
        node.pop("webhookId", None)
        for cred in (node.get("credentials") or {}).values():
            cred["id"] = "REPLACE_ME"
    s = json.dumps(d, ensure_ascii=False, indent=2)

    # Google Sheets / Drive: replace every ID that appears in a Google URL,
    # then the same ID wherever it appears bare (documentId fields etc.).
    ids = set(re.findall(r"docs\.google\.com/(?:spreadsheets|document)/d/([\w-]{20,})", s))
    ids |= set(re.findall(r"drive\.google\.com/drive/folders/([\w-]{20,})", s))
    for n, i in enumerate(sorted(ids), 1):
        s = s.replace(i, f"<GOOGLE_FILE_ID_{n}>")

    s = re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+", "you@example.com", s)
    s = re.sub(r"https://[a-z0-9]+\.lambda-url\.[a-z0-9-]+\.on\.aws/?", LAMBDA_PLACEHOLDER, s)
    s = re.sub(
        r"https://raw\.githubusercontent\.com/[^/\"]+/[^/\"]+/[^/\"]+/(?:[^/\"]+/)*?(\w+_prompt\.md)",
        lambda m: REPO_RAW + m.group(1),
        s,
    )
    return s + "\n"


def leftovers(s: str):
    checks = {
        "google id": r"(?<![\w-])[\w-]{40,}(?![\w-])",
        "e-mail": r"[\w.+-]+@(?!example\.com)[\w-]+\.[\w.-]+",
        "lambda url": r"https://(?!<)[a-z0-9]{20,}\.lambda-url",
        "bearer/key": r"(?i)(bearer\s+\w{10,}|sk-[A-Za-z0-9]{20,}|AIza[\w-]{20,})",
    }
    return {k: re.findall(p, s)[:3] for k, p in checks.items() if re.search(p, s)}


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    out = sanitize(open(src, encoding="utf-8").read())
    problems = leftovers(out)
    open(dst, "w", encoding="utf-8").write(out)
    if problems:
        print("WARNING - review manually:", problems)
        sys.exit(1)
    print(f"wrote {dst} ({len(json.loads(out)['nodes'])} nodes)")
