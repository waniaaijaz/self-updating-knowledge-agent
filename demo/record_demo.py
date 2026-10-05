#!/usr/bin/env python3
"""Record the Streamlit demo with Playwright.

Starts `streamlit run app.py` against a throw-away storage dir, resets the KB,
uploads demo/docs/demo_policy_v1.md then v2 through the UI ("upload my own"),
and then walks the Knowledge graph, Ask and Review tabs.

Outputs (git-ignored): demo/raw/demo_raw.webm and demo/raw/captions.json.
Fails loudly if a tab is empty or a step errors -- nothing is faked.

    python demo/record_demo.py
"""

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "demo"
RAW = DEMO / "raw"
STORAGE = DEMO / ".storage"
PORT = 8599
URL = f"http://localhost:{PORT}"
DOC_ID = "alder_ridge_security"
QUESTION = "How often must passwords be changed?"

captions: list[dict] = []
t0 = 0.0


def now() -> float:
    return time.time() - t0


def caption(text: str, hold: float):
    """Show a caption for `hold` seconds of video time (also blocks for that long)."""
    start = now()
    captions.append({"start": round(start, 2), "end": round(start + hold, 2), "text": text})
    time.sleep(hold)


def start_server() -> subprocess.Popen:
    shutil.rmtree(STORAGE, ignore_errors=True)
    env = {**os.environ, "KB_STORAGE_DIR": str(STORAGE)}
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py",
         "--server.port", str(PORT), "--server.headless", "true",
         "--browser.gatherUsageStats", "false"],
        cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    for _ in range(120):
        try:
            if urllib.request.urlopen(f"{URL}/_stcore/health", timeout=1).status == 200:
                return proc
        except Exception:
            time.sleep(1)
    proc.terminate()
    raise RuntimeError("Streamlit did not become ready in 120s")


def stop_server(proc: subprocess.Popen):
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        proc.terminate()


def wait_ready(page, attempts: int = 3):
    """Wait for the app shell. Streamlit occasionally loads a page whose sidebar
    never appears; a reload fixes that, so retry before giving up."""
    for attempt in range(attempts):
        try:
            expect(page.get_by_text("Ingest a document version")).to_be_visible(timeout=40_000)
            expect(page.get_by_role("button", name="Reset knowledge base")).to_be_visible(timeout=20_000)
            return
        except AssertionError:
            if attempt == attempts - 1:
                raise
            page.reload()


def idle(page):
    """Wait until Streamlit is not mid-rerun (its status widget shows 'Stop' while running)."""
    expect(page.get_by_text("Stop", exact=True)).to_have_count(0, timeout=120_000)
    page.wait_for_timeout(300)


def set_field(page, label: str, value: str):
    box = page.get_by_label(label, exact=True)
    box.fill(value)
    box.press("Enter")
    page.wait_for_timeout(300)


def ingest(page, version: str, date: str):
    f = DEMO / "docs" / f"demo_policy_{version}.md"
    page.get_by_role("tab", name="Ingest").click()
    idle(page)
    for rm in page.get_by_role("button", name="Remove ").all():
        rm.click()  # single-file uploader: drop the previous version first
        idle(page)
    page.locator('input[type="file"]').set_input_files(str(f))
    expect(page.get_by_text(f.name)).to_be_visible(timeout=30_000)
    idle(page)
    set_field(page, "Document ID", DOC_ID)
    set_field(page, "Version", version)
    set_field(page, "Effective date", date)
    page.wait_for_timeout(600)
    page.get_by_role("button", name="Ingest", exact=True).click()
    expect(page.get_by_text("Duplicates skipped")).to_be_visible(timeout=120_000)
    page.get_by_text("Duplicates skipped").scroll_into_view_if_needed()


def main():
    global t0
    RAW.mkdir(exist_ok=True)
    for old in RAW.glob("*.webm"):
        old.unlink()
    server = start_server()
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            # Warm-up context (not recorded): load the models/KB once so the
            # recorded session does not show a start-up spinner.
            warm = browser.new_context(viewport={"width": 1280, "height": 720})
            wp = warm.new_page()
            wp.goto(URL)
            wait_ready(wp)
            warm.close()

            ctx = browser.new_context(
                viewport={"width": 1280, "height": 720},
                record_video_dir=str(RAW),
                record_video_size={"width": 1280, "height": 720},
            )
            page = ctx.new_page()
            t0 = time.time()
            page.goto(URL)
            wait_ready(page)
            trim = max(now() - 0.3, 0.0)  # drop the page-load lead-in from the final cut

            # --- reset
            caption("Start from an empty knowledge base", 2.0)
            page.get_by_role("button", name="Reset knowledge base").click()
            wait_ready(page)
            expect(page.get_by_text("Active").first).to_be_visible()
            page.wait_for_timeout(500)

            # --- v1
            caption("Ingest policy v1 (uploaded file)", 1.5)
            ingest(page, "v1", "2024-01-15")
            caption("v1: 8 clauses inserted, nothing to compare against", 4.5)

            # --- v2
            page.evaluate("window.scrollTo(0, 0)")
            caption("Ingest policy v2 as a new version", 1.5)
            ingest(page, "v2", "2026-01-15")
            expect(page.get_by_text("conflict(s) routed to human review")).to_be_visible()
            caption("v2: 3 clauses superseded, 2 sent to review, 3 unchanged skipped", 5.5)

            # --- graph
            # The sidebar counters are drawn before ingest runs, so they lag one
            # rerun behind. Reload the page (a page reload).
            page.reload()
            wait_ready(page)
            idle(page)
            page.get_by_role("tab", name="Knowledge graph").click()
            expect(page.get_by_text("Audit trail")).to_be_visible(timeout=30_000)
            expect(page.locator("svg g.node")).to_have_count(14, timeout=30_000)
            expect(page.locator("svg g.edge")).to_have_count(3)
            caption("Knowledge graph: each v2 clause points to the v1 clause it supersedes (red)", 6.0)
            page.get_by_text("Audit trail").scroll_into_view_if_needed()
            caption("Audit trail records the score for each decision", 4.0)

            # --- ask
            cut = None
            page.get_by_role("tab", name="Ask").click()
            q = page.get_by_label("Question", exact=True)
            q.fill("")
            q.click()
            page.keyboard.type(QUESTION, delay=60)
            q.press("Enter")
            caption("Ask about a superseded clause (password rotation)", 1.5)
            page.get_by_role("button", name="Ask", exact=True).click()
            asked = now()
            expect(page.get_by_text("Answer", exact=True)).to_be_visible(timeout=90_000)
            answered = now()
            # LLM latency varies (5-20s). Keep ~1.5s of the spinner and let
            # build_video.py cut the rest, labelled in the captions.
            if answered - asked > 2.5:
                cut = [asked + 1.5, answered - 0.3]
                captions.append({"start": round(asked, 2), "end": round(cut[0], 2),
                                 "text": f"Waiting for Gemini ({answered - asked:.0f}s, shortened in the video)"})
            # The demo is meant to show a real LLM answer: refuse to record the
            # extractive fallback (no key, bad key or model error) as if it were one.
            panel = page.get_by_role("tabpanel", name="Ask").inner_text()
            answer = panel.split("Answer", 1)[1].split("generated by", 1)[0]
            generator = panel.split("generated by", 1)[1].split("\n", 1)[0].strip()
            if generator == "extractive" or "extractive" in answer:
                raise RuntimeError(
                    f"LLM did not answer (generator={generator!r}). Put GEMINI_API_KEY "
                    "in .env and check the key/model before recording.")
            if "180" not in answer:
                raise RuntimeError(f"LLM answer does not state the current rule: {answer!r}")
            caption("The LLM answers from active clauses only: 180 days", 5.0)
            page.get_by_text("Retrieval detail").click()
            expect(page.get_by_text("Filtered out", exact=True)).to_be_visible()
            page.mouse.move(800, 400)
            page.mouse.wheel(0, 380)  # bring the "Filtered out" column above the caption
            page.wait_for_timeout(400)
            caption("The superseded v1 clause (90 days) is filtered out before ranking", 5.0)

            # --- review
            page.get_by_role("tab", name="Review queue").click()
            expect(page.get_by_text("Existing (still active)").first).to_be_visible()
            assert page.get_by_text("Queue is empty.").count() == 0
            caption("Review queue: ambiguous changes wait for a human; the old rule stays active", 5.5)
            page.mouse.move(800, 400)
            page.mouse.wheel(0, 330)  # show the second pending item
            page.wait_for_timeout(400)
            caption("Approve or reject each item", 4.0)

            ctx.close()  # flushes the video
            browser.close()
        vids = sorted(RAW.glob("*.webm"), key=lambda p: p.stat().st_mtime)
        final = RAW / "demo_raw.webm"
        if final.exists():
            final.unlink()
        vids[-1].rename(final)
        (RAW / "captions.json").write_text(json.dumps(
            {"trim": round(trim, 2),
             "cut": [round(x - trim, 2) for x in cut] if cut else None,
             "captions": [{**c, "start": round(c["start"] - trim, 2), "end": round(c["end"] - trim, 2)}
                          for c in captions]}, indent=2), encoding="utf-8")
        print(f"Recorded {final} ({now():.1f}s of video), {len(captions)} captions")
    finally:
        stop_server(server)


if __name__ == "__main__":
    main()
