#!/usr/bin/env python3
"""Query the knowledge base and show exactly what was filtered out.

    python scripts/ask.py "what is the home office stipend?"
    python scripts/ask.py --suite          # run the built-in demo questions
    python scripts/ask.py --tenant acme --role employee "what are the salary bands?"

Without --tenant/--role there is no permission filter (old behaviour).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.access import UserContext  # noqa: E402
from src.kb import KnowledgeBase  # noqa: E402

SUITE = [
    "What is the home office stipend?",
    "How many days per week can I work remotely?",
    "How much PTO do I accrue and can I carry it over?",
    "Can we deploy to production on a Friday?",
    "What is the minimum test coverage to merge?",
    "How often are database backups taken?",
    "How are production database credentials shared?",
]


def show(kb, question, user=None):
    res = kb.ask(question, user=user)
    print(f"\nQ: {question}")
    print(f"A: {res.answer}\n")
    print("  Context used:")
    for rc in res.used:
        print(f"    + {rc.doc_id} {rc.version} {rc.breadcrumb}  "
              f"sim={rc.similarity:.3f} fresh={rc.freshness:.3f} score={rc.score:.3f}")
    if res.filtered_out:
        print("  Filtered out (superseded or decayed below floor):")
        for rc in res.filtered_out[:6]:
            print(f"    - {rc.doc_id} {rc.version} {rc.breadcrumb}  "
                  f"sim={rc.similarity:.3f} status={rc.status}")
    if res.blocked_ids:
        print(f"  Hidden by permissions: {len(res.blocked_ids)} chunk(s)")
    print("-" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("--suite", action="store_true")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--tenant")
    ap.add_argument("--role")
    ap.add_argument("--user", default="cli")
    args = ap.parse_args()
    if bool(args.tenant) != bool(args.role):
        ap.error("--tenant and --role go together")
    user = UserContext(args.tenant, args.role, args.user) if args.tenant else None

    kb = KnowledgeBase(offline=True if args.offline else None)
    print("Backends:", kb.backends())

    if args.suite:
        for q in SUITE:
            show(kb, q, user)
    elif args.question:
        show(kb, args.question, user)
    else:
        ap.error("give a question or use --suite")


if __name__ == "__main__":
    main()
