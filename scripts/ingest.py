#!/usr/bin/env python3
"""Ingest documents into the knowledge base.

    python scripts/ingest.py --demo              # ingest v1 then v2 for both doc pairs
    python scripts/ingest.py --reset --demo      # wipe first
    python scripts/ingest.py path/to/doc.md --doc-id hr --version v3 --timestamp 2027-01-01
    python scripts/ingest.py --access-demo       # two tenants, role-restricted clauses
    python scripts/ingest.py doc.md --doc-id hr --version v1 --timestamp 2024-01-01 --tenant acme --roles manager,hr
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.kb import KnowledgeBase  # noqa: E402
from src import config  # noqa: E402

DEMO = [
    ("hr_policy_v1.md", "hr_handbook", "v1", "2024-01-15"),
    ("devops_sop_v1.md", "devops_sop", "v1", "2024-03-01"),
    ("hr_policy_v2.md", "hr_handbook", "v2", "2026-01-05"),
    ("devops_sop_v2.md", "devops_sop", "v2", "2026-02-10"),
]

# Same doc_id in both tenants on purpose. Per-clause roles come from the
# <!-- roles: ... --> lines inside the files.
ACCESS_DEMO = [
    ("acme_policy_v1.md", "acme", "people_policy", "v1", "2024-02-01"),
    ("globex_policy_v1.md", "globex", "people_policy", "v1", "2024-03-01"),
    ("acme_policy_v2.md", "acme", "people_policy", "v2", "2026-01-10"),
    ("globex_policy_v2.md", "globex", "people_policy", "v2", "2026-02-01"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", help="markdown file to ingest")
    ap.add_argument("--doc-id")
    ap.add_argument("--version")
    ap.add_argument("--timestamp")
    ap.add_argument("--tenant", default="default", help="tenant the document belongs to")
    ap.add_argument("--roles", default="all", help="comma-separated roles allowed to see it")
    ap.add_argument("--demo", action="store_true", help="ingest the bundled corpus in order")
    ap.add_argument("--access-demo", action="store_true",
                    help="ingest the two-tenant permissions demo (acme / globex)")
    ap.add_argument("--reset", action="store_true", help="clear all stores first")
    ap.add_argument("--offline", action="store_true", help="force deterministic offline backends")
    args = ap.parse_args()

    kb = KnowledgeBase(offline=True if args.offline else None)
    print("Backends:", kb.backends())

    if args.reset:
        kb.reset()
        print("Stores cleared.\n")

    if args.demo or args.access_demo:
        if args.demo:
            for filename, doc_id, version, ts in DEMO:
                print(f"\n=== Ingesting {filename} ({doc_id} {version}, dated {ts}) ===")
                res = kb.ingest_file(config.DATA_DIR / filename, doc_id, version, ts)
                print(f"  -> {res['chunks']} chunks {res['summary']}")
        if args.access_demo:
            for filename, tenant, doc_id, version, ts in ACCESS_DEMO:
                print(f"\n=== Ingesting {filename} (tenant {tenant}, {doc_id} {version}, dated {ts}) ===")
                res = kb.ingest_file(config.DATA_DIR / filename, doc_id, version, ts, tenant_id=tenant)
                print(f"  -> {res['chunks']} chunks {res['summary']}")
    elif args.path:
        if not (args.doc_id and args.version and args.timestamp):
            ap.error("--doc-id, --version and --timestamp are required with a path")
        res = kb.ingest_file(args.path, args.doc_id, args.version, args.timestamp,
                             tenant_id=args.tenant, allowed_roles=args.roles)
        print(f"\n-> {res['chunks']} chunks {res['summary']}")
    else:
        ap.error("give a path or use --demo / --access-demo")

    print("\nKnowledge base state:", kb.stats())
    if kb.queue.open_items():
        print("\nOpen human reviews:")
        for item in kb.queue.open_items():
            print(f"  [{item['review_id']}] {item['section_path']} "
                  f"score={item['nli']['contradiction_score']}")
            print(f"      old: {item['old_text'][:90]}")
            print(f"      new: {item['new_text'][:90]}")


if __name__ == "__main__":
    main()
