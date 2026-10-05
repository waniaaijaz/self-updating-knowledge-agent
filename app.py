#!/usr/bin/env python3
"""Streamlit demo UI.

    streamlit run app.py

Four tabs: Ingest (upload a version, watch conflicts get flagged),
Ask (query + see what got filtered out), Knowledge graph (SUPERSEDES
edges), Review (HITL queue for the ambiguous cases).
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config  # noqa: E402
from src.kb import KnowledgeBase  # noqa: E402

st.set_page_config(page_title="Self-Updating Knowledge Agent", layout="wide")


@st.cache_resource(show_spinner=False)
def load_kb(offline: bool = True):
    return KnowledgeBase(offline=offline)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("System")
    mode = st.radio(
        "Backend Mode",
        ["Fast / Offline (Instant)", "Transformer (DeBERTa + MiniLM)"],
        index=0 if config.OFFLINE else 0,
        help="Fast mode runs deterministic heuristics instantly without waiting for 600MB model downloads."
    )
    is_offline = "Fast" in mode

with st.spinner(f"Initializing Knowledge Base ({'Fast heuristic' if is_offline else 'Loading transformer weights'})..."):
    kb = load_kb(offline=is_offline)

with st.sidebar:
    st.caption("---")
    for key, value in kb.backends().items():
        st.caption(f"**{key}** · `{value}`")

    st.divider()
    api_key_input = st.text_input(
        "Gemini API Key (free)",
        value=config.GEMINI_API_KEY,
        type="password",
        help="Get a free key from https://aistudio.google.com/app/apikey"
    )
    if api_key_input != config.GEMINI_API_KEY:
        config.GEMINI_API_KEY = api_key_input
        kb.query_engine._llm = None  # Reset cached LLM

    if config.GEMINI_API_KEY and not config.GEMINI_API_KEY.startswith(("AIza", "AQ.")):
        st.sidebar.caption("⚠️ *Key does not start with `AIzaSy...`. Get a free key at [Google AI Studio](https://aistudio.google.com/app/apikey)*")


    stats = kb.stats()
    c1, c2 = st.columns(2)
    c1.metric("Active", stats["active"])
    c2.metric("Superseded", stats["superseded"])
    c1.metric("Pending", stats["pending_review"])
    c2.metric("Edges", stats["supersedes_edges"])

    st.divider()
    thr_auto = config.AUTO_THRESHOLD_OVERRIDE or kb.detector.auto_threshold
    thr_hitl = config.HITL_THRESHOLD_OVERRIDE or kb.detector.hitl_threshold
    st.caption(f"Auto-supersede ≥ **{thr_auto:.2f}**")
    st.caption(f"Human review ≥ **{thr_hitl:.2f}**")
    st.caption(f"Freshness half-life **{config.DECAY_HALF_LIFE_DAYS:.0f} days**")

    if st.button("Reset knowledge base", type="secondary"):
        kb.reset()
        st.rerun()

tab_ingest, tab_ask, tab_graph, tab_review = st.tabs(
    ["Ingest", "Ask", "Knowledge graph", "Review queue"]
)

# ------------------------------------------------------------------- ingest
with tab_ingest:
    st.subheader("Ingest a document version")

    demo_files = sorted(config.DATA_DIR.glob("*.md"))
    col_a, col_b = st.columns([2, 1])
    with col_a:
        choice = st.selectbox(
            "Bundled document", ["— upload my own —"] + [f.name for f in demo_files]
        )
        uploaded = st.file_uploader("Markdown file", type=["md", "txt"]) \
            if choice == "— upload my own —" else None
    with col_b:
        doc_id = st.text_input("Document ID", value="hr_handbook")
        version = st.text_input("Version", value="v1")
        timestamp = st.text_input("Effective date", value="2024-01-15")
        force_reingest = st.checkbox(
            "Bypass duplicate hash gate",
            value=False,
            help="Re-run this exact file through the pipeline even if its "
                 "chunks were already ingested. Useful when re-testing the "
                 "same version without hitting Reset first.",
        )

    if st.button("Ingest", type="primary"):
        if uploaded is not None:
            text = uploaded.read().decode("utf-8")
        elif choice != "— upload my own —":
            text = (config.DATA_DIR / choice).read_text(encoding="utf-8")
        else:
            st.error("Pick a bundled document or upload a file.")
            st.stop()

        with st.spinner("Chunking, embedding, running NLI…"):
            result = kb.ingest_markdown(
                text, doc_id, version, timestamp, verbose=False, force_reingest=force_reingest
            )

        s = result["summary"]
        if result["chunks"] == 0:
            st.warning(
                "No chunks were produced — the file looks empty after "
                "whitespace is stripped. Add some text and try again."
            )
        else:
            m = st.columns(4)
            m[0].metric("Chunks", result["chunks"])
            m[1].metric("Inserted", s["INSERT"])
            m[2].metric("Superseded", s["SUPERSEDE"], delta_color="inverse")
            m[3].metric("Duplicates skipped", s["DUPLICATE"])
            if s["HITL"]:
                st.warning(f"{s['HITL']} conflict(s) routed to human review.")

        st.code("\n".join(result["log"]), language="text")

# ---------------------------------------------------------------------- ask
with tab_ask:
    st.subheader("Query the current truth")
    question = st.text_input("Question", value="What is the home office stipend?")

    if st.button("Ask", type="primary") and question:
        with st.spinner("Retrieving…"):
            res = kb.ask(question)

        st.markdown("### Answer")
        st.write(res.answer)
        st.caption(f"generated by `{res.generator}`")

        with st.expander(f"Retrieval detail — {len(res.used)} used, {len(res.filtered_out)} filtered out"):
            left, right = st.columns(2)
            with left:
                st.markdown("**Context used**")
                for rc in res.used:
                    st.markdown(
                        f"- **{rc.doc_id} {rc.version}** · {rc.breadcrumb}  \n"
                        f"  sim `{rc.similarity:.3f}` · fresh `{rc.freshness:.3f}` "
                        f"· score `{rc.score:.3f}`"
                    )
            with right:
                st.markdown("**Filtered out**")
                if not res.filtered_out:
                    st.caption("Nothing was filtered for this query.")
                for rc in res.filtered_out[:8]:
                    st.markdown(
                        f"- **{rc.doc_id} {rc.version}** · {rc.breadcrumb}  \n"
                        f"  status `{rc.status}` · sim `{rc.similarity:.3f}` "
                        f"· fresh `{rc.freshness:.3f}`"
                    )

# -------------------------------------------------------------------- graph
with tab_graph:
    st.subheader("Knowledge graph")
    nodes = kb.graph.nodes()
    edges = kb.graph.edges()

    # Status -> color, applied from the node's real `status` field rather
    # than from which side of an edge it happens to sit on. That's what
    # makes PENDING_REVIEW nodes (no edge yet) show up at all, and it keeps
    # an ACTIVE node green even if it later appears as an edge's target in
    # some other, unrelated lineage.
    STATUS_COLOR = {
        "ACTIVE": "#d7f5dd",          # green
        "SUPERSEDED": "#f8d7da",      # red
        "PENDING_REVIEW": "#fff3cd",  # yellow
    }
    STATUS_BORDER = {
        "ACTIVE": "#2e7d32",
        "SUPERSEDED": "#c0392b",
        "PENDING_REVIEW": "#b8860b",
    }

    if not nodes:
        st.caption("Nothing ingested yet.")
    else:
        try:
            import graphviz

            dot = graphviz.Digraph()
            dot.attr(rankdir="TB", bgcolor="transparent", nodesep="0.25", ranksep="0.6")
            dot.attr("node", fontname="Helvetica", fontsize="12", margin="0.06,0.04")
            dot.attr("edge", fontname="Helvetica", fontsize="10")

            # Chunk ids contain "::", which graphviz parses as a node:port
            # separator in edges and emits invalid DOT. Use plain ids.
            gid = {node["id"]: f"n{i}" for i, node in enumerate(nodes)}

            for node in nodes:
                status = node.get("status", "ACTIVE")
                dot.node(
                    gid[node["id"]],
                    f"{node.get('version','?')} · {node.get('section_path','?')}\n{status.split('_')[0].title()}",
                    shape="box", style="filled",
                    fillcolor=STATUS_COLOR.get(status, "#e0e0e0"),
                    color=STATUS_BORDER.get(status, "#666666"),
                )

            for edge in edges:
                if edge["source"] not in gid or edge["target"] not in gid:
                    continue
                dot.edge(gid[edge["source"]], gid[edge["target"]],
                         label=f"  {edge.get('nli_score', 0):.2f}", color="#c0392b")

            st.graphviz_chart(dot, use_container_width=True)

            legend = "  ".join(
                f"{icon} {status.replace('_', ' ').title()}"
                for status, icon in [("ACTIVE", "🟩"), ("SUPERSEDED", "🟥"), ("PENDING_REVIEW", "🟨")]
            )
            st.caption(legend)
        except Exception as exc:  # noqa: BLE001
            st.caption(f"(graphviz unavailable: {exc})")

    if not edges:
        st.caption("No supersession edges yet. Ingest a v1 then a v2 document.")
    else:

        st.markdown("#### Audit trail")
        st.dataframe(
            [
                {
                    "supersedes": e["source"].split("::", 1)[-1],
                    "replaced": e["target"].split("::", 1)[-1],
                    "nli_score": e.get("nli_score"),
                    "reason": e.get("reason"),
                    "detected_at": e.get("detected_at", "")[:19],
                }
                for e in edges
            ],
            use_container_width=True,
        )

# ------------------------------------------------------------------- review
with tab_review:
    st.subheader("Human-in-the-loop queue")
    st.caption(
        "Conflicts in the ambiguous confidence band land here. Until a human "
        "decides, the existing rule stays active — the system does not delete "
        "policy on a coin flip."
    )

    open_items = kb.queue.open_items()
    if not open_items:
        st.success("Queue is empty.")

    for item in open_items:
        with st.container(border=True):
            st.markdown(
                f"**{item['review_id']}** · section `{item['section_path']}` · "
                f"contradiction score `{item['nli']['contradiction_score']}` · "
                f"similarity `{item['similarity']}`"
            )
            c1, c2 = st.columns(2)
            c1.markdown("**Existing (still active)**")
            c1.warning(item["old_text"])
            c2.markdown("**Incoming**")
            c2.info(item["new_text"])

            b1, b2, _ = st.columns([1, 1, 4])
            if b1.button("Approve", key=f"ok-{item['review_id']}", type="primary"):
                kb.queue.resolve(item["review_id"], True, kb.graph, kb.vectors)
                st.rerun()
            if b2.button("Reject", key=f"no-{item['review_id']}"):
                kb.queue.resolve(item["review_id"], False, kb.graph, kb.vectors)
                st.rerun()
