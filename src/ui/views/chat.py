"""Assistant page: a question in, a grounded answer with verifiable sources out."""

import os
import time

import streamlit as st

from src.core.config import GEN_MODEL_NAME, GROQ_VERIFY_MODEL, active_generator
from src.rag.rag_query import answer_query, verify_answer
from src.ui.theme import empty_state, hero, source_card

ABSTENTION_MARKER = "Multiple candidate materials found"
USER_AVATAR = ":material/person:"
ASSISTANT_AVATAR = ":material/biotech:"
EXAMPLES = [
    "What causes anode degradation in Ni-YSZ SOFCs?",
    "What are the advantages of proton-conducting electrolytes?",
    "Which cathode materials show high oxygen reduction activity?",
    "What is the role of GDC as an interlayer in SOFC cathodes?",
]


@st.cache_data
def corpus_stats() -> dict:
    def count_lines(path: str) -> int:
        if not os.path.exists(path):
            return 0
        with open(path, encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())

    return {
        "papers": count_lines("data/sofc_corpus.jsonl"),
        "full_text": count_lines("data/full_papers.jsonl"),
    }


def verification_badge(text: str) -> None:
    verdict = text.strip().lower()
    if verdict.startswith("yes"):
        st.badge("Supported by sources", icon=":material/verified:", color="green")
    elif verdict.startswith("no"):
        st.badge("Not confirmed by sources", icon=":material/report:", color="orange")
    else:
        st.badge("Verification inconclusive", icon=":material/help:", color="gray")


def render_answer(msg: dict) -> None:
    if msg.get("abstained"):
        st.warning(msg["content"], icon=":material/call_split:")
    else:
        st.markdown(msg["content"])

    if msg.get("verification"):
        verification_badge(msg["verification"])
        reason = msg["verification"].strip()
        if reason.lower().rstrip(".") not in {"yes", "no"}:
            st.caption(f"Verifier: {reason}")

    meta = msg.get("meta")
    if meta:
        st.caption(" · ".join(meta))

    sources = msg.get("sources") or []
    if sources:
        with st.expander(f"Sources ({len(sources)})", icon=":material/library_books:"):
            for s in sources:
                source_card(s["rank"], s["title"], s["paper_id"], s["section"], s["text"])


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("Retrieval")
    strategy = st.segmented_control(
        "Strategy",
        ["Hybrid", "Dense"],
        default="Hybrid",
        help="Hybrid = dense + BM25 fused with RRF and an NER-entity boost (ADR-009).",
        label_visibility="collapsed",
    )
    use_hybrid = strategy != "Dense"

    has_groq = bool(
        os.environ.get("GROQ_API_KEY")
        or (
            hasattr(st, "secrets")
            and ("GROQ_API_KEY" in st.secrets or "groq_api_key" in st.secrets)
        )
    )
    if has_groq and not os.environ.get("GROQ_API_KEY") and hasattr(st, "secrets"):
        val = st.secrets.get("GROQ_API_KEY") or st.secrets.get("groq_api_key")
        if val and isinstance(val, str):
            os.environ["GROQ_API_KEY"] = val.strip()

    use_expansion = st.toggle(
        "Query expansion",
        value=has_groq,
        disabled=not has_groq,
        help="Rewrite the question into a keyword-rich search query with an LLM before "
        "retrieval. The generator still sees your original question."
        + ("" if has_groq else " Requires GROQ_API_KEY in Streamlit Secrets."),
    )
    use_verification = st.toggle(
        "Self-verification",
        value=True,
        help="A second model call checks whether the retrieved passages support the answer.",
    )

    st.divider()
    if st.button("New conversation", icon=":material/add_comment:", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pop("example", None)
        st.rerun()

    with st.expander("System", icon=":material/memory:"):
        st.markdown(
            f"""
| | |
|---|---|
| Generator | `{active_generator()}` |
| Local fallback | `{GEN_MODEL_NAME}` |
| Verifier | `{GROQ_VERIFY_MODEL}` |
| Embeddings | `bge-small-en-v1.5` |
| NER | `matscibert` |
| Vector store | ChromaDB |
"""
        )

# ── Main ─────────────────────────────────────────────────────────────────────
stats = corpus_stats()
hero(
    "SOFC literature assistant",
    "Ask the literature",
    "Answers are generated only from retrieved passages of the corpus, checked against "
    "those passages, and cited — or the assistant says it doesn't know.",
    chips=[
        ("papers", str(stats["papers"])),
        ("with full text", str(stats["full_text"])),
        ("retrieval", "Hybrid" if use_hybrid else "Dense"),
        ("generator", active_generator().split("/")[-1]),
    ],
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    avatar = USER_AVATAR if msg["role"] == "user" else ASSISTANT_AVATAR
    with st.chat_message(msg["role"], avatar=avatar):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            render_answer(msg)

picked = None
if not st.session_state.messages:
    empty_state(
        "Start with a question",
        "Ask about materials, degradation mechanisms, processing conditions or performance. "
        "Follow-up questions can refer back to earlier answers.",
    )
    picked = st.pills("Try one of these", EXAMPLES, selection_mode="single", key="example")

query = st.chat_input("Ask about solid oxide fuel cells…") or picked

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(query)

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        start = time.perf_counter()
        with st.status("Searching the literature…", expanded=False) as status:
            if use_expansion:
                st.write("Rewriting the question for search")
            st.write(f"Retrieving passages ({'hybrid' if use_hybrid else 'dense'})")
            answer, docs = answer_query(
                query,
                use_hybrid=use_hybrid,
                chat_history=st.session_state.messages[:-1],
                expand_q=use_expansion,
            )
            abstained = ABSTENTION_MARKER in answer
            verification = None
            if use_verification and not abstained:
                status.update(label="Verifying the answer…")
                verification = verify_answer(query, answer, docs)
            elapsed = time.perf_counter() - start
            status.update(label=f"Answered in {elapsed:.1f}s", state="complete")

        meta = [
            "Hybrid retrieval" if use_hybrid else "Dense retrieval",
            *(["query expansion"] if use_expansion else []),
            f"{len(docs)} passages",
            f"{elapsed:.1f}s",
        ]
        msg = {
            "role": "assistant",
            "content": answer,
            "abstained": abstained,
            "verification": verification,
            "meta": meta,
            "sources": [
                {
                    "rank": i,
                    "title": d.metadata.get("title", "Untitled"),
                    "paper_id": d.metadata.get("paper_id", ""),
                    "section": d.metadata.get("section"),
                    "text": d.page_content,
                }
                for i, d in enumerate(docs, 1)
            ],
        }
        render_answer(msg)

    st.session_state.messages.append(msg)
    st.rerun()
