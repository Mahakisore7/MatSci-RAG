"""The user-facing chat interface. This is the diagram from Section 2 above,
made real: a question in, a grounded answer + verifiable sources out."""

import streamlit as st

from src.rag.rag_query import answer_query, verify_answer

# Visual Polish: Configure the page layout and title
st.set_page_config(page_title="MatSci-RAG Assistant", page_icon="🔬", layout="centered")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔬 MatSci-RAG")
    st.markdown("---")
    st.markdown("**Retrieval Strategy**")
    retrieval_choice = st.radio(
        "Choose strategy",
        ["Hybrid (Dense + BM25 + NER)", "Dense Only"],
        index=0,
        label_visibility="collapsed",
    )
    use_hybrid = retrieval_choice == "Hybrid (Dense + BM25 + NER)"

    st.markdown("---")
    st.markdown("### System Architecture")
    st.markdown("- **Corpus:** 400 SOFC papers")
    st.markdown("- **Embedding:** `BAAI/bge-small-en-v1.5`")
    st.markdown("- **Generator:** `google/flan-t5-base`")
    st.markdown("- **Database:** ChromaDB")

# ── Main Chat UI ─────────────────────────────────────────────────────────────
st.title("Materials Science Literature Assistant")
st.markdown(
    "Ask questions about Solid Oxide Fuel Cells (SOFCs). "
    "The AI will answer based *only* on the provided literature corpus."
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if query := st.chat_input("Ask a question about SOFCs..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(query)

    # Display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching literature and generating answer..."):
            answer, retrieved = answer_query(query, use_hybrid=use_hybrid)

        st.markdown(f"**{answer}**")

        # Self-Verification Step (skip if we abstained due to ambiguity)
        if "Multiple candidate materials found" not in answer:
            with st.spinner("Verifying answer against context..."):
                verification_result = verify_answer(query, answer, retrieved)

                if verification_result.lower().startswith("yes"):
                    st.success("✅ **Verified:** " + verification_result)
                elif verification_result.lower().startswith("no"):
                    st.warning("⚠️ **Unverified:** " + verification_result)
                else:
                    st.info("ℹ️ **Verification Note:** " + verification_result)

        st.markdown("---")
        st.markdown("### 📚 Sources used")
        for i, p in enumerate(retrieved):
            title = p.metadata.get("title", "Unknown Title")
            source_id = p.metadata.get("paper_id", "Unknown Source")
            with st.expander(f"[{i + 1}] {title}"):
                st.write(p.page_content)
                st.caption(f"Source: {source_id}")

    st.session_state.messages.append({"role": "assistant", "content": answer})
