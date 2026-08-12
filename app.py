"""The user-facing chat interface. This is the diagram from Section 2 above,
made real: a question in, a grounded answer + verifiable sources out."""

import sys

import streamlit as st

sys.path.append("src")

from rag.rag_query import answer_query

# Visual Polish: Configure the page layout and title
st.set_page_config(page_title="MatSci-RAG Assistant", page_icon="🔬", layout="centered")

# Visual Polish: Sidebar with system info
with st.sidebar:
    st.title("🔬 MatSci-RAG")
    st.markdown("---")
    st.markdown("### System Architecture")
    st.markdown("- **Corpus**: 400 SOFC Papers")
    st.markdown("- **Embedding**: `BGE-small-en-v1.5`")
    st.markdown("- **Generator**: `google/flan-t5-base`")
    st.markdown("- **Database**: ChromaDB")
    st.markdown("---")
    st.caption("Built for Materials Science Literature Retrieval.")

st.title("Materials Science Literature Assistant")
st.write(
    "Ask a question about Solid Oxide Fuel Cells (SOFC) or related materials science "
    "literature, and the AI will generate an answer strictly based on retrieved scientific papers."
)

# Initialize chat history in session state for a better UI experience
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history (we only really need to show the last one, but this is a nice chat layout)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query = st.chat_input("Ask a question about SOFCs...")

if query:
    # Display user message
    with st.chat_message("user"):
        st.markdown(query)

    st.session_state.messages.append({"role": "user", "content": query})

    # Display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching literature and generating answer..."):
            answer, retrieved = answer_query(query)

        st.markdown(f"**{answer}**")

        st.markdown("---")
        st.markdown("### 📚 Sources used")
        for i, p in enumerate(retrieved):
            title = p.metadata.get("title", "Unknown Title")
            source_id = p.metadata.get("paper_id", "Unknown Source")
            with st.expander(f"[{i+1}] {title}"):
                st.write(p.page_content)
                st.caption(f"Source: {source_id}")

    st.session_state.messages.append({"role": "assistant", "content": answer})
