# ruff: noqa: E501
"""Measures RAG retrieval quality against a hand-written set of test
questions. See Section 5.1 in docs/build-guide/06_evaluation.md for how to
write good test cases -- this is the part that can't be automated."""

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# pyrefly: ignore [missing-import]
from src.core.logging_config import get_logger

log = get_logger(__name__)

test_cases = [
    # Batch 1 (Seed 42)
    {
        "query": "What are the specific y values for manganese in the S1 and S2 samples of LiNi0.8Mn0.1Co0.1O2?",
        "expected_source_id": "http://arxiv.org/abs/2004.10297v2",
    },
    {
        "query": "Over what temperature range was the thermal conductivity of oxide, sulfide, and halide solid electrolytes measured?",
        "expected_source_id": "http://arxiv.org/abs/2103.08718v1",
    },
    {
        "query": "For the direct methanol fuel cell anode model, what expression allows expressing the anodic overpotential as a function of cell current?",
        "expected_source_id": "http://arxiv.org/abs/1001.1923v1",
    },
    {
        "query": "What control framework is used to formulate a predictive constraint-aware control scheme for localized hydration management in PEMFCs?",
        "expected_source_id": "http://arxiv.org/abs/2510.15250v1",
    },
    {
        "query": "What type of numerical model is established to explore the transport process in PEM fuel cells with ammonia impurities?",
        "expected_source_id": "http://arxiv.org/abs/2606.25992v1",
    },
    {
        "query": "What size of DMFC provided the data used to estimate kinetic and diffusional parameters in the semi-analytical model?",
        "expected_source_id": "http://arxiv.org/abs/2003.04083v1",
    },
    {
        "query": "What three specific factors are stated to significantly influence the 3D morphology and have a profound impact on the resulting performance of SOFC electrodes?",
        "expected_source_id": "http://arxiv.org/abs/2411.15259v2",
    },
    {
        "query": "In the SoLid experiment, what is the basic detection unit size for the highly segmented detector?",
        "expected_source_id": "http://arxiv.org/abs/2404.03580v5",
    },
    {
        "query": "In advanced PEMFCs, what two competing objectives does topology optimization aim to balance along the cathode gas channels?",
        "expected_source_id": "http://arxiv.org/abs/2509.16619v1",
    },
    {
        "query": "According to research on Anodic Aluminum Oxide templates, which two factors most influence the interpore distance?",
        "expected_source_id": "http://arxiv.org/abs/1301.2058v1",
    },
    # Batch 2 (Seed 43)
    {
        "query": "What is considered a major reason for anode degradation in Ni-yttria stabilized zirconia solid oxide fuel cell anodes?",
        "expected_source_id": "http://arxiv.org/abs/1201.1567v1",
    },
    {
        "query": "According to a recent prototypical model for actin waves, what type of feedback does F-actin exert onto a GTPase?",
        "expected_source_id": "http://arxiv.org/abs/2602.08779v1",
    },
    {
        "query": "Which two specific spinel-type cathode materials are stated as promising candidates for Li-ion battery technologies?",
        "expected_source_id": "http://arxiv.org/abs/2301.11143v2",
    },
    {
        "query": "What process is used to coat Nafion membranes with aqueous dispersions of electrochemically exfoliated graphene on the anode side?",
        "expected_source_id": "http://arxiv.org/abs/2204.04509v1",
    },
    {
        "query": "What does the sub-layer formed by in-phase oscillations of overpotential and oxygen concentration work as?",
        "expected_source_id": "http://arxiv.org/abs/2310.08945v1",
    },
    {
        "query": "How wide are the adhesive stripes used in the assay developed for studying nuclear migration dynamics?",
        "expected_source_id": "http://arxiv.org/abs/q-bio/0310014v3",
    },
    {
        "query": "What tool was used to unveil the stable nature of the Solid Electrolyte Interphase between Lithium Metal and LiPON?",
        "expected_source_id": "http://arxiv.org/abs/2006.12764v2",
    },
    {
        "query": "What two well-established methodologies for numerical studies of ionic motion involve static elastic bands or costly molecular dynamics?",
        "expected_source_id": "http://arxiv.org/abs/2504.06595v1",
    },
    {
        "query": "Why was graphene, once expected to be a promising LIB candidate, ultimately determined to be a poor choice for interacting with Li?",
        "expected_source_id": "http://arxiv.org/abs/1810.02498v1",
    },
    {
        "query": "What was the maximum room-temperature specific power exhibited by the prototype fuel cell using a carbon supported polyaniline anode catalyst?",
        "expected_source_id": "http://arxiv.org/abs/physics/0609025v1",
    },
]


def run_evaluation(test_cases: list[dict], k: int = 5) -> float:
    # Setup Chroma Retriever directly to bypass the slow LLM loading in rag_query.py
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = Chroma(
        collection_name="sofc_papers",
        embedding_function=embeddings,
        persist_directory="data/chroma_db",
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    hits = 0
    for case in test_cases:
        prefixed_query = (
            "Represent this question for searching relevant documents: " + case["query"]
        )
        retrieved_docs = retriever.invoke(prefixed_query)

        retrieved_ids = [doc.metadata.get("paper_id") for doc in retrieved_docs]

        hit = case["expected_source_id"] in retrieved_ids
        hits += hit
        log.info(f"{'HIT ' if hit else 'MISS'}  {case['query']}")
        if not hit:
            log.warning(f"  Expected: {case['expected_source_id']}")
            log.warning(f"  Got: {retrieved_ids}")

    hit_rate = hits / len(test_cases)
    log.info(f"Retrieval hit rate: {hits}/{len(test_cases)} = {hit_rate:.1%}")
    return hit_rate


if __name__ == "__main__":
    run_evaluation(test_cases)
