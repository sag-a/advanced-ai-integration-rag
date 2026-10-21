from rag_core.registry import Architecture, register
from .retriever import check, make_retriever


def show_gate(details):
    import streamlit as st

    route = f"Jev router → {details['route']} · " if "route" in details else ""
    if "gate_probability" not in details:
        st.caption(f"{route}no collection covers this topic; abstained before retrieval")
        return
    decision = "passed" if details["gate_passed"] else "abstained before the LLM"
    st.caption(
        f"{route}Jev gate: P(answer in passages) = {details['gate_probability']:.2f}, "
        f"threshold {details['gate_threshold']:.2f} → {decision} · "
        f"reranker {details['reranker']} · Jev cost ${details['jev_cost_usd']:.5f}"
    )


ARCHITECTURE = Architecture(
    "jev_rag",
    "Jev RAG",
    make_retriever,
    check=check,
    render_details=show_gate,
    needs_llm_for_retrieval=True,  # Jev is a paid API call
)
register(ARCHITECTURE)
