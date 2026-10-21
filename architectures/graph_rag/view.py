"""An inspectable graph plus original evidence, including restored turns."""

import json
import streamlit as st
from rich.console import Group
from rich.text import Text
from rag_core.lesson_log import plain, passage_table as shared_passage_table


def dot_graph(details):
    lines = ["digraph RAG { rankdir=LR; node [shape=box, fontsize=12]; edge [fontsize=10];"]
    anchors = set(details["anchors"])
    for entity in details["entities"]:
        label = entity["label"]
        color = "#fff2cc" if entity["id"] in anchors else "#edf2f7"
        lines.append(
            f'{json.dumps(entity["id"])} [label={json.dumps(label[:40], ensure_ascii=False)}, tooltip={json.dumps(label, ensure_ascii=False)}, style=filled, fillcolor="{color}"];'
        )
    colors = {"positive": "#4b5563", "negative": "#b91c1c", "conditional": "#92400e"}
    for edge in details["relationships"]:
        label = edge["id"] + ": " + edge["polarity"]
        lines.append(
            f'{json.dumps(edge["source"])} -> {json.dumps(edge["target"])} [label={json.dumps(label)}, color="{colors[edge["polarity"]]}", tooltip={json.dumps(edge["statement"], ensure_ascii=False)}];'
        )
    return "\n".join(lines + ["}"])


def show_graph(details):
    with st.expander("Graph retrieval"):
        st.write(details["explanation"])
        st.caption(
            f"{len(details['seed_ids'])} vector seeds · {details.get('selected_expansion_count', 0)} graph-added passages in final context · index {details['index_fingerprint'][:16]}"
        )
        if details.get("ambiguous_query_aliases"):
            st.warning(
                "Ambiguous aliases were excluded from question matching: "
                + ", ".join(details["ambiguous_query_aliases"])
            )
        edges = details.get("relationships", [])
        if edges:
            st.graphviz_chart(dot_graph(details))
            st.caption(
                "Yellow nodes are anchors. Arrows show fact direction; traversal may follow either direction. Match edge IDs to the table. Negative facts are red, conditions brown. Extraction can be wrong; inspect its quotation."
            )
            st.dataframe(
                [
                    {
                        k: edge.get(k, "")
                        for k in (
                            "id",
                            "source",
                            "target",
                            "statement",
                            "polarity",
                            "hop",
                            "page_start",
                            "clause",
                            "quote",
                            "chunk_id",
                        )
                    }
                    for edge in edges
                ],
                hide_index=True,
            )
            with st.expander("Read supporting quotations"):
                for edge in edges:
                    st.caption(
                        f"{edge['id']} · {edge['polarity']} · PDF p. {edge.get('page_start', '?')} · §{edge.get('clause', '')} · hop {edge['hop']}"
                    )
                    st.text(edge["statement"])
                    st.text(edge["quote"])
        with st.expander("Vector seeds and selected PDF passages"):
            seeds = set(details["seed_ids"])
            for passage in details.get("selected_passages", []):
                tags = passage["metadata"]
                label = (
                    "Vector seed"
                    if passage["id"] in seeds
                    else "Graph-added"
                    if details["graph_used"]
                    else "Vector result"
                )
                st.caption(
                    f"{label} · PDF p. {tags['page_start']}–{tags['page_end']} · §{tags.get('clause', '')}"
                )
                st.text(passage["page_content"])
            omitted = [
                p
                for p in details.get("seed_passages", [])
                if p["id"] not in details["selected_ids"]
            ]
            for passage in omitted:
                st.caption("Vector seed omitted by final ranking")
                st.text(passage["page_content"])


def passage_table(documents, seed_ids=None, graph_used=False):
    origins = {
        d.id: "Vector seed"
        if seed_ids is None or d.id in seed_ids
        else "Graph-added"
        if graph_used
        else "Vector result"
        for d in documents
    }
    return shared_passage_table(documents, origins)


def render_panels(self, event, data):
    request = data["request_id"]
    if event == "retrieval_started":
        self.panel(
            "Graph RAG · retrieval walkthrough",
            Text(
                f"Vector seeds → entity anchors → graph hops → original PDF passages\nFinal passage budget: {data['top_k']} · traversal makes no Anthropic call"
            ),
            request_id=request,
        )
    elif event == "seeds_selected":
        self.panel(
            "1 · Vector seeds",
            Group(
                Text(
                    f"{len(data['documents'])} passages found by local embeddings · {data['elapsed_seconds']:.2f}s"
                ),
                passage_table(data["documents"]),
                Text(
                    "Lower cosine distance means closer vectors; it is not proof of relevance.",
                    style="dim",
                ),
            ),
            request_id=request,
        )
    elif event == "anchors_selected":
        anchors = Text(" · ").join((plain(a, 80) for a in data["anchors"]))
        self.panel(
            "2 · Entity anchors",
            Group(
                anchors if data["anchors"] else Text("No seed-linked entities."),
                Text(
                    "Question-alias matches are prioritized; ambiguous aliases are excluded.",
                    style="dim",
                ),
            ),
            request_id=request,
        )
        if data["ambiguous_aliases"]:
            self.panel(
                "Ambiguous aliases",
                Text(", ").join((plain(a) for a in data["ambiguous_aliases"])),
                style="yellow",
                request_id=request,
            )
    elif event == "hop_completed":
        edges = data["relationships"]
        lines = [
            Text(
                f"{len(edges)} relationships · {data['new_passages']} additional candidate passages"
            )
        ]
        first_by_polarity = {}
        for edge in edges:
            first_by_polarity.setdefault(edge["polarity"], edge)
        samples = list(first_by_polarity.values())
        samples += [edge for edge in edges if edge not in samples]
        samples = sorted(samples[:3], key=edges.index)
        for edge in samples:
            line = Text(
                f"{edge['polarity']} · ",
                style={"negative": "red", "conditional": "yellow"}.get(
                    edge["polarity"], "green"
                ),
            )
            line.append_text(plain(edge["source"], 50)).append(" → ").append_text(
                plain(edge["target"], 50)
            )
            line.append(f" · PDF p. {edge['page_start']} · §").append_text(
                plain(edge.get("clause", ""), 30)
            )
            lines.append(line)
            lines.append(plain(edge["statement"], 160))
        if data["hop"] == 1:
            lines.append(
                Text(
                    "Arrows show fact direction; traversal can follow a connection either way.",
                    style="dim",
                )
            )
        if len(edges) > 3:
            lines.append(
                Text(
                    f"Showing 3 of {len(edges)} traversed facts; full paths and quotations remain in the UI/JSON.",
                    style="dim",
                )
            )
        self.panel(f"3 · Hop {data['hop']}", Group(*lines), request_id=request)
    elif event == "passages_selected":
        expanded = data["selected_expansion_count"]
        summary = Text(
            f"{len(data['documents'])} passages selected · {expanded} beyond the vector seeds · {data['elapsed_seconds']:.2f}s total"
        )
        rule = (
            Text(
                "Equal-weight reciprocal-rank fusion (constant 60); highest vector seed retained.",
                style="dim",
            )
            if data["graph_used"]
            else Text(
                "No graph expansion contributed passages. Using vector results.", style="yellow"
            )
        )
        limits = data.get("limits", {})
        budget = Text(
            f"Budget used: {data.get('hops_used', 0)}/{limits.get('hops', 2)} hops · {data['traversed_count']}/{limits.get('relationships', 30)} facts · {data.get('candidate_count', len(data['documents']))}/{limits.get('candidates', 12)} candidate passages",
            style="dim",
        )
        self.panel(
            "4 · Final PDF passages",
            Group(
                summary,
                budget,
                passage_table(data["documents"], set(data["seed_ids"]), data["graph_used"]),
                rule,
                Text(
                    f"{data['traversed_count']} relationships traversed · index {data['index_fingerprint'][:16]}\nAnswers cite these original passages; extracted facts only guide retrieval.",
                    style="dim",
                ),
            ),
            style="green",
            request_id=request,
        )
    elif event == "retrieval_failed":
        self.panel(
            "Retrieval failed",
            Text(
                f"{data['error_type']}: stopped before evidence selection completed. No retrieval success was reported."
            ),
            style="red",
            request_id=request,
        )
