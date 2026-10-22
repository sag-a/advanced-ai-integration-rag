"""Inspect target-word vectors before/after context, not pooled RAG embeddings."""

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re

import numpy as np

from modules.c01_embeddings.local_embeddings import (
    EMBEDDING_MODEL as MODEL,
    REVISION,
    CODE_REVISION,
    MAX_TOKENS,
    load_model,
)

PREFIX = "search_document: "


@dataclass(frozen=True)
class Example:
    id: str
    text: str
    target: str
    sense: str
    occurrence: int = 0


EXAMPLES = [
    Example("O1", "I drink an orange juice.", "orange", "fruit"),
    Example("O2", "The sunset is orange.", "orange", "colour"),
    Example("O3", "She peeled an orange for breakfast.", "orange", "fruit"),
    Example("O4", "She painted the wall orange.", "orange", "colour"),
    Example("O5", "The orange juice is sweet.", "orange", "fruit"),
    Example("O6", "The orange paint is bright.", "orange", "colour"),
    Example("R1", "I ate a red apple.", "red", "colour"),
    Example("R2", "Red is my favourite colour.", "red", "colour"),
    Example("R3", "I ate a red grape.", "red", "colour"),
    Example("R4", "The business is in the red.", "red", "financial idiom"),
    Example("R5", "The company is in the red.", "red", "financial idiom"),
    Example("D1", "I drink an orange juice.", "orange", "repeat of O1"),
]


def target_span(text: str, target: str, occurrence: int = 0) -> tuple[int, int]:
    """Find the target's character positions, with an exclusive end position.

    For example, target_span("I drink an orange juice.", "orange") returns
    (11, 17): text[11:17] is "orange". Matching ignores case; occurrence=0
    selects the first match, occurrence=1 selects the second, and so on.
    """
    if not target.strip() or occurrence < 0:
        raise ValueError("Target must be nonempty and occurrence nonnegative.")
    whole_word_pattern = r"(?<!\w)" + re.escape(target) + r"(?!\w)"
    matches = list(re.finditer(whole_word_pattern, text, re.IGNORECASE))
    if occurrence >= len(matches):
        raise ValueError(f"Target occurrence {occurrence} not found: {target!r}.")
    match = matches[occurrence]
    return match.start(), match.end()


def token_indices(offsets, span: tuple[int, int]) -> list[int]:
    """Map a character span to token positions, including all subword pieces.

    Each tokenizer offset is a (start, end) character pair. Special tokens
    have empty offsets, such as (0, 0), and must not be selected.
    """
    word_start, word_end = span
    selected = []
    covered_characters = set()

    for token_index, (token_start, token_end) in enumerate(offsets):
        if token_start >= token_end:
            continue
        overlaps_word = token_start < word_end and token_end > word_start
        if overlaps_word:
            selected.append(token_index)
            overlap_start = max(word_start, token_start)
            overlap_end = min(word_end, token_end)
            covered_characters.update(range(overlap_start, overlap_end))

    # Reject partial matches, for example if tokenization cut off the word.
    word_characters = set(range(word_start, word_end))
    if not word_characters or covered_characters != word_characters:
        raise ValueError(
            "Target is not fully covered by tokenizer offsets; check truncation."
        )
    return selected


def mean_target_vector(hidden: np.ndarray, indices: list[int]) -> np.ndarray:
    """Average the target's token vectors into one vector if split into pieces."""
    if not indices:
        raise ValueError("At least one target token is required.")
    target_vectors = np.asarray(hidden)[indices]
    return target_vectors.mean(axis=0)


def cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Expected a finite matrix of vectors.")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0) or not np.isfinite(norms).all():
        raise ValueError("Cannot compare zero or invalid vectors.")
    unit = values / norms
    return np.clip(unit @ unit.T, -1, 1)


def extract_vectors(examples=EXAMPLES, *, allow_download=False, layer=-1):
    import torch

    torch.manual_seed(0)
    tokenizer, model = load_model(allow_download=allow_download)
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer is required for exact character offsets.")
    layers = len(model.encoder.layers)
    layer = layers if layer == -1 else layer
    if not 1 <= layer <= layers:
        raise ValueError(f"Choose a transformer layer between 1 and {layers}.")
    texts = [PREFIX + example.text for example in examples]
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=False,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets = encoded.pop("offset_mapping").tolist()
    if encoded["input_ids"].shape[1] > MAX_TOKENS:
        raise ValueError(f"Input exceeds the model limit of {MAX_TOKENS} tokens.")


    captured = []

    def capture_layer(module, inputs, output):
        captured.append(output[0].detach().cpu().numpy())

    handle = model.encoder.layers[layer - 1].register_forward_hook(capture_layer)
    try:
        with torch.inference_mode():
            lookup = (
                model.embeddings.word_embeddings(encoded["input_ids"]).cpu().numpy()
            )
            model(**encoded)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise ValueError("Expected exactly one captured transformer-layer output.")
    contextual = captured[0]
    records, before, after = [], [], []
    for i, example in enumerate(examples):
        # 1. Find the word's character positions in the original sentence.
        start, end = target_span(example.text, example.target, example.occurrence)

        # 2. Account for the prefix added before tokenizing the sentence.
        prefixed_span = (start + len(PREFIX), end + len(PREFIX))

        # 3. Select the tokens that cover those characters.
        target_token_positions = token_indices(offsets[i], prefixed_span)
        target_token_ids = encoded["input_ids"][i, target_token_positions].tolist()
        records.append(
            {
                **asdict(example),
                "token_positions": target_token_positions,
                "token_ids": target_token_ids,
                "token_pieces": tokenizer.convert_ids_to_tokens(target_token_ids),
            }
        )

        # 4. Extract the same tokens before and after the transformer layers.
        before.append(mean_target_vector(lookup[i], target_token_positions))
        after.append(mean_target_vector(contextual[i], target_token_positions))
    return records, np.stack(before), np.stack(after), layer


def build_report(records, before, after, layer):
    from importlib.metadata import version

    static_cos, context_cos = cosine_matrix(before), cosine_matrix(after)
    positions = {r["id"]: i for i, r in enumerate(records)}
    pairs = [
        ("O1", "O2", "Original orange examples: fruit versus colour"),
        ("O1", "O3", "Orange in two fruit contexts"),
        ("O2", "O4", "Orange in two colour contexts"),
        ("O5", "O6", "Matched target position: juice versus paint"),
        ("R1", "R2", "Original red examples: both still mean colour"),
        ("R1", "R3", "Matched target position: red apple versus red grape"),
        ("R4", "R5", "Red in two financial-idiom contexts"),
        ("O1", "D1", "Identical input repeat control"),
    ]
    measurements = []
    for first, second, description in pairs:
        i, j = positions[first], positions[second]
        measurements.append(
            {
                "first": first,
                "second": second,
                "comparison": description,
                "lookup_cosine": float(static_cos[i, j]),
                "contextual_cosine": float(context_cos[i, j]),
                "same_token_ids": records[i]["token_ids"] == records[j]["token_ids"],
                "same_token_positions": records[i]["token_positions"]
                == records[j]["token_positions"],
            }
        )
    return {
        "model": MODEL,
        "revision": REVISION,
        "code_revision": CODE_REVISION,
        "prefix": PREFIX,
        "layer": layer,
        "dimensions": int(after.shape[1]),
        "device": "cpu",
        "model_mode": "eval",
        "subword_aggregation": "Mean of all token pieces overlapping the target character span",
        "versions": {
            p: version(p) for p in ["torch", "transformers", "numpy", "scikit-learn"]
        },
        "examples": records,
        "comparisons": measurements,
        "lookup_cosine": static_cos.tolist(),
        "contextual_cosine": context_cos.tolist(),
        "interpretation": "Context changes token representations in the same layer/space. A changed vector alone does not prove a changed word sense. These are token hidden states, not pooled OpenAI or Nomic retrieval embeddings.",
    }


def save_plots(records, before, after, report, directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    colours = {"orange": "#B76700", "red": "#B42D49"}
    markers = {"fruit": "o", "colour": "D", "financial idiom": "s", "repeat of O1": "o"}
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    fig.subplots_adjust(left=0.06, right=0.97, top=0.83, bottom=0.37, wspace=0.27)
    fig.suptitle(
        "Same word, different context: the vector changes",
        fontsize=22,
        fontweight="bold",
        y=0.96,
    )
    fig.text(
        0.5,
        0.91,
        f"Nomic v2 MoE · target-token vectors · {after.shape[1]} dimensions · transformer layer {report['layer']}",
        ha="center",
        fontsize=12,
    )
    projections = []
    for ax, vectors, title in zip(
        axes,
        [before, after],
        ["Before context: vocabulary lookup", "After context: token hidden state"],
        strict=True,
    ):
        unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        pca = PCA(n_components=2, svd_solver="full")
        xy = pca.fit_transform(unit)
        projections.append(
            {
                "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                "coordinates": xy.tolist(),
            }
        )
        # Repeated lookup vectors and the identical-input control share positions.
        groups = []
        for i in range(len(vectors)):
            match = next(
                (
                    g
                    for g in groups
                    if np.allclose(vectors[i], vectors[g[0]], rtol=1e-5, atol=1e-7)
                ),
                None,
            )
            if match is None:
                groups.append([i])
            else:
                match.append(i)
        for group in groups:
            i = group[0]
            ex = records[i]
            ax.scatter(
                *xy[i],
                c=colours[ex["target"]],
                marker=markers[ex["sense"]] if title.startswith("After") else "o",
                s=100,
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )
            label = "/".join(records[j]["id"] for j in group)
            alignment = "left"
            offset = (7, 8)
            if title.startswith("Before"):
                start, end = target_span(ex["text"], ex["target"], ex["occurrence"])
                label = ex["text"][start:end] + "\n" + label
                if ex["target"] == "orange":
                    offset, alignment = (-7, 8), "right"
            else:
                offset = {
                    "R1": (-25, 10),
                    "R3": (7, -15),
                    "R4": (-25, 12),
                    "R5": (-25, -18),
                }.get(ex["id"], offset)
            ax.annotate(
                label,
                xy[i],
                xytext=offset,
                textcoords="offset points",
                fontsize=10,
                ha=alignment,
            )
        ax.set_title(title, fontsize=14, pad=14)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
        ax.grid(alpha=0.15)
        ax.margins(x=0.22, y=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.06,
        0.30,
        "Orange markers = “orange” · red markers = “red”   |   circle = fruit · diamond = colour · square = financial idiom",
        fontsize=11,
    )
    for i, ex in enumerate(records):
        column, row = divmod(i, 6)
        fig.text(
            0.06 + column * 0.47,
            0.26 - row * 0.031,
            f"{ex['id']:>2}  {ex['text']}",
            fontsize=11,
        )
    fig.text(
        0.06,
        0.035,
        "Independent PCA fits: axes cannot be compared across panels. 2D projections distort distances.\nCheck the full-dimensional cosine matrix. Different vectors alone do not establish different word senses.",
        fontsize=11,
        color="#444444",
    )
    fig.savefig(directory / "contextual_word_positions.png", dpi=140)
    fig.savefig(directory / "contextual_word_positions.pdf")
    plt.close(fig)
    report["pca"] = {
        "input": "L2-normalized target vectors; independent before/after fits",
        "lookup": projections[0],
        "contextual": projections[1],
    }

    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    labels = [ex["id"] for ex in records]
    for ax, key, title in zip(
        axes,
        ["lookup_cosine", "contextual_cosine"],
        [
            "Vocabulary lookup: same token IDs = same vectors",
            f"Contextual token vectors: layer {report['layer']}",
        ],
        strict=True,
    ):
        matrix = np.array(report[key])
        picture = ax.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu", aspect="equal")
        ax.set_xticks(range(len(records)), labels, rotation=45)
        ax.set_yticks(range(len(records)), labels)
        ax.set_title(title, fontsize=13, pad=15)
        for (row, col), value in np.ndenumerate(matrix):
            ax.text(
                col,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if abs(value) > 0.55 else "#222222",
            )
    fig.subplots_adjust(left=0.05, right=0.90, top=0.85, bottom=0.14, wspace=0.15)
    fig.colorbar(
        picture,
        cax=fig.add_axes([0.92, 0.22, 0.015, 0.55]),
        label="Cosine similarity (not probability)",
    )
    fig.suptitle(
        f"Measure similarity in the original {after.shape[1]} dimensions",
        fontsize=20,
        fontweight="bold",
        y=0.96,
    )
    fig.text(
        0.05,
        0.04,
        "O1–O6: orange contexts · R1–R3: red colour contexts · R4–R5: financial idiom · D1: exact repeat of O1\nThese selected examples test context dependence, not word-sense classification accuracy.",
        fontsize=11,
    )
    fig.savefig(directory / "contextual_word_similarity.png", dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/contextual_words_v2")
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=-1,
        help="Transformer layer 1–12; default final layer",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Permit downloading the pinned model if absent from the local cache",
    )
    args = parser.parse_args()
    try:
        records, before, after, layer = extract_vectors(
            allow_download=args.allow_download, layer=args.layer
        )
        report = build_report(records, before, after, layer)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        save_plots(records, before, after, report, args.output_dir)
        np.savez_compressed(
            args.output_dir / "target_vectors.npz", lookup=before, contextual=after
        )
        (args.output_dir / "measurements.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Model: {MODEL} @ {REVISION}; layer={layer}; vectors={after.shape}")
        for row in report["comparisons"]:
            print(
                f"{row['first']}/{row['second']}: lookup={row['lookup_cosine']:.4f}, contextual={row['contextual_cosine']:.4f} — {row['comparison']}"
            )
        print(
            "Saved PNG plots, a PDF, exact vectors, and measurements to",
            args.output_dir,
        )
        print(report["interpretation"])
    except OSError:
        parser.exit(
            1,
            "Model/files unavailable. Prepare the model with --allow-download while online; the default uses the local cache only. Check output-directory access.\n",
        )
    except ValueError as error:
        parser.exit(1, f"Context experiment error: {error}\n")


if __name__ == "__main__":
    main()
