"""Grade both parts of starter.py: chunk size (Part 1) and Markdown headers (Part 2)."""

SECTIONS = ["Refunds", "Cancellations", "Payments"]

# ---------------------------------------------------------------------------
# Part 1: size-based chunking of the refund policy
# ---------------------------------------------------------------------------


def chunk_with(chunks, phrase):
    return next((chunk for chunk in chunks if phrase in " ".join(chunk.split())), None)


def test_settings_are_valid(lab):
    assert lab.CHUNK_SIZE > 0
    assert 0 <= lab.CHUNK_OVERLAP < lab.CHUNK_SIZE


def test_the_refund_rule_keeps_its_condition(lab):
    chunks = lab.make_chunks()
    rule = chunk_with(chunks, "full refund if they cancel within 14 days")
    assert rule is not None, "the refund rule was cut in the middle: chunk_size is too small"
    assert "provided they have not used the service" in " ".join(rule.split()), (
        "the refund rule lost its condition: a reader of this chunk alone would think "
        "every customer gets a full refund"
    )


def test_the_late_cancellation_rule_is_complete(lab):
    chunks = lab.make_chunks()
    rule = chunk_with(chunks, "After 14 days")
    assert rule is not None
    assert "No refund is issued for the remaining days" in rule, (
        "the late-cancellation rule was split from its 'no refund' consequence"
    )


def test_each_chunk_holds_one_rule(lab):
    chunks = lab.make_chunks()
    mixed = [c for c in chunks if "within 14 days" in " ".join(c.split()) and "After 14 days" in c]
    assert not mixed, "both rules are in one chunk: chunk_size is too big to keep them apart"


def test_no_sentence_is_cut(lab):
    for chunk in lab.make_chunks():
        assert chunk.rstrip().endswith("."), f"this chunk ends mid-sentence: {chunk[-40:]!r}"


# ---------------------------------------------------------------------------
# Part 2: header-based chunking of support_manual.md
# ---------------------------------------------------------------------------


def test_the_manual_is_unchanged(lab):
    # The task is to change the splitter settings, not the document.
    for heading in ("# Customer Support Manual", "## Refunds", "### Eligibility", "### Exceptions"):
        assert heading in lab.MANUAL, f"{heading!r} is missing: leave support_manual.md unchanged"


def test_one_chunk_per_main_section(lab):
    sections = lab.make_sections()
    found = [chunk.metadata.get("Section") for chunk in sections]
    assert found == SECTIONS, (
        f"expected one chunk each for {SECTIONS}, got {len(sections)} chunks with Section={found}: "
        "split at the ## headings"
    )


def test_every_chunk_names_its_document_and_section(lab):
    for chunk in lab.make_sections():
        assert chunk.metadata.get("Document") == "Customer Support Manual", chunk.metadata
        assert chunk.metadata.get("Section") in SECTIONS, chunk.metadata


def test_the_refund_rule_and_its_exception_share_one_chunk(lab):
    refunds = [c.page_content for c in lab.make_sections() if "### Eligibility" in c.page_content]
    assert refunds, "no chunk holds the Eligibility rule"
    assert "### Exceptions" in refunds[0], (
        "Eligibility and Exceptions are in different chunks: the chatbot would promise a full "
        "refund without the exception for customers who used the service"
    )


def test_no_chunk_mixes_main_sections(lab):
    for chunk in lab.make_sections():
        headings = [line.strip() for line in chunk.page_content.splitlines() if line.startswith("#")]
        main = [h for h in headings if h.startswith("## ")]
        assert len(main) <= 1, f"one chunk holds several sections: {main}"
        # A ## heading after a ### subsection means the chunk ends inside the next section.
        if main:
            assert not any(h.startswith("### ") for h in headings[: headings.index(main[0])]), (
                f"{main[0]!r} sits at the end of a chunk that belongs to another section"
            )
