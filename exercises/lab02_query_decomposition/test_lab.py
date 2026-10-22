def test_a_single_question_is_returned_unchanged(lab):
    q = "Πότε λήγουν οι Καρδιές;"
    assert lab.split_question(q) == [q]


def test_english_multi_part_question_splits_on_commas_and_and(lab):
    q = "How do I register for Magenta Moments, and how long will the Hearts I earn remain valid?"
    assert lab.split_question(q) == [
        "How do I register for Magenta Moments",
        "how long will the Hearts I earn remain valid",
    ]


def test_greek_multi_part_question_splits_too(lab):
    q = "Θέλω να φύγω από το πρόγραμμα: πώς αποχωρώ, πότε μπορώ να ξαναμπώ;"
    # "πώς αποχωρώ" ("how do I leave") has two words, so the three-word rule drops it:
    # a real sub-question lost. Rules are brittle; the README asks you to discuss it.
    assert lab.split_question(q) == ["Θέλω να φύγω από το πρόγραμμα", "πότε μπορώ να ξαναμπώ"]


def test_fragments_shorter_than_three_words_are_dropped(lab):
    assert lab.split_question("Prices: monthly fee and free minutes for Double Play 24 L?") == [
        "Prices: monthly fee and free minutes for Double Play 24 L?"
    ]


def test_at_most_four_parts(lab):
    q = "what is one thing, what is two thing, what is three thing, what is four thing, what is five thing"
    assert len(lab.split_question(q)) == 4


def test_merge_takes_turns_and_keeps_the_full_question_first(lab):
    full, part_a, part_b = ["a", "b", "c"], ["d", "a", "e"], ["f", "g"]
    assert lab.merge(full, [part_a, part_b], k=5) == ["a", "d", "f", "b", "g"]


def test_merge_skips_duplicates_and_handles_empty_rankings(lab):
    assert lab.merge(["a", "b"], [["a", "b"], []], k=3) == ["a", "b"]
    assert lab.merge([], [], k=3) == []


class FakeChat:
    def __init__(self, questions):
        self.questions, self.calls = questions, []

    def call(self, schema, system, payload):
        self.calls.append((system, payload))
        return schema(questions=self.questions), {"input_tokens": 1}


def test_llm_decompose_asks_for_standalone_questions_without_pronouns(lab):
    chat = FakeChat(["How do I join Magenta Moments?", "How long are Hearts valid?"])
    parts = lab.llm_decompose("How do I join, and how long are they valid?", chat)
    assert parts == ["How do I join Magenta Moments?", "How long are Hearts valid?"]
    system, payload = chat.calls[0]
    assert "pronoun" in system.lower() and payload == {"question": "How do I join, and how long are they valid?"}


def test_llm_decompose_falls_back_and_enforces_limits(lab):
    q = "Πότε λήγουν οι Καρδιές;"
    assert lab.llm_decompose(q, FakeChat([])) == [q]
    assert lab.llm_decompose(q, FakeChat(["Πότε λήγουν οι Καρδιές;"])) == [q]
    six = [f"Question {i} about plans?" for i in range(6)]
    assert lab.llm_decompose("long question", FakeChat(six)) == six[:4]
    assert lab.llm_decompose("q", FakeChat(["A b c?", " A b c? ", "D e f?"])) == ["A b c?", "D e f?"]


def test_the_schema_is_a_list_of_questions(lab):
    assert lab.SubQuestions(questions=["x"]).questions == ["x"]
