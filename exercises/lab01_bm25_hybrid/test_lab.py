import pytest

DOCS = [
    "Double Play 24 XL includes 420 free minutes to mobiles.",
    "Double Play 24 XL includes 500 free minutes to mobiles.",
    "Home Telephony for pensioners with a discount on local calls.",
    "minutes minutes minutes minutes minutes minutes minutes minutes minutes minutes",
]


def test_tokenize_lowercases_folds_accents_and_keeps_prices(lab):
    assert lab.tokenize("Οι «Καρδιές» λήγουν.") == ["οι", "καρδιες", "ληγουν"]
    assert lab.tokenize("0,3072 €/min · 24 S · XL") == ["0,3072", "min", "24", "s", "xl"]


def test_bm25_ranks_the_exact_number_first(lab):
    assert lab.BM25(DOCS).top("Double Play 24 XL 500 minutes", 1) == [1]


def test_documents_without_shared_terms_score_zero_and_are_not_returned(lab):
    bm25 = lab.BM25(DOCS)
    assert bm25.score("Fiberspeed broadband")[0] == 0
    assert bm25.top("Fiberspeed broadband", 3) == []


def test_rare_terms_weigh_more_than_common_ones(lab):
    bm25 = lab.BM25(DOCS)
    # "pensioners" appears in one document, "minutes" in three.
    assert bm25.score("pensioners")[2] > bm25.score("minutes")[0]


def test_term_frequency_saturates(lab):
    # Ten repetitions must not be worth anywhere near ten times one occurrence.
    bm25 = lab.BM25(["minutes"] + ["minutes " * 10] + ["other words here"] * 3, b=0)
    one, ten = bm25.score("minutes")[:2]
    assert one < ten < 2.5 * one


def test_length_normalization_favours_the_shorter_document(lab):
    bm25 = lab.BM25(["hearts expire", "hearts expire " + "padding " * 30, "unrelated text"])
    short, long, _ = bm25.score("hearts")
    assert short > long
    no_norm = lab.BM25(["hearts expire", "hearts expire " + "padding " * 30, "unrelated text"], b=0)
    assert no_norm.score("hearts")[0] == pytest.approx(no_norm.score("hearts")[1])


def test_rrf_rewards_items_that_both_rankings_agree_on(lab):
    # c (ranks 1 and 3): 1/61 + 1/63 = 0.032267 just beats b (ranks 2 and 2): 2/62 = 0.032258.
    # Both beat a and x, which only one ranking contains.
    assert lab.rrf([["a", "b", "c"], ["c", "b", "x"]], top_k=4) == ["c", "b", "a", "x"]


def test_rrf_uses_rank_not_raw_scores_and_respects_top_k(lab):
    assert lab.rrf([["a", "b"], ["b", "a"]], top_k=1) == ["a"]  # tie: first ranking wins
    assert lab.rrf([["a"], []], top_k=3) == ["a"]
    assert lab.rrf([["a", "b"], ["b"]], k=0, top_k=2) == ["b", "a"]
