import pytest

from conftest import make_chunk
from hybrid_rag.generation import NO_ANSWER, check_citations, make_answer, split_sentences
from hybrid_rag.schema import ScoredChunk

SOURCES = [
    ScoredChunk(chunk=make_chunk("rrf.md::c0", "The fusion formula uses k = 60 to weight ranks."), score=1),
    ScoredChunk(chunk=make_chunk("intro.md::c0", "A local system that fuses dense and sparse search."), score=1),
    ScoredChunk(chunk=make_chunk("setup.md::c0", "Install Python 3.11 and run Ollama."), score=1),
]


def answer(text: str):
    return make_answer("q", text, SOURCES, "fake-model")


@pytest.mark.parametrize(
    "text, expected",
    [
        ("One. Two [1]. Three!", ["One.", "Two [1].", "Three!"]),
        ("It is red. [2] It is big [1].", ["It is red. [2]", "It is big [1]."]),
        ("Single sentence without stop", ["Single sentence without stop"]),
    ],
)
def test_split_sentences_keeps_citations_attached(text, expected):
    assert split_sentences(text) == expected


def test_correct_citation_is_supported():
    [check] = check_citations(answer("The fusion formula uses k = 60 [1]."))
    assert check.status == "supported" and check.best_source == 1


def test_wrong_citation_is_flagged_with_best_match():
    # Mirrors a real qwen2.5:1.5b mistake: right fact, wrong passage number
    [check] = check_citations(answer("The fusion formula uses k = 60 [2]."))
    assert check.status == "unsupported"
    assert check.best_source == 1


def test_uncited_sentence_is_reported():
    [check] = check_citations(answer("You need Python 3.11 and Ollama."))
    assert check.status == "uncited" and check.best_source == 3


def test_every_citation_in_a_sentence_must_support_it():
    [check] = check_citations(answer("Install Python 3.11 and run Ollama [3][2]."))
    assert check.cited == [3, 2] and check.status == "unsupported"


def test_threshold_is_configurable():
    text = "The fusion formula uses k = 60 [2]."
    assert check_citations(answer(text), min_support=0.0)[0].status == "supported"


def test_refusals_and_empty_answers_are_not_checked():
    assert check_citations(answer(NO_ANSWER)) == []
    assert check_citations(make_answer("q", NO_ANSWER, [], "fake-model")) == []
