import pytest

from conftest import FakeEmbedder, make_chunk
from hybrid_rag.indexing import BM25Index, IndexStore, tokenize

CHUNKS = [
    make_chunk("cats.txt::c0", "Cats are small domesticated carnivorous mammals that purr."),
    make_chunk("errors.pdf::p2::c0", "Windows error 0x80070005 means access is denied.", page=2),
    make_chunk("bikes.md::c0", "Repairing a punctured bicycle tire needs a patch kit and pump."),
]


@pytest.fixture
def store(settings):
    store = IndexStore("test_docs", settings=settings, embedder=FakeEmbedder())
    store.build(CHUNKS)
    return store


def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("The Error is 0x80070005 in Windows!") == ["error", "0x80070005", "windows"]


def test_tokenize_with_stemming_matches_word_forms():
    assert tokenize("Vitamins and diabetic studies", stem=True) == ["vitamin", "diabet", "studi"]
    assert tokenize("Vitamin, diabetes, study", stem=True) == ["vitamin", "diabet", "studi"]
    assert tokenize("error 0x80070005", stem=True) == ["error", "0x80070005"]  # codes untouched


def test_stemmed_bm25_matches_plural_query(tmp_path):
    plain, stemmed = BM25Index(tmp_path / "plain.json"), BM25Index(tmp_path / "stem.json", stem=True)
    for index in (plain, stemmed):
        index.add(CHUNKS)
    assert plain.search("punctured tires", top_k=1)[0][0] == "bikes.md::c0"  # via "punctured"
    assert plain.search("tires repaired", top_k=1) == []  # no exact word in common
    assert stemmed.search("tires repaired", top_k=1)[0][0] == "bikes.md::c0"  # tire/repair roots


def test_bm25_file_records_its_tokenizer(tmp_path):
    path = tmp_path / "bm25.json"
    BM25Index(path, stem=True).add(CHUNKS)

    assert not BM25Index(path, stem=True).needs_rebuild
    mismatched = BM25Index(path, stem=False)
    assert mismatched.needs_rebuild and mismatched.count() == 0


def test_store_retokenizes_bm25_when_stemming_setting_changes(tmp_path, fake_embedder):
    from hybrid_rag.config import Settings

    plain = Settings(_env_file=None, data_dir=tmp_path, bm25_stemming=False)
    IndexStore("docs", settings=plain, embedder=fake_embedder).build(CHUNKS)

    stemmed = Settings(_env_file=None, data_dir=tmp_path, bm25_stemming=True)
    store = IndexStore("docs", settings=stemmed, embedder=fake_embedder)

    assert store.sparse.count() == len(CHUNKS) and not store.sparse.needs_rebuild
    assert store.sparse.search("tires repaired", top_k=1)[0][0] == "bikes.md::c0"


def test_dense_all_chunks_pages_through_everything(store):
    chunks = store.dense.all_chunks(batch_size=2)
    assert sorted(c.id for c in chunks) == sorted(c.id for c in CHUNKS)
    assert next(c for c in chunks if c.id == "errors.pdf::p2::c0").page == 2


def test_bm25_finds_exact_code(tmp_path):
    index = BM25Index(tmp_path / "bm25.json")
    index.add(CHUNKS)
    results = index.search("what does 0x80070005 mean", top_k=3)
    assert results[0][0] == "errors.pdf::p2::c0"
    assert all(score > 0 for _, score in results)


def test_bm25_persists_and_resets(tmp_path):
    path = tmp_path / "bm25.json"
    BM25Index(path).add(CHUNKS)

    reloaded = BM25Index(path)
    assert reloaded.count() == 3
    assert reloaded.search("bicycle patch", top_k=1)[0][0] == "bikes.md::c0"

    reloaded.reset()
    assert not path.exists()
    assert reloaded.search("bicycle", top_k=1) == []


def test_bm25_empty_or_stopword_query(tmp_path):
    index = BM25Index(tmp_path / "bm25.json")
    assert index.search("cats", top_k=5) == []
    index.add(CHUNKS)
    assert index.search("the and of", top_k=5) == []


def test_dense_search_ranks_most_similar_first(store):
    results = store.dense.search("bicycle tire repair", top_k=3)
    assert results[0][0] == "bikes.md::c0"
    assert results[0][1] > results[-1][1]


def test_get_chunks_keeps_requested_order_and_metadata(settings):
    store = IndexStore("meta_docs", settings=settings, embedder=FakeEmbedder())
    store.build([
        make_chunk("a.pdf::p1::c0", "alpha text", page=1, title="Alpha", tags=["x"]),
        make_chunk("b.txt::c0", "beta text"),
    ])

    b, a = store.get_chunks(["b.txt::c0", "a.pdf::p1::c0", "missing::c0"])

    assert (b.id, b.page, b.source) == ("b.txt::c0", None, "b.txt")
    assert (a.id, a.page, a.text) == ("a.pdf::p1::c0", 1, "alpha text")
    assert a.metadata == {"title": "Alpha"}  # list values can't be stored in ChromaDB


def test_build_replaces_previous_contents(store):
    store.build(CHUNKS[:1])
    assert store.dense.count() == 1 and store.sparse.count() == 1


def test_adding_duplicate_ids_is_rejected(store):
    with pytest.raises(ValueError, match="already indexed"):
        store.add(CHUNKS[:1])


def test_collections_are_isolated(store, settings):
    other = IndexStore("other_docs", settings=settings, embedder=FakeEmbedder())
    assert other.count() == 0
    assert store.count() == 3


def test_index_survives_reopening(store, settings):
    reopened = IndexStore("test_docs", settings=settings, embedder=FakeEmbedder())
    assert reopened.count() == 3
    assert reopened.sparse.search("purr", top_k=1)[0][0] == "cats.txt::c0"
