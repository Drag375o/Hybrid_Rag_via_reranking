import pytest
from pydantic import ValidationError

from hybrid_rag.config import PROJECT_ROOT, Settings


def test_defaults_are_valid():
    settings = Settings(_env_file=None)
    assert settings.rerank_top_k <= settings.fusion_top_k
    assert settings.chunk_overlap < settings.chunk_size
    assert settings.docs_dir == PROJECT_ROOT / "data" / "docs"


def test_env_override(monkeypatch):
    monkeypatch.setenv("RAG_LLM_MODEL", "llama3.1:8b")
    monkeypatch.setenv("RAG_RERANK_TOP_K", "8")
    settings = Settings(_env_file=None)
    assert settings.llm_model == "llama3.1:8b"
    assert settings.rerank_top_k == 8


def test_relative_data_dir_resolves_to_project_root():
    settings = Settings(_env_file=None, data_dir="my_data")
    assert settings.data_dir == PROJECT_ROOT / "my_data"
    assert settings.index_dir == PROJECT_ROOT / "my_data" / "index"


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, chunk_size=100, chunk_overlap=100)


def test_rerank_top_k_cannot_exceed_fusion_top_k():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, fusion_top_k=5, rerank_top_k=10)
