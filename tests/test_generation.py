"""Tests for src/rag/generation.py's Groq-first, local-fallback routing.

Both backends are mocked: no API calls, no model downloads.
"""

import os
import sys

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rag import generation  # noqa: E402


@pytest.fixture
def backends(monkeypatch):
    calls = []

    def groq(prompt, model, max_tokens):
        calls.append(("groq", model))
        return "groq answer"

    def local(prompt, max_new_tokens):
        calls.append(("local", None))
        return "local answer"

    monkeypatch.setattr(generation, "_groq_generate", groq)
    monkeypatch.setattr(generation, "_local_generate", local)
    monkeypatch.setattr(generation, "GEN_BACKEND", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    return calls


def test_uses_groq_when_configured(backends):
    assert generation.generate("p", "qwen/x", 50) == ("groq answer", "qwen/x")
    assert backends == [("groq", "qwen/x")]


def test_falls_back_to_local_when_groq_fails(backends, monkeypatch):
    def broken(*_):
        raise RuntimeError("503 over capacity")

    monkeypatch.setattr(generation, "_groq_generate", broken)
    text, used = generation.generate("p", "qwen/x", 50)
    assert text == "local answer"
    assert used == generation.GEN_MODEL_NAME


def test_uses_local_without_api_key(backends, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY")
    assert generation.generate("p", "qwen/x", 50)[0] == "local answer"
    assert backends == [("local", None)]


def test_local_backend_never_calls_groq(backends, monkeypatch):
    monkeypatch.setattr(generation, "GEN_BACKEND", "local")
    generation.generate("p", "qwen/x", 50)
    assert backends == [("local", None)]
