"""The settings config.py uses when neither the environment nor .env sets them,
and how it reads the ones that aren't plain words or numbers."""
import pytest

from app.config import Settings


def test_agent_switches_default_to_what_the_phase_4_eval_chose(monkeypatch):
    """No query rewrite, grading on, not strict: see the comments in config.py."""
    for name in ("QUERY_REWRITE", "GRADING", "REFUSE_ON_FAILED_GRADE"):
        monkeypatch.delenv(name, raising=False)     # conftest.py pins them for the other tests
    s = Settings()
    assert (s.query_rewrite, s.grading, s.refuse_on_failed_grade) == (False, True, False)


def test_llm_extra_body_is_read_as_a_json_object(monkeypatch):
    monkeypatch.setenv("LLM_EXTRA_BODY", ' {"thinking": {"type": "disabled"}} ')
    assert Settings().llm_extra_body == {"thinking": {"type": "disabled"}}


@pytest.mark.parametrize("raw", [None, "", "   "])
def test_llm_extra_body_is_empty_when_not_set(monkeypatch, raw):
    if raw is None:
        monkeypatch.delenv("LLM_EXTRA_BODY", raising=False)
    else:
        monkeypatch.setenv("LLM_EXTRA_BODY", raw)
    assert Settings().llm_extra_body == {}


@pytest.mark.parametrize("raw", ['{"thinking": ', '["thinking"]', '"disabled"', "{'thinking': 1}"])
def test_llm_extra_body_must_be_a_json_object(monkeypatch, raw):
    monkeypatch.setenv("LLM_EXTRA_BODY", raw)
    with pytest.raises(ValueError, match="LLM_EXTRA_BODY"):
        Settings()
