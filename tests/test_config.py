"""The settings config.py uses when neither the environment nor .env sets them."""
from app.config import Settings


def test_agent_switches_default_to_what_the_phase_4_eval_chose(monkeypatch):
    """No query rewrite, grading on, not strict: see the comments in config.py."""
    for name in ("QUERY_REWRITE", "GRADING", "REFUSE_ON_FAILED_GRADE"):
        monkeypatch.delenv(name, raising=False)     # conftest.py pins them for the other tests
    s = Settings()
    assert (s.query_rewrite, s.grading, s.refuse_on_failed_grade) == (False, True, False)
