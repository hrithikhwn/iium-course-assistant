"""API tests with a fake agent — no Ollama or Chroma needed."""
import pytest
from fastapi.testclient import TestClient

from app import main
from app.agent import ASK_STUDENT, Result, Step
from app.text import Chunk


class FakeAgent:
    def __init__(self, result=None, error=None):
        self.calls, self.result, self.error = [], result, error

    def run(self, question, batch=None, programme=None):
        self.calls.append((question, batch, programme))
        if self.error:
            raise self.error
        if batch is None or programme is None:
            return Result(ASK_STUDENT, [], [Step("student", "missing batch/programme")], needs_student_info=True)
        return self.result


@pytest.fixture
def client(monkeypatch):
    agent = FakeAgent(Result("FYP I needs 90 credits [1].", [Chunk("FYP text", "plan.pdf", 1, "241", "BCS")],
                             [Step("student", "BCS 241 -> plan 241")]))
    monkeypatch.setattr(main, "get_agent", lambda: agent)
    return TestClient(main.app), agent


def test_ask_passes_programme_and_batch_to_the_agent(client):
    http, agent = client
    body = http.post("/ask", json={"question": "When can I take FYP?", "programme": "BCS", "batch": 241}).json()
    assert agent.calls == [("When can I take FYP?", 241, "BCS")]
    assert body["answer"].endswith("[1].") and body["found"] and not body["needs_student_info"]
    assert body["sources"] == [{"ref": 1, "citation": "plan.pdf, p.1", "excerpt": "FYP text"}]


def test_ask_without_student_info_says_so(client):
    http, _ = client
    body = http.post("/ask", json={"question": "When can I take FYP?"}).json()
    assert body["needs_student_info"] and body["answer"] == ASK_STUDENT and body["sources"] == []


def test_batch_may_be_a_matric_number_string(client):
    http, agent = client
    http.post("/ask", json={"question": "When can I take FYP?", "programme": "BIT", "batch": "2410123"})
    assert agent.calls[-1][1] == "2410123"


def test_empty_vector_store_is_503(monkeypatch):
    monkeypatch.setattr(main, "get_agent", lambda: FakeAgent(error=RuntimeError("Vector store is empty")))
    resp = TestClient(main.app).post("/ask", json={"question": "anything", "programme": "BCS", "batch": 241})
    assert resp.status_code == 503
