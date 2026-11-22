"""The LLM client against a fake server (httpx.MockTransport): no network, no Ollama."""
import json

import httpx
import pytest

from app.llm import OpenAICompatibleLLM


def reply(text: str = "ok", status: int = 200, headers: dict | None = None) -> httpx.Response:
    return httpx.Response(status, json={"choices": [{"message": {"content": text}}]}, headers=headers)


def fake_server(*answers, extra_body=None, **kwargs):
    """A client whose server gives `answers` in order (a response, or an
    exception to raise), and the lists of requests it got and seconds it waited."""
    requests, waits, queue = [], [], list(answers)

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        answer = queue.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    llm = OpenAICompatibleLLM(base_url="https://llm.test/v1/", model="some-model", api_key="test-key",
                              extra_body=extra_body or {}, transport=httpx.MockTransport(handle),
                              sleep=waits.append, **kwargs)
    return llm, requests, waits


def test_sends_an_openai_style_chat_request():
    llm, requests, waits = fake_server(reply("  the answer \n"))
    assert llm.chat("be brief", "what?", temperature=0.2) == "the answer"
    (request,) = requests
    assert str(request.url) == "https://llm.test/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer test-key"
    assert json.loads(request.content) == {
        "model": "some-model",
        "temperature": 0.2,
        "messages": [{"role": "system", "content": "be brief"}, {"role": "user", "content": "what?"}],
    }
    assert waits == []


def test_extra_body_goes_with_every_request():
    """DeepSeek thinks before it answers unless the request turns that off."""
    llm, requests, _ = fake_server(reply(), reply(), extra_body={"thinking": {"type": "disabled"}})
    llm.chat("s", "first")
    llm.chat("s", "second")
    assert [json.loads(r.content)["thinking"] for r in requests] == [{"type": "disabled"}] * 2


def test_reads_a_reply_after_keep_alive_blank_lines():
    """A busy DeepSeek server sends empty lines until it starts answering."""
    body = b"\n\n\n" + json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()
    llm, _, _ = fake_server(httpx.Response(200, content=body))
    assert llm.chat("s", "u") == "ok"


@pytest.mark.parametrize("header, wait", [
    ("3", 3.0),                                   # seconds, as the server asked
    ("600", 60.0),                                # but never more than a minute
    ("Sun, 22 Nov 2026 07:28:00 GMT", 2.0),       # a date: the usual backoff instead
])
def test_rate_limit_waits_as_long_as_retry_after_says(header, wait, caplog):
    llm, requests, waits = fake_server(reply(status=429, headers={"Retry-After": header}), reply("ok"))
    assert llm.chat("s", "u") == "ok"
    assert (len(requests), waits) == (2, [wait])
    assert "HTTP 429" in caplog.text


def test_server_errors_back_off_2_then_4_seconds():
    llm, requests, waits = fake_server(reply(status=503), reply(status=500), reply("ok"))
    assert llm.chat("s", "u") == "ok"
    assert (len(requests), waits) == (3, [2.0, 4.0])


def test_dropped_connections_and_timeouts_are_tried_again():
    llm, requests, waits = fake_server(httpx.ConnectError("refused"), httpx.ReadTimeout("slow"), reply("ok"))
    assert llm.chat("s", "u") == "ok"
    assert (len(requests), waits) == (3, [2.0, 4.0])


def test_gives_up_after_the_last_try():
    llm, requests, waits = fake_server(*[reply(status=429)] * 3, max_attempts=3)
    with pytest.raises(httpx.HTTPStatusError, match="429"):
        llm.chat("s", "u")
    assert (len(requests), waits) == (3, [2.0, 4.0])


def test_a_connection_that_keeps_failing_raises_its_error():
    llm, requests, _ = fake_server(*[httpx.ConnectError("refused")] * 2, max_attempts=2)
    with pytest.raises(httpx.ConnectError):
        llm.chat("s", "u")
    assert len(requests) == 2


@pytest.mark.parametrize("status", [400, 401, 402, 422])
def test_errors_that_would_happen_again_are_not_retried(status):
    """A bad request, a wrong key, an empty balance or invalid parameters."""
    llm, requests, waits = fake_server(reply(status=status), reply("never asked"))
    with pytest.raises(httpx.HTTPStatusError, match=str(status)):
        llm.chat("s", "u")
    assert (len(requests), waits) == (1, [])
