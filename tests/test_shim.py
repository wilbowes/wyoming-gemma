"""Drive the shim over Wyoming, as Home Assistant does, against a scripted
stand-in for llama-server. Needs: pytest wyoming==1.8.0 (Python 3.12)."""
import asyncio, base64, io, json, math, os, struct, sys, threading, wave
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

import pytest
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.info import Describe, Info
from wyoming.server import AsyncServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import shim


class Llama:
    """Answers each chat completion with the next scripted reply."""

    def __init__(self, replies):
        self.replies, self.requests = list(replies), []
        outer = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                outer.requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                body = json.dumps({"choices": [{"message": {"content": outer.replies.pop(0)}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()


def tone(seconds=1.0, rate=16000, channels=1, amplitude=8000):
    n = int(seconds * rate)
    return b"".join(struct.pack("<h", int(amplitude * math.sin(2 * math.pi * 440 * i / rate))) * channels
                    for i in range(n))


def ask(url, pcm, rate=16000, channels=1, min_rms=100):
    """One Home Assistant request; returns the transcript text."""
    async def run():
        args = SimpleNamespace(gemma_url=url, min_rms=min_rms, timeout=5)
        server = AsyncServer.from_uri("tcp://127.0.0.1:0")
        await server.start(lambda *a, **kw: shim.Handler(Info(), args, *a, **kw))
        port = server._server.sockets[0].getsockname()[1]
        try:
            async with AsyncTcpClient("127.0.0.1", port) as c:
                await c.write_event(Describe().event())
                assert (await c.read_event()).type == "info"
                await c.write_event(Transcribe(language="en").event())
                await c.write_event(AudioStart(rate=rate, width=2, channels=channels).event())
                step = 2048 * channels
                for i in range(0, len(pcm), step):
                    await c.write_event(AudioChunk(rate=rate, width=2, channels=channels,
                                                   audio=pcm[i:i + step]).event())
                await c.write_event(AudioStop().event())
                return Transcript.from_event(await asyncio.wait_for(c.read_event(), 10)).text
        finally:
            await server.stop()
    return asyncio.run(run())


def sent_wav(request):
    part = request["messages"][1]["content"][1]
    assert part["type"] == "input_audio" and part["input_audio"]["format"] == "wav"
    with wave.open(io.BytesIO(base64.b64decode(part["input_audio"]["data"]))) as w:
        return w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()


def test_transcript_is_returned_and_the_request_is_shaped_as_gemma_needs():
    llama = Llama(["turn on the kitchen light"])
    assert ask(llama.url, tone(1.0)) == "turn on the kitchen light"
    (req,) = llama.requests
    assert req["messages"][0]["role"] == "system"
    # The instruction goes BEFORE the audio, or gemma answers the clip.
    assert [p["type"] for p in req["messages"][1]["content"]] == ["text", "input_audio"]
    assert req["chat_template_kwargs"] == {"enable_thinking": False}
    assert req["temperature"] == 0.0
    assert sent_wav(req) == (16000, 1, 2, 16000)


def test_any_input_format_reaches_gemma_as_16k_mono():
    llama = Llama(["hello"])
    assert ask(llama.url, tone(1.0, rate=48000, channels=2), rate=48000, channels=2) == "hello"
    rate, channels, width, frames = sent_wav(llama.requests[0])
    assert (rate, channels, width) == (16000, 1, 2)
    assert abs(frames - 16000) <= 16


def test_silence_is_an_empty_transcript_without_asking():
    llama = Llama([])
    assert ask(llama.url, tone(1.0, amplitude=20)) == ""
    assert llama.requests == []


def test_no_audio_at_all_is_an_empty_transcript():
    llama = Llama([])
    assert ask(llama.url, b"") == ""
    assert llama.requests == []


def test_a_refusal_is_retried_once_with_sampling():
    llama = Llama(["I'm unable to hear audio.", "what time is it"])
    assert ask(llama.url, tone()) == "what time is it"
    assert [r["temperature"] for r in llama.requests] == [0.0, 0.4]


def test_a_second_refusal_is_an_empty_transcript():
    llama = Llama(["I cannot process audio files.", "Please provide the audio."])
    assert ask(llama.url, tone()) == ""
    assert len(llama.requests) == 2


def test_leaked_control_tokens_are_an_empty_transcript():
    llama = Llama(["<|channel>thought the user said hello"])
    assert ask(llama.url, tone()) == ""


def test_llama_down_is_an_empty_transcript_not_a_hang():
    assert ask("http://127.0.0.1:1", tone()) == ""


@pytest.mark.parametrize("text", [
    "I'm unable to hear audio", "I cannot transcribe that", "Please provide the audio.",
    "There is no audio in your message", "As an AI, I can't listen to clips"])
def test_refusals_are_recognised(text):
    assert shim.REFUSAL.search(text)


@pytest.mark.parametrize("text", [
    "turn on the kitchen light", "I can't find my keys", "play some music in the lounge",
    "what's the weather tomorrow"])
def test_ordinary_requests_are_not_refusals(text):
    assert not shim.REFUSAL.search(text) and not shim.LEAK.search(text)
