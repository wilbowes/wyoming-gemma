#!/usr/bin/env python3
"""Wyoming speech-to-text backed by gemma's audio input on llama-server.

HA's Assist pipeline speaks Wyoming to its STT. This takes the audio HA
streams, wraps it as a 16 kHz mono WAV, and asks gemma (already serving HA's
conversation agent) to transcribe it. One model hears and thinks, and
whisper's VRAM comes back.

Measured 2026-09-27 against whisper on 25 saved Echo utterances: 20/22 clear
requests matched, ~250 ms a clip. Two prompt details decide whether it works:
the instruction goes BEFORE the audio, and a system line says a clip is
attached. Without them gemma answers the clip instead of transcribing it, or
claims no audio arrived.

Two ways it fails, both handled here rather than handed to HA as a command:
  - near-silence: gemma invents words (so does whisper), so a quiet clip is
    an empty transcript without asking;
  - refusal: "I'm unable to hear audio…" is gemma talking, not the user.

The language is the one HA names in its transcribe event, per request. The
measurements above are English only.
"""
import argparse, asyncio, audioop, base64, io, json, logging, re, time, urllib.request, wave

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import AsrModel, AsrProgram, Attribution, Describe, Info
from wyoming.server import AsyncEventHandler, AsyncServer

log = logging.getLogger("wyoming-gemma")

SYSTEM = "Each user message carries an audio clip recorded by a smart speaker microphone. Listen to it."
# Named in the prompt by the primary subtag of HA's language ("en-GB" is "en").
# A language not listed here is transcribed with no language named.
LANGUAGES = {
    "ar": "Arabic", "bg": "Bulgarian", "bn": "Bengali", "ca": "Catalan", "cs": "Czech",
    "da": "Danish", "de": "German", "el": "Greek", "en": "English", "es": "Spanish",
    "et": "Estonian", "fa": "Persian", "fi": "Finnish", "fr": "French", "he": "Hebrew",
    "hi": "Hindi", "hr": "Croatian", "hu": "Hungarian", "id": "Indonesian", "it": "Italian",
    "ja": "Japanese", "ko": "Korean", "lt": "Lithuanian", "lv": "Latvian", "ms": "Malay",
    "nb": "Norwegian", "nl": "Dutch", "no": "Norwegian", "pl": "Polish", "pt": "Portuguese",
    "ro": "Romanian", "ru": "Russian", "sk": "Slovak", "sl": "Slovenian", "sr": "Serbian",
    "sv": "Swedish", "sw": "Swahili", "ta": "Tamil", "th": "Thai", "tr": "Turkish",
    "uk": "Ukrainian", "ur": "Urdu", "vi": "Vietnamese", "zh": "Chinese",
}


def ask(language) -> str:
    name = LANGUAGES.get(re.split(r"[-_]", language or "")[0].lower())
    return ("Transcribe exactly what is said in this audio. "
            + (f"The speaker speaks {name}. " if name else "")
            + "Reply with the words only.")

# gemma speaking about the request rather than transcribing it
REFUSAL = re.compile(r"\b(cannot|can't|unable to) (fulfil|hear|listen|process|transcribe)|"
                     r"\baudio (files?|clips?)\b|\bno audio\b|provide the audio|\bas an ai\b", re.I)
# gemma 4's reasoning channel leaking despite thinking off: "<|channel>thought…"
LEAK = re.compile(r"<\|?[a-z_]+\|?>")


def transcribe(url: str, wav: bytes, timeout: float, language=None, temperature: float = 0.0) -> str:
    req = {"messages": [
               {"role": "system", "content": SYSTEM},
               {"role": "user", "content": [
                   {"type": "text", "text": ask(language)},
                   {"type": "input_audio", "input_audio": {
                       "data": base64.b64encode(wav).decode(), "format": "wav"}}]}],
           # A 10 s utterance is ~30 words; the cap bounds a runaway, it never trims speech.
           "max_tokens": 80, "temperature": temperature,
           "chat_template_kwargs": {"enable_thinking": False}}
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        url + "/v1/chat/completions", json.dumps(req).encode(),
        {"Content-Type": "application/json"}), timeout=timeout))
    return (r["choices"][0]["message"].get("content") or "").strip()


class Handler(AsyncEventHandler):
    def __init__(self, info: Info, args, *a, **kw):
        super().__init__(*a, **kw)
        self.info, self.args = info, args
        self.conv = AudioChunkConverter(rate=16000, width=2, channels=1)
        self.pcm = bytearray()
        self.language = args.languages[0]
        self.limit = int(args.max_seconds * 32000)

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self.info.event())
            return True
        if Transcribe.is_type(event.type):
            self.language = Transcribe.from_event(event).language or self.args.languages[0]
            return True
        if AudioStart.is_type(event.type):
            self.pcm.clear()
            return True
        if AudioChunk.is_type(event.type):
            # A client that never sends audio-stop must not grow this without
            # limit: past --max-seconds the rest is dropped, not buffered.
            if len(self.pcm) < self.limit:
                self.pcm += self.conv.convert(AudioChunk.from_event(event)).audio
                if len(self.pcm) >= self.limit:
                    del self.pcm[self.limit:]
                    log.warning("audio passed --max-seconds %g, the rest is dropped", self.args.max_seconds)
            return True
        if AudioStop.is_type(event.type):
            text = await asyncio.get_running_loop().run_in_executor(
                None, self._run, bytes(self.pcm), self.language)
            await self.write_event(Transcript(text=text).event())
            self.pcm.clear()
            return False
        return True

    def _run(self, pcm: bytes, language) -> str:
        secs = len(pcm) / 32000
        rms = audioop.rms(pcm, 2) if pcm else 0
        if rms < self.args.min_rms:
            log.info("%.2fs rms %d: below --min-rms, empty transcript", secs, rms)
            return ""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(pcm)
        t0 = time.monotonic()
        try:
            text = transcribe(self.args.gemma_url, buf.getvalue(), self.args.timeout, language)
            # gemma sometimes insists no audio arrived, and at temperature 0 it
            # insists every time for that clip; a little sampling gets past it.
            if REFUSAL.search(text):
                log.info("%.2fs rms %d: refused, retrying: %r", secs, rms, text[:80])
                text = transcribe(self.args.gemma_url, buf.getvalue(), self.args.timeout, language, 0.4)
        except Exception as e:  # HA gets an empty transcript, never a hang
            log.error("%.2fs rms %d: gemma failed: %s", secs, rms, e)
            return ""
        ms = (time.monotonic() - t0) * 1000
        if LEAK.search(text):
            log.info("%.2fs rms %d %.0fms: control tokens leaked, empty transcript: %r", secs, rms, ms, text[:80])
            return ""
        if REFUSAL.search(text):
            log.info("%.2fs rms %d %.0fms: refusal, empty transcript: %r", secs, rms, ms, text)
            return ""
        log.info("%.2fs rms %d %.0fms %s: %r", secs, rms, ms, language, text)
        return text


def make_info(args) -> Info:
    return Info(asr=[AsrProgram(
        name="gemma-audio", description=f"{args.model_name} audio input via llama-server",
        attribution=Attribution(name="EchoMuse", url="https://github.com/wilbowes/wyoming-gemma"),
        installed=True, version="0.2.0",
        models=[AsrModel(name=args.model_name, description=f"{args.model_name} (audio)",
                         attribution=Attribution(name="Google", url="https://ai.google.dev/gemma"),
                         installed=True, languages=args.languages, version="4")])])


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default="tcp://0.0.0.0:10302")
    ap.add_argument("--gemma-url", default="http://127.0.0.1:8000")
    ap.add_argument("--min-rms", type=int, default=100,
                    help="clips quieter than this are silence (a saved near-silent clip measured 26)")
    ap.add_argument("--timeout", type=float, default=15)
    ap.add_argument("--max-seconds", type=float, default=30,
                    help="audio kept per request; the rest is dropped")
    ap.add_argument("--languages", nargs="+", default=["en"], metavar="CODE",
                    help="languages offered to Home Assistant; the first is used when a request names none")
    ap.add_argument("--model-name", default="gemma-4-12B",
                    help="the model llama-server is serving, as Home Assistant will show it")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    info = make_info(args)
    log.info("listening on %s, gemma at %s, languages %s", args.uri, args.gemma_url, args.languages)
    await AsyncServer.from_uri(args.uri).run(lambda *a, **kw: Handler(info, args, *a, **kw))

if __name__ == "__main__":
    asyncio.run(main())
