#!/usr/bin/env python3
"""Stream saved clips over Wyoming, as HA does, to whisper and the shim.

    HOST=192.168.1.10 python3 compare.py 25    # the newest 25 of /rec/*.wav
"""
import asyncio, glob, os, sys, time, wave
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient

async def stt(port, path):
    with wave.open(path) as w:
        rate, width, ch, pcm = w.getframerate(), w.getsampwidth(), w.getnchannels(), w.readframes(w.getnframes())
    t0 = time.monotonic()
    async with AsyncTcpClient(os.environ.get("HOST", "127.0.0.1"), port) as c:
        await c.write_event(Transcribe(language="en").event())
        await c.write_event(AudioStart(rate=rate, width=width, channels=ch).event())
        for i in range(0, len(pcm), 2048):
            await c.write_event(AudioChunk(rate=rate, width=width, channels=ch, audio=pcm[i:i+2048]).event())
        await c.write_event(AudioStop().event())
        while True:
            e = await c.read_event()
            if e and Transcript.is_type(e.type):
                return Transcript.from_event(e).text, (time.monotonic() - t0) * 1000

async def main():
    for p in sorted(glob.glob("/rec/*.wav"))[-int(sys.argv[1]):]:
        (w, wm), (g, gm) = await stt(10300, p), await stt(10302, p)
        print(f"{os.path.basename(p):28} whisper {wm:4.0f}ms {w!r:30} gemma {gm:4.0f}ms {g!r}")
asyncio.run(main())
