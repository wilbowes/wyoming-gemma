# wyoming-gemma

Speech-to-text for Home Assistant from Gemma 4's audio input, over the
[Wyoming protocol](https://github.com/OHF-Voice/wyoming).

If Gemma 4 on `llama-server` is already your conversation agent, it can do the
listening as well, and a separate whisper model is no longer needed. This is a
133-line shim: it takes the audio Home Assistant streams, wraps it as a 16 kHz
mono WAV, and asks Gemma to transcribe it through `/v1/chat/completions`. The
shim itself needs no GPU.

It came out of [EchoMuse](https://github.com/wilbowes/EchoMuse), where it has
been the only speech-to-text on one household's Echo Dots since late
September 2026.

## What to expect

Measured on 25 saved utterances from Echo Dot microphones, with
gemma-4-12B-it (QAT, Q4_K_XL) on an RTX 5060 Ti: 20 of 22 clear requests
transcribed the same as faster-whisper `distil-large-v3`, at about 250 ms a
clip. That is one room, one model and a small sample.

### Other languages

English is what it has been used for. For a first look at the rest, two
short commands in each of 20 languages were synthesised with Piper and sent
through the shim. Whisper `large-v3-turbo` heard the same clips as a check on
the clips themselves, and got all 40 right or within a word or two.

| Gemma's transcripts | Languages |
|---|---|
| Both exact | German, Spanish, Portuguese, Russian, Arabic, Hindi |
| One exact, one with an error | Italian, Chinese, Turkish, Indonesian |
| Close, with a wrong word | Dutch, Polish, Ukrainian |
| Both wrong | Swedish, Finnish, Hungarian, Greek, Persian, Vietnamese, Swahili |

French, tried separately on three sentences, was two exact and one with a
wrong word.

A wrong word can be the opposite command: the Ukrainian for "turn on" came
back as "turn off". This is 40 clean synthetic clips on gemma-4-12B, so read
it as where to start testing, and try your own voice before relying on a
language.

Naming the language matters outside the first row. With English named in the
prompt, the languages in the last two rows mostly came back as an English
sentence, as another language, or as a refusal. That is why each request is
transcribed in the language Home Assistant names for it.

How it fails:

- A clip quieter than `--min-rms` (default 100) returns an empty transcript
  without asking the model, because Gemma invents words for near-silence.
- If Gemma answers about the request ("I'm unable to hear audio") the shim
  retries once at temperature 0.4, and returns an empty transcript if it
  refuses again.
- If `llama-server` is down or slow (`--timeout`, default 15 s), Home
  Assistant gets an empty transcript.
- Audio past `--max-seconds` (default 30) is dropped.

## Requirements

`llama-server` (llama.cpp) serving a Gemma 4 model **with its multimodal
projector**, which is what gives it audio input:

```
llama-server --model gemma-4-12B-it-qat-UD-Q4_K_XL.gguf --mmproj mmproj-BF16.gguf \
  --jinja --host 0.0.0.0 --port 8000
```

## Run

```
git clone https://github.com/wilbowes/wyoming-gemma && cd wyoming-gemma
GEMMA_URL=http://192.168.1.10:8000 docker compose up -d --build
```

`GEMMA_URL` is where `llama-server` listens. Left unset, it is port 8000 on
the Docker host.

In Home Assistant: Settings → Devices & services → Add integration →
**Wyoming Protocol**, with this machine's address and port `10302`. Then pick
`gemma-audio` as the speech-to-text of your Assist pipeline.

## Options

Set them on `command:` in `docker-compose.yml`.

| Option | Default | |
|---|---|---|
| `--gemma-url` | `http://127.0.0.1:8000` | where `llama-server` listens |
| `--languages` | `en` | the languages offered to Home Assistant, e.g. `--languages en fr de`. A pipeline can only pick this engine in a language listed here. Each request is transcribed in the language Home Assistant names for it. |
| `--model-name` | `gemma-4-12B` | the model's name as Home Assistant shows it |
| `--min-rms` | `100` | clips quieter than this are treated as silence |
| `--max-seconds` | `30` | audio kept per request |
| `--timeout` | `15` | seconds to wait for `llama-server` |

Each request is logged with its length, level, time taken, language and transcript:
`docker logs -f wyoming-gemma`.

## Comparing against whisper

`compare.py` streams the newest N of `/rec/*.wav` to a Wyoming whisper on port
10300 and to this shim on 10302, and prints both transcripts with timings:

```
pip install wyoming==1.8.0
HOST=192.168.1.10 python3 compare.py 25
```

## Tests

`tests/test_shim.py` drives the shim over Wyoming against a scripted stand-in
for `llama-server`: the request shape, resampling, silence, refusals, leaked
control tokens, a server that is down, the language named per request and
the audio cap.

```
pip install pytest wyoming==1.8.0 && python -m pytest tests/
```

## Licence

MIT.
