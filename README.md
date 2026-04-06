# Discord AI Voicebot v2

Current repo state: April 6, 2026.

This repo is the Discord bot in `Main.py`, using LM Studio for the LLM and Inworld streaming TTS for low-latency replies, with Piper as fallback.

## 2026 Status

What is current in this repo now:

- `discord.py` and `discord-ext-voice-recv` were updated to current upstream commits
- LM Studio is the primary LLM backend
- Inworld websocket TTS is the primary speech backend
- Piper remains available as a local fallback TTS backend
- direct mention replies and voice-channel replies use the same provider selection flow

Important caveat:

- Discord voice receive is better than the old 2025 setup, but it is still not fully clean on current Discord voice protocol changes. This repo includes local compatibility patches so the bot can join, listen, and often respond, but `discord-ext-voice-recv` may still drop undecodable incoming packets on some runs.
- you do not need to manually patch installed packages for this repo; the local compatibility patching is applied at runtime from `Main.py`

## Features

- LM Studio chat via OpenAI-compatible local server
- Inworld streaming TTS for low-latency speech output
- Piper fallback TTS
- Discord text replies and voice-channel replies
- Whisper-based transcription for voice input
- optional memory store with fail-open behavior if embedding dependencies are broken

## Prerequisites

- Python 3.8+
- Git
- LM Studio with a model loaded and local server enabled at `http://127.0.0.1:1234/v1`
- Discord bot token if you want to run the Discord bot
- Inworld API key if you want streaming TTS
- `ffmpeg` on `PATH` for voice/audio processing

## Install

```bash
git clone https://github.com/xsploit/Discord-AI-Voicebot-v2.git
cd Discord-AI-Voicebot-v2
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Windows quick setup:

```powershell
.\setup_bot.ps1
```

That script:

- creates `.venv` unless you pass `-SkipVenv`
- installs the pinned requirements
- optionally installs CUDA PyTorch with `-InstallCudaTorch`
- runs `doctor_bot.py` to verify the environment

If you need CUDA PyTorch for local embedding or other GPU-backed pieces:

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Environment

Copy `.env.example` to `.env`.

Current defaults are:

```env
LLM_PROVIDER=lmstudio
TTS_PROVIDER=inworld
LM_STUDIO_BASE_URL=http://127.0.0.1:1234/v1
LM_STUDIO_MODEL=
LM_STUDIO_API_KEY=lm-studio
INWORLD_TTS_MODEL_ID=inworld-tts-1.5-max
INWORLD_TTS_SAMPLE_RATE_HZ=48000
```

Token notes:

- the bot accepts either `DISCORD_BOT_TOKEN` or `DISCORD_TOKEN`
- if `LM_STUDIO_MODEL` is blank, the first model returned by `/v1/models` is used
- if `TTS_PROVIDER=piper`, the local Piper paths in `Main.py` are used

## Run The Discord Bot

Start LM Studio first, make sure a model is loaded, then run:

```bash
python Main.py
```

Main commands:

- `!vc` joins your current voice channel
- `!stop` disconnects from voice
- `!die` shuts the bot down
- `@mention` triggers a text reply

## Discord Voice Notes

Current behavior:

- voice connect and playback are updated for newer Discord voice changes
- receive-side packet handling is patched locally to avoid hard crashes on some undecodable packets
- the bot can still log dropped packet warnings during voice receive

If the bot joins and replies but transcription quality is inconsistent, that is usually receive-side packet corruption in `discord-ext-voice-recv`, not LM Studio or Inworld.

## Troubleshooting

### LM Studio

Check that LM Studio is actually serving a model:

```bash
python -c "import json, urllib.request; print(json.load(urllib.request.urlopen('http://127.0.0.1:1234/v1/models')))"
```

If the configured model name does not match, set `LM_STUDIO_MODEL` explicitly in `.env`.

### Environment Doctor

You can verify a machine with:

```bash
python doctor_bot.py
```

Or on Windows through:

```powershell
.\setup_bot.ps1
```

### Inworld TTS

If TTS fails:

- verify `INWORLD_TTS_API_KEY`
- verify `INWORLD_TTS_VOICE_ID`
- verify `INWORLD_TTS_MODEL_ID`

### Piper

If you switch to Piper:

- verify the executable path exists
- verify the `.onnx` model path exists
- verify the `.onnx.json` config path exists

### Memory Store

If `sentence-transformers`, `torch`, or `torchvision` are mismatched, the bot now degrades gracefully and disables memory embeddings instead of crashing startup.

## Files

- `Main.py`: Discord bot
- `setup_bot.ps1`: Windows setup helper
- `doctor_bot.py`: environment verification script
- `.env.example`: example provider config
- `requirements.txt`: primary install file

## License

MIT
