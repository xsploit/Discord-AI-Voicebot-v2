# Discord AI Voicebot - Hikari-chan

**UPDATE (August 2025)**: Fixed Discord 4006 voice connection errors with latest discord.py version!

This is a **Discord bot** that combines **AI chat, voice recognition, and TTS** into one intelligent assistant. Meet **Hikari-chan** - an AI with Hinata's personality that can chat via text or voice, remember conversations, and respond with natural speech.

## Features:
- 🤖 **AI-Powered Chat**: Uses LM Studio's OpenAI-compatible local server for intelligent responses
- 🎙️ **Voice Recognition**: Listens to voice chat and responds naturally  
- 🔊 **Text-to-Speech**: Uses Inworld streaming TTS for low-latency voice replies, with Piper fallback
- 🧠 **Memory System**: Remembers past conversations for context
- 💬 **Text & Voice**: Works in both text channels and voice channels
- 🎯 **Smart Responses**: Only responds when mentioned or in voice chat

---

## Prerequisites

- **Python 3.8+** 
- **LM Studio** with a loaded chat model and local server enabled
- **Discord Bot Token** from [Discord Developer Portal](https://discord.com/developers/applications)
- **Inworld TTS API key** for streaming speech
- **Git** for installation

---

## Quick Installation (Windows)

### 1. Clone Repository
```bash
git clone https://github.com/xsploit/Discord-AI-Voicebot.git
cd Discord-AI-Voicebot
```

### 2. Set Up Virtual Environment
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies (Easy Method)
```bash
# Install exact working versions
pip install -r requirements.txt

# Install PyTorch with CUDA support (for RTX GPUs)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4. Configure Environment
Copy `.env.example` to `.env` and set your Discord token, LM Studio settings, and Inworld TTS key.

### 5. Start LM Studio
Load your model in LM Studio, then start the local server. The default config in `.env.example` expects `http://127.0.0.1:1234/v1`.

---

## Running the Bot

```bash
python Main.py
```

You should see:
```
Bot ready: Hikari-chan#1234
Commands: !vc, !stop, !die
```

---

## Standalone Talking Avatar

There is also a standalone PSD-based talking avatar renderer in [talking_avatar.py](/C:/Users/SUBSECT/Documents/GitHub/dvb/talking_avatar.py). This does not use Discord. It reuses the same LLM and TTS provider setup from this repo, then animates `mouth_open` and `mouth_closed` PSD layers from audio loudness.

PSD setup:
- Keep the full avatar in one PSD.
- Put the open mouth on a layer named `mouth_open`.
- Put the closed mouth on a layer named `mouth_closed`.
- Leave every other layer in the PSD visible as normal.

Example commands:

```bash
# Generate text with LM Studio, synthesize speech, and render an MP4
python talking_avatar.py --psd path/to/avatar.psd --prompt "Introduce yourself as Jim Lahey"

# Skip the LLM and speak explicit text
python talking_avatar.py --psd path/to/avatar.psd --text "Cheers, bud."

# Use an existing audio file instead of TTS
python talking_avatar.py --psd path/to/avatar.psd --audio path/to/line.wav
```

The script writes:
- `output/*.mp4` video
- matching `.txt` reply text when text was generated
- matching copied audio file next to the video

Requirements for this path:
- `ffmpeg` on `PATH`
- `Pillow`
- `psd-tools`
- the same `.env` LLM/TTS settings already used by the bot

---

## Commands

- **!vc** - Join voice channel and start listening
- **!stop** - Disconnect from voice  
- **!die** - Shutdown bot
- **@mention** - Chat with AI in text channels

---

## Troubleshooting

### Voice Connection Issues
If you get 4006 errors, make sure you installed the latest discord.py:
```bash
python -c "import discord; print('Discord.py version:', discord.__version__)"
```
Should show version `2.6.0a5254` or newer.

### TTS Issues  
If you use Inworld, verify `INWORLD_TTS_API_KEY` and the selected voice/model IDs. If you switch back to Piper, make sure the paths in `Main.py` still point to your local Piper files.

### Model Issues
If LM Studio is serving a different model, set `LM_STUDIO_MODEL` in `.env`. If left empty, the bot will use the first loaded model returned by LM Studio.

---

## What's New (August 2025)

- ✅ **Fixed Discord 4006 errors** - Updated to latest discord.py with voice fix
- 🎯 **Simplified voice connection** - More reliable voice channel joining  
- 🧠 **Enhanced AI memory** - Better conversation context
- 🔧 **Improved error handling** - Better debugging and retry logic

---

## Contributing

Feel free to open issues or submit pull requests!

---

## License

This project is licensed under the MIT License.
