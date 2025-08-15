# Discord AI Voicebot - Hikari-chan

**UPDATE (August 2025)**: Fixed Discord 4006 voice connection errors with latest discord.py version!

This is a **Discord bot** that combines **AI chat, voice recognition, and TTS** into one intelligent assistant. Meet **Hikari-chan** - an AI with Hinata's personality that can chat via text or voice, remember conversations, and respond with natural speech.

## Features:
- 🤖 **AI-Powered Chat**: Uses Ollama with your custom models for intelligent responses
- 🎙️ **Voice Recognition**: Listens to voice chat and responds naturally  
- 🔊 **Text-to-Speech**: Generates natural speech using PiperTTS
- 🧠 **Memory System**: Remembers past conversations for context
- 💬 **Text & Voice**: Works in both text channels and voice channels
- 🎯 **Smart Responses**: Only responds when mentioned or in voice chat

---

## Prerequisites

- **Python 3.8+** 
- **Ollama** running your preferred model
- **Discord Bot Token** from [Discord Developer Portal](https://discord.com/developers/applications)
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

### 4. Configure Bot Token
Edit `Main.py` and replace the token:
```python
DISCORD_BOT_TOKEN = 'YOUR_ACTUAL_BOT_TOKEN_HERE'
```

### 5. Set Up Ollama Model
Make sure Ollama is running with your model:
```bash
ollama run hf.co/subsectmusic/qwriko3-4b-instruct-2507:Q4_K_M
```

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

### Piper TTS Issues  
The bot includes Piper TTS files. Make sure the paths in `Main.py` point to your actual Piper installation.

### Model Issues
Edit the model name in `Main.py` to match your Ollama model:
```python
model='your-model-name-here'
```

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