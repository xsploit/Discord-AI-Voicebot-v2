# Talking Avatar Notes

This file documents the standalone `talking_avatar.py` tool in this directory. It is not part of the Discord bot itself.

## What It Does

`talking_avatar.py` reuses the local LLM and TTS configuration from this repo, then renders a PSD-based talking avatar to MP4.

PSD expectations:

- one PSD file for the full character
- a layer named `mouth_open`
- a layer named `mouth_closed`
- all other layers left visible as part of the base character

## Examples

```bash
# Use LM Studio + TTS + PSD animation
python talking_avatar.py --psd path/to/avatar.psd --prompt "Introduce yourself in one sentence."

# Skip the LLM and synthesize direct text
python talking_avatar.py --psd path/to/avatar.psd --text "Cheers, bud."

# Skip TTS and animate an existing audio file
python talking_avatar.py --psd path/to/avatar.psd --audio path/to/line.wav
```

## Useful Tuning Flags

- `--hold-frames`
- `--frequency-reactivity`
- `--attack-boost`
- `--release-boost`
- `--open-threshold`
- `--close-threshold`

## Outputs

- `output/*.mp4`
- matching copied audio next to the video
- matching `.txt` reply file when text was generated
