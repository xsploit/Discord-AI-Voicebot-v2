import argparse
import asyncio
import base64
import json
import logging
import math
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Iterable, List, Optional, Sequence, Tuple

import aiohttp
import numpy as np
import ollama
from dotenv import load_dotenv
from PIL import Image
from psd_tools import PSDImage


load_dotenv()


LLM_PROVIDER = os.getenv(
    "LLM_PROVIDER",
    "lmstudio" if os.getenv("LM_STUDIO_BASE_URL") else "ollama",
).strip().lower()
TTS_PROVIDER = os.getenv(
    "TTS_PROVIDER",
    "inworld" if os.getenv("INWORLD_TTS_API_KEY") else "piper",
).strip().lower()

DEFAULT_SYSTEM_PROMPT = (
    "You are generating dialog for a cartoon talking avatar. "
    "Keep replies concise, spoken, and natural."
)


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


class OllamaLLMClient:
    supports_streaming = False

    def __init__(self):
        self.model = os.getenv(
            "OLLAMA_MODEL",
            "hf.co/subsectmusic/qwriko3-4b-instruct-2507:Q4_K_M",
        )
        self.logger = logging.getLogger("OllamaLLM")

    def _options(self, max_tokens: Optional[int] = None) -> Dict[str, Any]:
        return {
            "num_predict": max_tokens or 2048,
            "temperature": 0.8,
            "top_k": 40,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
            "presence_penalty": 0.2,
            "frequency_penalty": 0.2,
        }

    async def generate_response(
        self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None
    ) -> Optional[str]:
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: ollama.chat(
                    model=self.model,
                    messages=messages,
                    options=self._options(max_tokens),
                ),
            )
            return response["message"]["content"]
        except Exception as exc:
            self.logger.error("Error generating response: %s", exc)
            return None

    async def stream_response(
        self, messages: List[Dict[str, str]]
    ) -> AsyncIterator[str]:
        response = await self.generate_response(messages)
        if response:
            yield response


class LMStudioLLMClient:
    supports_streaming = True

    def __init__(self):
        self.base_url = os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1").rstrip("/")
        self.model = os.getenv("LM_STUDIO_MODEL", "").strip() or None
        self.api_key = os.getenv("LM_STUDIO_API_KEY", "lm-studio")
        self.timeout = aiohttp.ClientTimeout(total=env_int("LM_STUDIO_TIMEOUT_SECONDS", 180))
        self.logger = logging.getLogger("LMStudioLLM")

    @property
    def headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _payload(
        self,
        model: str,
        messages: List[Dict[str, str]],
        stream: bool,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        return {
            "model": model,
            "messages": messages,
            "stream": stream,
            "max_tokens": max_tokens or env_int("LM_STUDIO_MAX_TOKENS", 2048),
            "temperature": float(os.getenv("LM_STUDIO_TEMPERATURE", "0.8")),
            "top_k": env_int("LM_STUDIO_TOP_K", 40),
            "top_p": float(os.getenv("LM_STUDIO_TOP_P", "0.9")),
            "repeat_penalty": float(os.getenv("LM_STUDIO_REPEAT_PENALTY", "1.1")),
            "presence_penalty": float(os.getenv("LM_STUDIO_PRESENCE_PENALTY", "0.2")),
            "frequency_penalty": float(os.getenv("LM_STUDIO_FREQUENCY_PENALTY", "0.2")),
        }

    async def _resolve_model(self) -> str:
        if self.model:
            return self.model

        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            async with session.get(f"{self.base_url}/models", headers=self.headers) as response:
                response.raise_for_status()
                payload = await response.json()

        models = payload.get("data") or []
        if not models:
            raise RuntimeError("LM Studio returned no loaded models from /v1/models")

        self.model = models[0]["id"]
        self.logger.info("Using LM Studio model: %s", self.model)
        return self.model

    async def generate_response(
        self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None
    ) -> Optional[str]:
        try:
            model = await self._resolve_model()
            payload = self._payload(model, messages, stream=False, max_tokens=max_tokens)
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as exc:
            self.logger.error("LM Studio request failed: %s", exc)
            return None


def build_llm_client():
    if LLM_PROVIDER == "lmstudio":
        return LMStudioLLMClient()
    if LLM_PROVIDER == "ollama":
        return OllamaLLMClient()
    raise ValueError(f"Unsupported LLM provider: {LLM_PROVIDER}")


class InworldTTS:
    supports_streaming_input = True

    def __init__(
        self,
        api_key: Optional[str] = None,
        voice_id: str = "Dennis",
        model_id: str = "inworld-tts-1.5-max",
        endpoint: str = "wss://api.inworld.ai/tts/v1/voice:streamBidirectional",
        output_dir: str = "output",
    ):
        self.api_key = api_key or os.getenv("INWORLD_TTS_API_KEY")
        self.voice_id = os.getenv("INWORLD_TTS_VOICE_ID", voice_id)
        self.model_id = os.getenv("INWORLD_TTS_MODEL_ID", model_id)
        self.endpoint = os.getenv("INWORLD_TTS_ENDPOINT", endpoint)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.sample_rate_hz = env_int("INWORLD_TTS_SAMPLE_RATE_HZ", 48000)
        self.buffer_char_threshold = env_int("INWORLD_TTS_BUFFER_CHAR_THRESHOLD", 40)
        self.auto_mode = env_bool("INWORLD_TTS_AUTO_MODE", True)
        self.timeout_seconds = env_int("INWORLD_TTS_TIMEOUT_SECONDS", 90)
        self.timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        self.initialized = False
        self.logger = logging.getLogger("InworldTTS")

    async def initialize(self) -> bool:
        if self.initialized:
            return True
        if not self.api_key:
            self.logger.error("Missing INWORLD_TTS_API_KEY")
            return False
        self.initialized = True
        return True

    @property
    def headers(self) -> Dict[str, str]:
        return {"Authorization": f"Basic {self.api_key}"}

    def _decode_audio_chunk(self, audio_content: str) -> bytes:
        if not audio_content:
            return b""
        chunk = base64.b64decode(audio_content)
        if len(chunk) <= 44:
            return b""
        if chunk.startswith(b"RIFF"):
            return chunk[44:]
        return chunk

    async def _iter_chunks(self, chunks: Iterable[str]) -> AsyncIterator[str]:
        if hasattr(chunks, "__aiter__"):
            async for chunk in chunks:
                yield chunk
            return
        for chunk in chunks:
            yield chunk

    async def stream_text_chunks(self, chunks: Iterable[str]) -> bytes:
        if not await self.initialize():
            return b""

        context_id = f"ctx-{uuid.uuid4().hex[:8]}"
        pending_flushes = 0
        completed_flushes = 0
        raw_audio = bytearray()
        receiver_error = None
        context_ready = asyncio.Event()

        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            async with session.ws_connect(self.endpoint, headers=self.headers, heartbeat=30) as ws:
                async def receiver():
                    nonlocal completed_flushes, receiver_error
                    try:
                        while True:
                            message = await ws.receive()
                            if message.type == aiohttp.WSMsgType.TEXT:
                                payload = json.loads(message.data)
                                result = payload.get("result") or {}
                                status = result.get("status") or {}
                                if status.get("code", 0) not in (0, None):
                                    raise RuntimeError(f"Inworld TTS error: {status}")
                                if "contextCreated" in result:
                                    context_ready.set()
                                if "audioChunk" in result:
                                    pcm_chunk = self._decode_audio_chunk(
                                        result["audioChunk"].get("audioContent", "")
                                    )
                                    if pcm_chunk:
                                        raw_audio.extend(pcm_chunk)
                                if "flushCompleted" in result:
                                    completed_flushes += 1
                                if "contextClosed" in result:
                                    break
                            elif message.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED):
                                break
                            elif message.type == aiohttp.WSMsgType.ERROR:
                                raise RuntimeError("Inworld websocket returned an error frame")
                    except Exception as exc:
                        receiver_error = exc
                        context_ready.set()

                receiver_task = asyncio.create_task(receiver())

                await ws.send_json(
                    {
                        "create": {
                            "voiceId": self.voice_id,
                            "modelId": self.model_id,
                            "audioConfig": {
                                "audioEncoding": "LINEAR16",
                                "sampleRateHertz": self.sample_rate_hz,
                            },
                            "bufferCharThreshold": self.buffer_char_threshold,
                            "autoMode": self.auto_mode,
                            "timestampType": "WORD",
                            "timestampTransportStrategy": "ASYNC",
                        },
                        "contextId": context_id,
                    }
                )

                await asyncio.wait_for(context_ready.wait(), timeout=10)
                if receiver_error:
                    raise receiver_error

                async for chunk in self._iter_chunks(chunks):
                    chunk = (chunk or "").strip()
                    if not chunk:
                        continue
                    pending_flushes += 1
                    await ws.send_json(
                        {
                            "send_text": {"text": chunk, "flush_context": {}},
                            "contextId": context_id,
                        }
                    )

                while completed_flushes < pending_flushes:
                    if receiver_error:
                        raise receiver_error
                    await asyncio.sleep(0.02)

                await ws.send_json({"close_context": {}, "contextId": context_id})

                try:
                    await asyncio.wait_for(receiver_task, timeout=5)
                except asyncio.TimeoutError:
                    receiver_task.cancel()

                if receiver_error:
                    raise receiver_error

        return bytes(raw_audio)

    async def generate_speech(self, text: str) -> Optional[str]:
        if not text:
            return None

        async def one_chunk():
            yield text

        try:
            pcm_audio = await self.stream_text_chunks(one_chunk())
            if not pcm_audio:
                return None
            output_file = self.output_dir / f"output_{int(time.time() * 1000)}.wav"
            import wave

            with wave.open(str(output_file), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.sample_rate_hz)
                wav_file.writeframes(pcm_audio)
            return str(output_file)
        except Exception as exc:
            self.logger.error("Error generating Inworld speech: %s", exc)
            return None


class PiperTTS:
    supports_streaming_input = False

    def __init__(
        self,
        piper_path: str = "piper/piper.exe",
        model_path: str = "piper/en_US-ryari-high.onnx",
        model_config: str = "piper/en_US-ryari-high.onnx.json",
        output_dir: str = "output",
        timeout: int = 10,
    ):
        self.piper_path = Path(piper_path)
        self.model_path = Path(model_path)
        self.model_config = Path(model_config)
        self.output_dir = Path(output_dir)
        self.timeout = timeout
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger("PiperTTS")

    async def verify_files(self) -> bool:
        required_files = [
            (self.piper_path, "Piper executable"),
            (self.model_path, "Model file"),
            (self.model_config, "Model configuration"),
        ]
        for file_path, description in required_files:
            if not file_path.exists():
                self.logger.error("Missing %s at %s", description, file_path)
                return False
        return True

    async def generate_speech(self, text: str) -> Optional[str]:
        if not await self.verify_files():
            return None
        output_file = str(self.output_dir / f"output_{int(time.time() * 1000)}.wav")
        input_json = {
            "text": text,
            "output_file": output_file,
            "length_scale": 1.0,
            "noise_scale": 0.667,
            "noise_w": 0.8,
        }
        process = await asyncio.create_subprocess_exec(
            str(self.piper_path),
            "-m",
            str(self.model_path),
            "-c",
            str(self.model_config),
            "--json-input",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, stderr = await asyncio.wait_for(
                process.communicate(json.dumps(input_json).encode() + b"\n"),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            process.kill()
            raise RuntimeError("Piper process timed out") from None

        if process.returncode != 0:
            raise RuntimeError(stderr.decode(errors="ignore").strip() or "Piper process failed")
        if not os.path.exists(output_file):
            raise RuntimeError("Piper did not create the output audio file")
        return output_file


def build_tts_provider():
    if TTS_PROVIDER == "inworld":
        return InworldTTS()
    if TTS_PROVIDER == "piper":
        return PiperTTS()
    raise ValueError(f"Unsupported TTS provider: {TTS_PROVIDER}")


def iter_layers(group) -> Iterable[Any]:
    for layer in group:
        yield layer
        if getattr(layer, "is_group", lambda: False)():
            yield from iter_layers(layer)


def find_layer_by_name(psd: PSDImage, layer_name: str):
    wanted = layer_name.strip().lower()
    for layer in iter_layers(psd):
        name = (layer.name or "").strip().lower()
        if name == wanted:
            return layer
    raise ValueError(f"Could not find PSD layer named '{layer_name}'")


def render_state_images(
    psd_path: Path, open_layer_name: str, closed_layer_name: str
) -> Tuple[Image.Image, Image.Image]:
    psd = PSDImage.open(str(psd_path))
    open_layer = find_layer_by_name(psd, open_layer_name)
    closed_layer = find_layer_by_name(psd, closed_layer_name)

    open_layer.visible = False
    closed_layer.visible = True
    closed_image = psd.composite(force=True)

    open_layer.visible = True
    closed_layer.visible = False
    open_image = psd.composite(force=True)

    if closed_image is None or open_image is None:
        raise RuntimeError("PSD composition failed")

    return open_image.convert("RGBA"), closed_image.convert("RGBA")


def decode_audio_samples(audio_path: Path, sample_rate: int = 48000) -> Tuple[np.ndarray, int]:
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(audio_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "pipe:1",
    ]
    result = subprocess.run(command, capture_output=True)
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(f"ffmpeg audio decode failed: {stderr}")

    samples = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32)
    if samples.size == 0:
        raise RuntimeError("Decoded audio is empty")
    return samples / 32768.0, sample_rate


def normalize_feature(values: np.ndarray, low_q: float = 0.10, high_q: float = 0.95) -> np.ndarray:
    if values.size == 0:
        return values

    low = float(np.quantile(values, low_q))
    high = float(np.quantile(values, high_q))
    spread = max(high - low, 1e-6)
    normalized = (values - low) / spread
    return np.clip(normalized, 0.0, 1.0)


def compute_mouth_states(
    samples: np.ndarray,
    sample_rate: int,
    fps: int,
    open_threshold: Optional[float] = None,
    close_threshold: Optional[float] = None,
    hold_frames: int = 1,
    frequency_reactivity: float = 0.25,
    attack_boost: float = 0.35,
    release_boost: float = 0.25,
) -> Tuple[List[bool], Dict[str, float]]:
    samples_per_frame = max(1, int(round(sample_rate / fps)))
    frame_count = int(math.ceil(len(samples) / samples_per_frame))
    energy = np.zeros(frame_count, dtype=np.float32)
    centroid = np.zeros(frame_count, dtype=np.float32)
    presence = np.zeros(frame_count, dtype=np.float32)

    fft_freqs = np.fft.rfftfreq(samples_per_frame, d=1.0 / sample_rate)
    speech_band = (fft_freqs >= 250.0) & (fft_freqs <= 2500.0)
    high_band = (fft_freqs >= 1500.0) & (fft_freqs <= 5000.0)
    window = np.hanning(samples_per_frame).astype(np.float32)

    for index in range(frame_count):
        start = index * samples_per_frame
        end = min(len(samples), start + samples_per_frame)
        chunk = samples[start:end]
        if not chunk.size:
            continue

        energy[index] = float(np.sqrt(np.mean(chunk * chunk)))
        padded = np.zeros(samples_per_frame, dtype=np.float32)
        padded[: chunk.size] = chunk
        spectrum = np.abs(np.fft.rfft(padded * window))
        spectrum_sum = float(np.sum(spectrum))
        if spectrum_sum <= 1e-8:
            continue

        centroid[index] = float(np.sum(fft_freqs * spectrum) / spectrum_sum)
        speech_energy = float(np.sum(spectrum[speech_band]))
        high_energy = float(np.sum(spectrum[high_band]))
        presence[index] = high_energy / max(speech_energy + high_energy, 1e-6)

    if frame_count >= 3:
        kernel = np.array([0.15, 0.70, 0.15], dtype=np.float32)
        energy = np.convolve(energy, kernel, mode="same")
        centroid = np.convolve(centroid, kernel, mode="same")
        presence = np.convolve(presence, kernel, mode="same")

    active_energy = energy[energy > 1e-4]
    if active_energy.size:
        floor = float(np.quantile(active_energy, 0.15))
        ceiling = float(np.quantile(active_energy, 0.95))
    else:
        floor = 0.0
        ceiling = max(float(energy.max()), 0.02)

    dynamic_range = max(ceiling - floor, 0.02)
    energy_score = np.clip((energy - floor) / dynamic_range, 0.0, 1.0)
    centroid_score = normalize_feature(centroid)
    presence_score = normalize_feature(presence)

    frequency_reactivity = float(min(max(frequency_reactivity, 0.0), 1.0))
    attack_boost = float(min(max(attack_boost, 0.0), 1.0))
    release_boost = float(min(max(release_boost, 0.0), 1.0))
    volume_weight = 1.0 - frequency_reactivity
    frequency_score = (0.65 * centroid_score) + (0.35 * presence_score)
    raw_activity = (volume_weight * energy_score) + (frequency_reactivity * frequency_score)
    delta = np.diff(raw_activity, prepend=raw_activity[0])
    onset_score = normalize_feature(np.clip(delta, 0.0, None), low_q=0.05, high_q=0.95)
    decay_score = normalize_feature(np.clip(-delta, 0.0, None), low_q=0.05, high_q=0.95)
    activity = raw_activity.copy()

    if frame_count >= 5:
        kernel = np.array([0.08, 0.22, 0.40, 0.22, 0.08], dtype=np.float32)
        activity = np.convolve(activity, kernel, mode="same")

    open_signal = np.clip(activity + (attack_boost * onset_score), 0.0, 1.5)
    close_signal = np.clip(activity - (release_boost * decay_score), 0.0, 1.5)

    active_activity = activity[energy > max(floor, 1e-4)]
    if active_activity.size:
        score_floor = float(np.quantile(active_activity, 0.12))
        score_ceiling = float(np.quantile(active_activity, 0.92))
    else:
        score_floor = 0.0
        score_ceiling = max(float(activity.max()), 0.2)

    score_range = max(score_ceiling - score_floor, 0.10)
    if open_threshold is None:
        open_threshold = score_floor + (score_range * 0.24)
    if close_threshold is None:
        close_threshold = score_floor + (score_range * 0.12)

    open_threshold = float(max(0.05, min(open_threshold, 0.95)))
    close_threshold = float(max(0.03, min(close_threshold, open_threshold * 0.95)))

    states: List[bool] = []
    mouth_open = False
    hold_counter = 0

    for open_level, close_level in zip(open_signal, close_signal):
        if mouth_open:
            if close_level >= close_threshold:
                hold_counter = hold_frames
            elif hold_counter > 0:
                hold_counter -= 1
            else:
                mouth_open = False
        else:
            if open_level >= open_threshold:
                mouth_open = True
                hold_counter = hold_frames
        states.append(mouth_open)

    return states, {
        "open_threshold": open_threshold,
        "close_threshold": close_threshold,
        "max_energy": float(energy.max()),
        "max_activity": float(activity.max()),
        "frequency_reactivity": frequency_reactivity,
        "attack_boost": attack_boost,
        "release_boost": release_boost,
        "duration_seconds": len(samples) / float(sample_rate),
    }


def write_video(
    open_image: Image.Image,
    closed_image: Image.Image,
    states: Sequence[bool],
    audio_path: Path,
    output_path: Path,
    fps: int,
) -> None:
    width, height = open_image.size
    if closed_image.size != (width, height):
        raise ValueError("Open and closed images must have the same size")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    open_bytes = open_image.tobytes()
    closed_bytes = closed_image.tobytes()

    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgba",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-i",
        str(audio_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "libx264",
        "-preset",
        os.getenv("AVATAR_FFMPEG_PRESET", "medium"),
        "-crf",
        os.getenv("AVATAR_FFMPEG_CRF", "18"),
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        "-shortest",
        str(output_path),
    ]

    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    try:
        for state in states:
            process.stdin.write(open_bytes if state else closed_bytes)
        process.stdin.close()
    except Exception:
        process.kill()
        raise

    stderr = process.stderr.read().decode("utf-8", errors="ignore").strip()
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg video encode failed: {stderr}")


async def resolve_text(args: argparse.Namespace) -> Optional[str]:
    if args.text:
        return args.text.strip()
    if not args.prompt:
        return None

    client = build_llm_client()
    messages = []
    if args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})
    messages.append({"role": "user", "content": args.prompt})

    response = await client.generate_response(messages, max_tokens=args.max_tokens)
    if not response:
        raise RuntimeError("LLM did not return any text")
    return response.strip()


async def resolve_audio(args: argparse.Namespace, text: Optional[str]) -> Path:
    if args.audio:
        audio_path = Path(args.audio)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        return audio_path

    if not text:
        raise ValueError("Need either --audio, --text, or --prompt")

    tts = build_tts_provider()
    audio_file = await tts.generate_speech(text)
    if not audio_file:
        raise RuntimeError("TTS did not produce audio")
    return Path(audio_file)


def default_output_path() -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return Path("output") / f"talking_avatar_{stamp}.mp4"


async def main_async(args: argparse.Namespace) -> None:
    output_path = Path(args.output) if args.output else default_output_path()
    output_path = output_path.resolve()

    text = await resolve_text(args)
    audio_path = await resolve_audio(args, text)
    open_image, closed_image = render_state_images(
        Path(args.psd),
        args.open_layer,
        args.closed_layer,
    )

    samples, sample_rate = decode_audio_samples(audio_path, sample_rate=args.sample_rate)
    states, metrics = compute_mouth_states(
        samples,
        sample_rate,
        fps=args.fps,
        open_threshold=args.open_threshold,
        close_threshold=args.close_threshold,
        hold_frames=args.hold_frames,
        frequency_reactivity=args.frequency_reactivity,
        attack_boost=args.attack_boost,
        release_boost=args.release_boost,
    )
    write_video(open_image, closed_image, states, audio_path, output_path, fps=args.fps)

    if text:
        text_output = output_path.with_suffix(".txt")
        text_output.write_text(text + "\n", encoding="utf-8")

    copied_audio_path = output_path.with_suffix(audio_path.suffix or ".wav")
    if audio_path.resolve() != copied_audio_path:
        shutil.copy2(audio_path, copied_audio_path)

    logging.info(
        "Rendered %s frames at %s fps. open_threshold=%.4f close_threshold=%.4f freq_mix=%.2f attack=%.2f release=%.2f duration=%.2fs",
        len(states),
        args.fps,
        metrics["open_threshold"],
        metrics["close_threshold"],
        metrics["frequency_reactivity"],
        metrics["attack_boost"],
        metrics["release_boost"],
        metrics["duration_seconds"],
    )
    print(f"Video: {output_path}")
    if text:
        print(f"Text:  {output_path.with_suffix('.txt')}")
    print(f"Audio: {copied_audio_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a simple PSD-based talking avatar video with open/closed mouth switching."
    )
    parser.add_argument("--psd", required=True, help="Path to the layered PSD file.")
    parser.add_argument(
        "--open-layer",
        default="mouth_open",
        help="Layer name for the open mouth state.",
    )
    parser.add_argument(
        "--closed-layer",
        default="mouth_closed",
        help="Layer name for the closed mouth state.",
    )
    parser.add_argument("--prompt", help="Prompt to send to the LLM.")
    parser.add_argument("--text", help="Direct text to synthesize without using the LLM.")
    parser.add_argument("--audio", help="Existing audio file to animate instead of synthesizing TTS.")
    parser.add_argument("--output", help="Output video path. Defaults to output/talking_avatar_TIMESTAMP.mp4.")
    parser.add_argument("--fps", type=int, default=24, help="Video frame rate.")
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=48000,
        help="Sample rate for audio analysis.",
    )
    parser.add_argument(
        "--hold-frames",
        type=int,
        default=1,
        help="How many extra frames to keep the mouth open after a loud frame.",
    )
    parser.add_argument(
        "--open-threshold",
        type=float,
        default=None,
        help="Manual activity threshold for opening the mouth.",
    )
    parser.add_argument(
        "--close-threshold",
        type=float,
        default=None,
        help="Manual activity threshold for closing the mouth.",
    )
    parser.add_argument(
        "--frequency-reactivity",
        type=float,
        default=0.25,
        help="Blend amount for frequency-based mouth activity, from 0.0 to 1.0.",
    )
    parser.add_argument(
        "--attack-boost",
        type=float,
        default=0.35,
        help="How strongly rising speech energy snaps the mouth open, from 0.0 to 1.0.",
    )
    parser.add_argument(
        "--release-boost",
        type=float,
        default=0.25,
        help="How strongly falling speech energy helps the mouth close, from 0.0 to 1.0.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="LLM max tokens when using --prompt.",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt to use with the LLM.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
