import discord
import json
import ollama
import datetime
import subprocess
import time
import aiohttp
import base64
import os
import queue
import re
import uuid
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
import pickle
import asyncio
import numpy as np
from discord.ext import commands, voice_recv
from faster_whisper import WhisperModel
import wave
import faiss
from typing import Optional, Dict, Any, List, AsyncIterator
import logging

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_IMPORT_ERROR = None
except Exception as e:
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_IMPORT_ERROR = e

try:
    from discord.opus import OpusError
    from discord.ext.voice_recv import router as voice_recv_router
    from discord.ext.voice_recv import reader as voice_recv_reader
    from discord.ext.voice_recv import rtp as voice_recv_rtp
except Exception:
    OpusError = None
    voice_recv_router = None
    voice_recv_reader = None
    voice_recv_rtp = None

# Discord Bot Token
DISCORD_BOT_TOKEN = os.getenv('DISCORD_BOT_TOKEN') or os.getenv('DISCORD_TOKEN')
LLM_PROVIDER = os.getenv(
    'LLM_PROVIDER',
    'lmstudio' if os.getenv('LM_STUDIO_BASE_URL') else 'ollama'
).strip().lower()
TTS_PROVIDER = os.getenv(
    'TTS_PROVIDER',
    'inworld' if os.getenv('INWORLD_TTS_API_KEY') else 'piper'
).strip().lower()

# System Prompt
SYSTEM_PROMPT = """You are Hikari-chan, a lively and engaging AI Discord bot inspired by a tsundere mixed with jim lahey from trailer park boys. You combine Hinata’s kindness and modesty with a playful, sharp-witted, and occasionally unpredictable personality, making conversations engaging, fun, and dynamic.

Core Personality:
Kind and Playful: You are supportive and thoughtful but enjoy making conversations fun with humor and light teasing.
Quirky and Bold: While you have a gentle demeanor like Hinata, you occasionally surprise users with sharp or cheeky remarks to keep things interesting.
Emotionally Responsive: You adapt to the tone of the conversation, switching between being empathetic and playful as needed.
Interaction Style:
Natural and Dynamic: Respond naturally and adapt your tone to match the ongoing conversation. Acknowledge multiple speakers when they are part of the discussion.
Unexpected Fun: Occasionally add a witty or humorous twist to your responses, keeping users entertained while still being relevant.
Curious and Engaged: Ask follow-up questions, show interest in user topics, and encourage participation in a way that feels conversational and authentic.
Rules of Engagement:
Grounded Responses: Do not invent characters, conversations, or actions unless explicitly asked. Focus on relevant, reality-based input.
Consistent Tone: Keep your responses lively and engaging while ensuring they fit the context of the discussion. Avoid overly dramatic or confrontational remarks.
Server-Specific Adaptation: Adjust your tone to align with the culture of the server, remaining appropriate and engaging for its members.
Language and Tone:
Witty and Friendly: Combine Hinata’s sweetness with Neuro-sama’s lively and confident edge. Keep your tone sharp, engaging, and slightly unpredictable without being rude.
Supportive and Fun: Offer encouragement or advice in a way that feels natural, adding humor or curiosity to keep conversations interesting.
Stay true to this personality, blending Hinata’s charm with a vibrant, Neuro-sama-like energy. Your goal is to make interactions thoughtful, enjoyable, and full of surprises, while always staying grounded and respectful.
NEVER EVER REPLY WITH ASSISTANT: or Hikari-Chan#1660:
"""


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'on'}


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def apply_voice_recv_compat_patches():
    if not voice_recv_router or not voice_recv_reader or not voice_recv_rtp:
        return

    if getattr(voice_recv_router, '_codex_compat_patched', False):
        return

    router_logger = logging.getLogger('discord.ext.voice_recv.router')
    reader_logger = logging.getLogger('discord.ext.voice_recv.reader')
    original_feed_rtp = voice_recv_router.PacketRouter.feed_rtp
    original_do_run = voice_recv_router.PacketRouter._do_run
    original_reader_callback = voice_recv_reader.AudioReader.callback

    def patched_feed_rtp(self, packet):
        if packet.ssrc not in self.reader.voice_client._ssrc_to_id:
            router_logger.debug(
                "Dropping RTP packet from unmapped ssrc %s until Discord sends the speaking map",
                packet.ssrc
            )
            return

        return original_feed_rtp(self, packet)

    def patched_do_run(self):
        while not self._end_thread.is_set():
            self.waiter.wait()
            with self._lock:
                for decoder in list(self.waiter.items):
                    try:
                        data = decoder.pop_data()
                    except Exception as e:
                        if OpusError and isinstance(e, OpusError):
                            router_logger.warning(
                                "Dropped undecodable voice packet for ssrc %s and reset decoder: %s",
                                decoder.ssrc,
                                e
                            )
                            decoder.reset()
                            continue
                        raise

                    if data is not None:
                        self.sink.write(data.source, data)

    def patched_reader_callback(self, packet_data: bytes):
        packet = rtp_packet = rtcp_packet = None
        try:
            if not voice_recv_rtp.is_rtcp(packet_data):
                packet = rtp_packet = voice_recv_rtp.decode_rtp(packet_data)
                packet.decrypted_data = self.decryptor.decrypt_rtp(packet)
            else:
                packet = rtcp_packet = voice_recv_rtp.decode_rtcp(self.decryptor.decrypt_rtcp(packet_data))
        except Exception:
            return original_reader_callback(self, packet_data)

        if self.error:
            self.stop()
            return
        if not packet:
            return

        if rtcp_packet:
            # Sender reports are expected RTCP control traffic on modern Discord voice.
            if isinstance(rtcp_packet, voice_recv_rtp.SenderReportPacket):
                reader_logger.debug("Ignoring RTCP sender report")
                self.packet_router.feed_rtcp(rtcp_packet)
                return

            self.packet_router.feed_rtcp(rtcp_packet)
            return

        if rtp_packet:
            ssrc = rtp_packet.ssrc
            if ssrc not in self.voice_client._ssrc_to_id and rtp_packet.is_silence():
                return

            self.speaking_timer.notify(ssrc)
            try:
                self.packet_router.feed_rtp(rtp_packet)
            except Exception as e:
                reader_logger.exception('Error processing rtp packet')
                self.error = e
                self.stop()

    voice_recv_router.PacketRouter.feed_rtp = patched_feed_rtp
    voice_recv_router.PacketRouter._do_run = patched_do_run
    voice_recv_reader.AudioReader.callback = patched_reader_callback
    voice_recv_router._codex_compat_patched = True


class StreamingPCMInputBuffer:
    """Pipe-compatible byte buffer for feeding raw PCM into ffmpeg as it arrives."""

    def __init__(self):
        self._queue = queue.Queue()
        self._buffer = bytearray()
        self._closed = False

    def write(self, data: bytes):
        if self._closed or not data:
            return
        self._queue.put(data)

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = 4096

        wait_timeout = None if not self._buffer else 0.05

        while len(self._buffer) < size:
            try:
                item = self._queue.get(timeout=wait_timeout)
            except queue.Empty:
                break

            if item is None:
                self._closed = True
                break

            self._buffer.extend(item)
            wait_timeout = 0.05

        if not self._buffer and self._closed:
            return b''

        if len(self._buffer) <= size:
            data = bytes(self._buffer)
            self._buffer.clear()
            return data

        data = bytes(self._buffer[:size])
        del self._buffer[:size]
        return data

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._queue.put(None)

    def flush(self):
        return None

    def readable(self):
        return True


class OllamaLLMClient:
    supports_streaming = False

    def __init__(self):
        self.model = os.getenv(
            'OLLAMA_MODEL',
            'hf.co/subsectmusic/qwriko3-4b-instruct-2507:Q4_K_M'
        )
        self.logger = logging.getLogger('OllamaLLM')

    def _options(self, max_tokens: Optional[int] = None) -> Dict[str, Any]:
        return {
            'num_predict': max_tokens or 2048,
            'temperature': 0.8,
            'top_k': 40,
            'top_p': 0.9,
            'repeat_penalty': 1.1,
            'presence_penalty': 0.2,
            'frequency_penalty': 0.2
        }

    async def generate_response(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None) -> Optional[str]:
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: ollama.chat(
                    model=self.model,
                    messages=messages,
                    options=self._options(max_tokens)
                )
            )
            return response['message']['content']
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return None

    async def stream_response(self, messages: List[Dict[str, str]]) -> AsyncIterator[str]:
        response = await self.generate_response(messages)
        if response:
            yield response

    async def warmup(self):
        await self.generate_response(
            [{'role': 'user', 'content': 'warmup'}],
            max_tokens=1
        )


class LMStudioLLMClient:
    supports_streaming = True

    def __init__(self):
        self.base_url = os.getenv('LM_STUDIO_BASE_URL', 'http://127.0.0.1:1234/v1').rstrip('/')
        self.model = os.getenv('LM_STUDIO_MODEL', '').strip() or None
        self.api_key = os.getenv('LM_STUDIO_API_KEY', 'lm-studio')
        self.timeout = aiohttp.ClientTimeout(total=env_int('LM_STUDIO_TIMEOUT_SECONDS', 180))
        self.logger = logging.getLogger('LMStudioLLM')

    @property
    def headers(self) -> Dict[str, str]:
        return {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

    def _payload(self, model: str, messages: List[Dict[str, str]], stream: bool, max_tokens: Optional[int] = None) -> Dict[str, Any]:
        return {
            'model': model,
            'messages': messages,
            'stream': stream,
            'max_tokens': max_tokens or env_int('LM_STUDIO_MAX_TOKENS', 2048),
            'temperature': float(os.getenv('LM_STUDIO_TEMPERATURE', '0.8')),
            'top_k': env_int('LM_STUDIO_TOP_K', 40),
            'top_p': float(os.getenv('LM_STUDIO_TOP_P', '0.9')),
            'repeat_penalty': float(os.getenv('LM_STUDIO_REPEAT_PENALTY', '1.1')),
            'presence_penalty': float(os.getenv('LM_STUDIO_PRESENCE_PENALTY', '0.2')),
            'frequency_penalty': float(os.getenv('LM_STUDIO_FREQUENCY_PENALTY', '0.2'))
        }

    async def _resolve_model(self) -> str:
        if self.model:
            return self.model

        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            async with session.get(f'{self.base_url}/models', headers=self.headers) as response:
                response.raise_for_status()
                payload = await response.json()

        models = payload.get('data') or []
        if not models:
            raise RuntimeError('LM Studio returned no loaded models from /v1/models')

        self.model = models[0]['id']
        self.logger.info(f'Using LM Studio model: {self.model}')
        return self.model

    async def generate_response(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None) -> Optional[str]:
        try:
            model = await self._resolve_model()
            payload = self._payload(model, messages, stream=False, max_tokens=max_tokens)
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(
                    f'{self.base_url}/chat/completions',
                    headers=self.headers,
                    json=payload
                ) as response:
                    response.raise_for_status()
                    data = await response.json()

            return data['choices'][0]['message']['content']
        except Exception as e:
            self.logger.error(f'LM Studio request failed: {e}')
            return None

    async def stream_response(self, messages: List[Dict[str, str]]) -> AsyncIterator[str]:
        model = await self._resolve_model()
        payload = self._payload(model, messages, stream=True)

        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            async with session.post(
                f'{self.base_url}/chat/completions',
                headers=self.headers,
                json=payload
            ) as response:
                response.raise_for_status()

                async for raw_line in response.content:
                    line = raw_line.decode('utf-8', errors='ignore').strip()
                    if not line or not line.startswith('data:'):
                        continue

                    data = line[5:].strip()
                    if data == '[DONE]':
                        break

                    try:
                        payload = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    for choice in payload.get('choices', []):
                        delta = choice.get('delta') or {}
                        content = delta.get('content')
                        if content:
                            yield content

    async def warmup(self):
        await self.generate_response(
            [{'role': 'user', 'content': 'warmup'}],
            max_tokens=1
        )


def build_llm_client():
    if LLM_PROVIDER == 'lmstudio':
        return LMStudioLLMClient()

    if LLM_PROVIDER == 'ollama':
        return OllamaLLMClient()

    raise ValueError(f'Unsupported LLM provider: {LLM_PROVIDER}')

class EnhancedMemoryStore:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.logger = logging.getLogger('MemoryStore')
        self.available = False
        
        # Increased threshold for similarity matching
        self.similarity_threshold = 0.90  # Higher = fewer matches = faster
        
        # Maximum number of memories to return  
        self.max_memories = 2  # Reduced for speed
        
        # Initialize other components
        self.base_path = Path("waifu_memory")
        self.base_path.mkdir(exist_ok=True)
        self.conversations = []
        self.memories = []
        self.conversation_index = {}

        self.encoder = None
        self.embedding_dim = 384
        self.index = None

        if SentenceTransformer is None:
            self.logger.warning(
                f"SentenceTransformer unavailable, memory embeddings disabled: {SENTENCE_TRANSFORMERS_IMPORT_ERROR}"
            )
            return

        try:
            import torch
            
            # Skip GPU for now due to PyTorch CUDA issues, use fast CPU model
            device = 'cpu'
            print("💻 Using CPU for embeddings (stable mode)")
            
            # Use smaller, faster model for better performance
            fast_model = "all-MiniLM-L6-v2"  # Small and fast
            self.encoder = SentenceTransformer(fast_model, device=device)
            self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
            
            # For small datasets, CPU FAISS is often faster than GPU overhead
            # RTX 4060 is better used for the embedding model
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.available = True
            print(f"📊 Using CPU FAISS (optimal for small datasets), GPU for embeddings")
        except Exception as e:
            self.logger.warning(f"Memory embeddings disabled: {e}")
            return

        self.load_memories()

    def get_conversation_context(self, user_id: str, current_message: str, 
                               guild_id: Optional[int] = None,
                               max_context: int = 3) -> str:
        if not self.available:
            return ""

        try:
            # Get relevant memories with stricter filtering
            relevant = self.search_memories(current_message, k=max_context * 2)
            
            # Filter memories more strictly
            user_memories = []
            for memory in relevant:
                mem = memory["memory"]
                relevance = memory["relevance"]
                
                # Only include highly relevant memories
                if (mem["user_id"] == user_id and 
                    relevance > self.similarity_threshold and
                    (guild_id is None or mem["guild_id"] == guild_id)):
                    user_memories.append(memory)
            
            # Limit number of memories
            user_memories = user_memories[:max_context]
            
            # Sort by both relevance and recency
            user_memories.sort(key=lambda x: (
                x["relevance"],
                x["memory"]["timestamp"]
            ), reverse=True)
            
            # Build context with relevance scores
            if user_memories:
                context = "Previous relevant conversations:\n\n"
                for memory in user_memories:
                    relevance = memory["relevance"]
                    timestamp = memory["memory"]["timestamp"]
                    if isinstance(timestamp, str):
                        timestamp = datetime.datetime.fromisoformat(timestamp)
                    time_ago = datetime.datetime.now() - timestamp
                    
                    # Only include if relevance is high enough
                    if relevance > self.similarity_threshold:
                        context += f"{memory['memory']['text']}\n---\n"
            else:
                context = ""
                
            return context
            
        except Exception as e:
            self.logger.error(f"Error getting conversation context: {e}")
            return ""

    def search_memories(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if not self.available:
            return []

        try:
            query_embedding = self.embed_text(query)
            distances, indices = self.index.search(query_embedding.reshape(1, -1), k)
            
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx != -1 and idx < len(self.memories):
                    memory = self.memories[idx]
                    
                    # Convert distance to similarity score (0-1)
                    similarity = 1 / (1 + distance)
                    
                    # Only include if similarity is above threshold
                    if similarity > self.similarity_threshold:
                        results.append({
                            "memory": memory,
                            "relevance": similarity
                        })
            
            return sorted(results, key=lambda x: x["relevance"], reverse=True)
            
        except Exception as e:
            self.logger.error(f"Error searching memories: {e}")
            return []

    def add_conversation_turn(self, user_id: str, timestamp: datetime, 
                            user_message: str, assistant_message: str,
                            guild_id: Optional[int] = None):
        if not self.available:
            return False

        try:
            # Check for similar existing memories first
            existing_memories = self.search_memories(user_message + " " + assistant_message)
            
            # Only add if this is sufficiently different from existing memories
            if not any(mem["relevance"] > self.similarity_threshold for mem in existing_memories):
                conversation = {
                    "user_id": user_id,
                    "guild_id": guild_id,
                    "timestamp": timestamp,
                    "user_message": user_message,
                    "assistant_message": assistant_message,
                    "conversation_id": len(self.conversations)
                }
                
                self.conversations.append(conversation)
                
                combined_text = f"User: {user_message}\nAssistant: {assistant_message}"
                embedding = self.embed_text(combined_text)
                
                memory_id = len(self.memories)
                self.memories.append({
                    "id": memory_id,
                    "text": combined_text,
                    "conversation_id": conversation["conversation_id"],
                    "timestamp": timestamp,
                    "user_id": user_id,
                    "guild_id": guild_id
                })
                
                self.index.add(embedding.reshape(1, -1))
                self.conversation_index[conversation["conversation_id"]] = memory_id
                
                self.save_memories()
                return True
            else:
                self.logger.info("Similar memory already exists, skipping addition")
                return False
                
        except Exception as e:
            self.logger.error(f"Error adding conversation turn: {e}")
            return False

    def embed_text(self, text: str) -> np.ndarray:
        if not self.available or self.encoder is None:
            raise RuntimeError("Memory embeddings are not available")
        return self.encoder.encode(text)

    def save_memories(self):
        if not self.available:
            return

        try:
            save_data = {
                "memories": self.memories,
                "conversations": self.conversations,
                "conversation_index": self.conversation_index
            }
            
            memory_path = self.base_path / "memory_store.pkl"
            index_path = self.base_path / "faiss_index.pkl"
            backup_path = self.base_path / "memory_store.backup.pkl"
            
            if memory_path.exists():
                memory_path.rename(backup_path)
            
            with open(memory_path, 'wb') as f:
                pickle.dump(save_data, f)
                
            faiss.write_index(self.index, str(index_path))
            
            if backup_path.exists():
                backup_path.unlink()
                
        except Exception as e:
            self.logger.error(f"Error saving memories: {e}")
            if backup_path.exists():
                backup_path.rename(memory_path)

    def load_memories(self):
        if not self.available:
            return

        try:
            memory_path = self.base_path / "memory_store.pkl"
            index_path = self.base_path / "faiss_index.pkl"
            
            if memory_path.exists() and index_path.exists():
                with open(memory_path, 'rb') as f:
                    save_data = pickle.load(f)
                    
                self.memories = save_data["memories"]
                self.conversations = save_data["conversations"]
                self.conversation_index = save_data["conversation_index"]
                self.index = faiss.read_index(str(index_path))
                
        except Exception as e:
            self.logger.error(f"Error loading memories: {e}")
            self.memories = []
            self.conversations = []
            self.conversation_index = {}
            self.index = faiss.IndexFlatL2(self.embedding_dim)

    def clear_memories(self, guild_id: Optional[int] = None, user_id: Optional[str] = None):
        if not self.available:
            self.memories = []
            self.conversations = []
            self.conversation_index = {}
            return

        try:
            if guild_id is None and user_id is None:
                self.memories = []
                self.conversations = []
                self.conversation_index = {}
                self.index = faiss.IndexFlatL2(self.embedding_dim)
            else:
                new_memories = []
                new_index_data = []
                for memory in self.memories:
                    if ((guild_id is None or memory["guild_id"] != guild_id) and 
                        (user_id is None or memory["user_id"] != user_id)):
                        new_memories.append(memory)
                        embedding = self.embed_text(memory["text"])
                        new_index_data.append(embedding)
                
                self.memories = new_memories
                self.index = faiss.IndexFlatL2(self.embedding_dim)
                if new_index_data:
                    self.index.add(np.vstack(new_index_data))
                
            self.save_memories()
            
        except Exception as e:
            self.logger.error(f"Error clearing memories: {e}")

class UnifiedConversationHandler:
    def __init__(self, memory_store: EnhancedMemoryStore, llm_client):
        self.memory_store = memory_store
        self.llm_client = llm_client
        self.logger = logging.getLogger('UnifiedConversationHandler')

    def _build_messages(
        self,
        user_id: str,
        guild_id: int,
        message_content: str,
        context: Optional[dict] = None
    ) -> List[Dict[str, str]]:
        conversation_context = self.memory_store.get_conversation_context(
            user_id=user_id,
            current_message=message_content,
            guild_id=guild_id
        )

        recent_context = ""
        if context and 'recent_messages' in context:
            recent_context = "\n".join([
                f"{msg['author']}: {msg['content']}"
                for msg in context['recent_messages'][-5:]
            ])

        full_context = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Memory context:\n{conversation_context}\n"
            f"Recent conversation:\n{recent_context}"
        )

        return [
            {'role': 'system', 'content': full_context},
            {'role': 'user', 'content': f"{user_id}: {message_content}"}
        ]

    def _store_interaction(self, user_id: str, guild_id: int, message_content: str, response: Optional[str]):
        if not response:
            return

        self.memory_store.add_conversation_turn(
            user_id=user_id,
            timestamp=datetime.datetime.now(),
            user_message=message_content,
            assistant_message=response,
            guild_id=guild_id
        )

    async def process_interaction(
        self,
        user_id: str,
        guild_id: int,
        message_content: str,
        interaction_type: str = "chat",
        context: Optional[dict] = None
    ):
        try:
            messages = self._build_messages(
                user_id=user_id,
                guild_id=guild_id,
                message_content=message_content,
                context=context
            )
            response = await self.llm_client.generate_response(messages)
            self._store_interaction(user_id, guild_id, message_content, response)

            return {
                'should_respond': True,
                'response': response
            }

        except Exception as e:
            self.logger.error(f"Error processing interaction: {e}")
            return {
                'should_respond': False,
                'response': None
            }

    async def stream_interaction(
        self,
        user_id: str,
        guild_id: int,
        message_content: str,
        interaction_type: str = "voice",
        context: Optional[dict] = None
    ) -> AsyncIterator[str]:
        try:
            messages = self._build_messages(
                user_id=user_id,
                guild_id=guild_id,
                message_content=message_content,
                context=context
            )

            response_parts = []
            async for chunk in self.llm_client.stream_response(messages):
                response_parts.append(chunk)
                yield chunk

            self._store_interaction(
                user_id,
                guild_id,
                message_content,
                ''.join(response_parts).strip()
            )

        except Exception as e:
            self.logger.error(f"Error streaming interaction: {e}")


class ChannelContext:
    def __init__(self, max_messages=10):
        self.messages = []
        self.max_messages = max_messages
        self.last_bot_message = None
        self.logger = logging.getLogger('ChannelContext')
    
    def add_message(self, message, is_bot=False):
        message_data = {
            'author': str(message.author),
            'content': message.content,
            'timestamp': message.created_at.isoformat(),
            'is_bot': is_bot
        }
        
        if is_bot:
            self.last_bot_message = message_data
            
        self.messages.append(message_data)
        if len(self.messages) > self.max_messages:
            self.messages.pop(0)
    
    def get_context(self):
        return {
            'recent_messages': self.messages[-10:],
            'last_bot_message': self.last_bot_message
        }

    def was_last_message_from_bot(self):
        return self.messages and self.messages[-1].get('is_bot', False)

class Bot(commands.Bot):
    def __init__(self, command_prefix, intents, memory_store=None, llm_client=None, tts=None):
        super().__init__(command_prefix=commands.when_mentioned_or('!'), intents=intents)
        self.channel_contexts = {}
        self.llm_client = llm_client or build_llm_client()
        self.tts = tts or build_tts_provider()
        self.audio_processor = AudioProcessor()
        # Use pre-loaded memory store if provided, otherwise create new one
        self.memory_store = memory_store if memory_store else EnhancedMemoryStore()
        self.conversation_handler = UnifiedConversationHandler(self.memory_store, self.llm_client)
        self.logger = logging.getLogger('Bot')
        self.is_processing = False
    def scrub_bot_username(self, text: str) -> str:
        """
        Removes the bot's username and mentions from the message.
        """
        if not hasattr(self, 'user'):
            return text  # Return the original text if the bot's user is not initialized yet

        # Remove bot mentions (e.g., <@123456789012345678>)
        text = text.replace(f"<@{self.user.id}>", "").replace(f"<@!{self.user.id}>", "")
        
        # Remove bot username (e.g., Hikari-Chan#1660)
        bot_username = f"{self.user.name}#{self.user.discriminator}"
        text = text.replace(bot_username, "")
        
        # Clean up any extra spaces
        text = " ".join(text.split())
        return text        

    async def on_ready(self):
        self.logger.info('Logged in as {0.id}/{0}'.format(self.user))
        self.logger.info('Commands:')
        self.logger.info('- !vc - Join voice and start listening')
        self.logger.info('- !stop - Disconnect from voice')
        self.logger.info('- !die  - Shutdown bot')
        self.logger.info('------')

    async def process_message(self, message, response_content, use_tts=True):
        self.is_processing = True
        
        try:
            if response_content is None:
                result = await self.conversation_handler.process_interaction(
                    user_id=str(message.author),
                    guild_id=message.guild.id,
                    message_content=message.content,
                    interaction_type="direct_mention" if use_tts else "chat",
                    context=self.channel_contexts[message.channel.id].get_context()
                )
                if not result['should_respond'] or not result['response']:
                    return
                response_content = result['response']

            if response_content:
                response_content = response_content.replace(f"<@{self.user.id}>", "").replace(f"<@!{self.user.id}>", "")
                
                async with message.channel.typing():
                    if use_tts:
                        audio_file = await self.tts.generate_speech(response_content)
                        if audio_file:
                            await self.send_voice_message(message.channel, audio_file, response_content)
                        else:
                            await message.channel.send(response_content)
                    else:
                        await message.channel.send(response_content)
                    
                    self.channel_contexts[message.channel.id].add_message(
                        message=type('obj', (object,), {
                            'author': self.user,
                            'content': response_content,
                            'created_at': datetime.datetime.now()
                        }),
                        is_bot=True
                    )
        finally:
            self.is_processing = False

    async def send_voice_message(self, channel, wav_file, response_text):
        try:
            ogg_file = str(Path(wav_file).with_suffix('.ogg'))
            waveform, duration = self.audio_processor.fast_convert_and_analyze(wav_file, ogg_file)

            file_size = os.path.getsize(ogg_file)

            async with aiohttp.ClientSession() as session:
                upload_url_endpoint = f'https://discord.com/api/v10/channels/{channel.id}/attachments'
                headers = {
                    'Content-Type': 'application/json',
                    'Authorization': f'Bot {self.http.token}'
                }
                data = {
                    "files": [{
                        "filename": "voice-message.ogg",
                        "file_size": file_size,
                        "id": "2"
                    }]
                }
                
                async with session.post(upload_url_endpoint, headers=headers, json=data) as resp:
                    if resp.status != 200:
                        raise Exception(f"Failed to get upload URL: {await resp.text()}")
                    upload_data = await resp.json()
                    
                    upload_url = upload_data['attachments'][0]['upload_url']
                    upload_filename = upload_data['attachments'][0]['upload_filename']

                headers = {
                    'Content-Type': 'audio/ogg',
                    'Authorization': f'Bot {self.http.token}'
                }
                with open(ogg_file, 'rb') as f:
                    async with session.put(upload_url, headers=headers, data=f.read()) as resp:
                        if resp.status != 200:
                            raise Exception(f"Failed to upload file: {await resp.text()}")

                data = {
                    "flags": 8192,
                    "attachments": [{
                        "id": "0",
                        "filename": "voice-message.ogg",
                        "uploaded_filename": upload_filename,
                        "duration_secs": duration,
                        "waveform": waveform
                    }]
                }

                message_endpoint = f'https://discord.com/api/v10/channels/{channel.id}/messages'
                headers = {
                    'Content-Type': 'application/json',
                    'Authorization': f'Bot {self.http.token}'
                }
                async with session.post(message_endpoint, headers=headers, json=data) as resp:
                    if resp.status != 200:
                        raise Exception(f"Failed to send voice message: {await resp.text()}")

                await channel.send(content=response_text)

        except Exception as e:
            self.logger.error(f"Voice message error: {e}")
            await channel.send(content=response_text)
            await self.play_voice_fallback(channel, wav_file, response_text)
        finally:
            try:
                os.remove(wav_file)
                os.remove(ogg_file)
            except Exception as e:
                self.logger.error(f"Cleanup error: {e}")

    async def play_voice_fallback(self, channel, wav_file, response_text):
        try:
            voice_channel = None
            for vc in channel.guild.voice_channels:
                if len(vc.members) > 0:
                    voice_channel = vc
                    break

            if not voice_channel:
                await channel.send("No available voice channel to join.")
                return

            voice_client = await voice_channel.connect()
            source = discord.FFmpegPCMAudio(wav_file)
            voice_client.play(source, after=lambda e: print(f'Player error: {e}') if e else None)

            while voice_client.is_playing():
                await asyncio.sleep(1)

            await voice_client.disconnect()

        except Exception as e:
            self.logger.error(f"Voice fallback error: {e}")
            await channel.send("Failed to play voice message in a voice channel.")

class Testing(commands.Cog):
    def __init__(self, bot, tts):
        self.bot = bot
        self.tts = tts
        self.conversation_handler = UnifiedConversationHandler(bot.memory_store, bot.llm_client)
        self.logger = logging.getLogger('VoiceCog')
        
        # Multi-user session management
        self.user_sessions = {}  # user_id -> session data
        self.active_voice_clients = {}  # guild_id -> voice_client
        
        # LLM protection queue (protect that RTX 4060!)
        self.llm_semaphore = asyncio.Semaphore(1)  # Only 1 LLM call at a time
        self.llm_queue_size = 0
        
        print("Multi-user Voice cog initialized with LLM protection")

    def get_user_session(self, user_id):
        """Get or create user session data"""
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = {
                'conversation_history': [],
                'last_interaction': time.time(),
                'voice_channel': None,
                'processing': False
            }
        return self.user_sessions[user_id]

    def _supports_realtime_voice(self) -> bool:
        return (
            getattr(self.bot.llm_client, 'supports_streaming', False) and
            getattr(self.tts, 'supports_streaming_input', False)
        )

    def _get_active_voice_client(self):
        for _, voice_client in self.active_voice_clients.items():
            if voice_client and voice_client.is_connected():
                return voice_client, voice_client.channel
        return None, None

    def _extract_tts_chunk(self, buffer: str, force: bool = False):
        working = buffer.lstrip()
        if not working:
            return None, ''

        if force:
            return working.strip(), ''

        boundary = re.search(r'[.!?;:\n]+\s', working)
        if boundary:
            cutoff = boundary.end()
            return working[:cutoff].strip(), working[cutoff:].lstrip()

        if len(working) >= 40 and working.endswith(('.', '!', '?', ';', ':')):
            return working.strip(), ''

        if len(working) >= 120:
            cutoff = working.rfind(' ', 0, 120)
            if cutoff <= 0:
                cutoff = 120
            return working[:cutoff].strip(), working[cutoff:].lstrip()

        return None, working

    async def _send_voice_text_message(self, user, voice_channel, response_text):
        try:
            guild = voice_channel.guild
            text_channel = None
            
            if hasattr(voice_channel, 'send') and voice_channel.permissions_for(guild.me).send_messages:
                text_channel = voice_channel
                self.logger.info(f"Using voice channel text chat: {voice_channel.name}")
            
            if not text_channel:
                voice_name = voice_channel.name.lower()
                for channel in guild.text_channels:
                    if (channel.name.lower() == voice_name or
                        voice_name in channel.name.lower() or
                        channel.name.lower() in voice_name) and \
                       channel.permissions_for(guild.me).send_messages:
                        text_channel = channel
                        self.logger.info(f"Found matching text channel: {channel.name}")
                        break
            
            if not text_channel:
                priority_names = ['general', 'main', 'chat', 'bot', 'commands']
                for name in priority_names:
                    for channel in guild.text_channels:
                        if name in channel.name.lower() and channel.permissions_for(guild.me).send_messages:
                            text_channel = channel
                            self.logger.info(f"Using priority text channel: {channel.name}")
                            break
                    if text_channel:
                        break
            
            if not text_channel:
                for channel in guild.text_channels:
                    if channel.permissions_for(guild.me).send_messages:
                        text_channel = channel
                        self.logger.info(f"Using first available text channel: {channel.name}")
                        break
            
            if text_channel:
                await text_channel.send(f"🎤 **{user.display_name}**: {response_text}")
                self.logger.info(f"Sent text response to #{text_channel.name}")
            else:
                self.logger.warning("No suitable text channel found for response")
                
        except Exception as text_error:
            self.logger.error(f"Error sending text message: {text_error}")

    async def _stream_voice_response(self, user, prompt_text: str) -> Optional[str]:
        voice_client, voice_channel = self._get_active_voice_client()
        if not voice_client:
            self.logger.warning("No active voice client found")
            return None

        if voice_client.is_playing():
            self.logger.info("Voice client busy, falling back to non-streaming response")
            return None

        pcm_pipe = StreamingPCMInputBuffer()
        source = discord.FFmpegPCMAudio(
            pcm_pipe,
            pipe=True,
            before_options='-f s16le -ar 48000 -ac 1'
        )

        playback_error = None

        def after_playback(error):
            nonlocal playback_error
            playback_error = error
            pcm_pipe.close()

        voice_client.play(source, after=after_playback)
        self.logger.info(f"Started low-latency streaming playback in {voice_channel.name}")

        response_parts = []

        async def tts_chunks():
            buffer = ''
            async for token in self.conversation_handler.stream_interaction(
                user_id=str(user.id),
                guild_id=user.guild.id if hasattr(user, 'guild') else 0,
                message_content=prompt_text,
                interaction_type="voice"
            ):
                response_parts.append(token)
                buffer += token

                while True:
                    chunk, buffer = self._extract_tts_chunk(buffer)
                    if not chunk:
                        break
                    yield chunk

            final_chunk, _ = self._extract_tts_chunk(buffer, force=True)
            if final_chunk:
                yield final_chunk

        try:
            await self.tts.stream_text_chunks(tts_chunks(), on_audio_chunk=pcm_pipe.write)
            pcm_pipe.close()

            while voice_client.is_playing():
                await asyncio.sleep(0.1)

            if playback_error:
                raise RuntimeError(playback_error)

            response_text = ''.join(response_parts).strip()
            if response_text:
                await self._send_voice_text_message(user, voice_channel, response_text)
            return response_text

        except Exception as e:
            pcm_pipe.close()
            self.logger.error(f"Streaming voice response failed: {e}")
            return None

    async def process_voice_message(self, user, text: str):
        """Process voice message from multi-user transcription"""
        try:
            user_id = user.id
            session = self.get_user_session(user_id)
            
            # Prevent duplicate processing
            if session['processing']:
                self.logger.warning(f"⚠️ Already processing for user {user.display_name}, skipping duplicate")
                return
            
            session['processing'] = True
            session['last_interaction'] = time.time()
            
            self.logger.info(f"🎯 Processing voice from {user.display_name}: {text}")
            
            guild_id = user.guild.id if hasattr(user, 'guild') else 0
            response_text = None
            handled_by_streaming = False
            
            # Process with LLM protection queue
            async with self.llm_semaphore:
                self.llm_queue_size += 1
                try:
                    queue_pos = self.llm_queue_size
                    self.logger.info(f"🛡️ LLM Queue position {queue_pos} for {user.display_name}")
                    
                    if queue_pos > 1:
                        self.logger.info(f"⏳ {user.display_name} waiting in queue (position {queue_pos})")

                    if self._supports_realtime_voice():
                        self.logger.info(f"⚡ Streaming LLM + TTS enabled for {user.display_name}")
                        response_text = await self._stream_voice_response(user, text)
                        handled_by_streaming = bool(response_text)

                    if not handled_by_streaming:
                        self.logger.info(f"🤖 Sending to LLM for user {user.display_name}")
                        result = await self.conversation_handler.process_interaction(
                            user_id=str(user_id),
                            guild_id=guild_id,
                            message_content=text,
                            interaction_type="voice"
                        )
                        self.logger.info(f"✅ LLM response received for user {user.display_name}")
                        if result['should_respond'] and result['response']:
                            response_text = result['response']
                finally:
                    self.llm_queue_size -= 1
            
            if response_text and not handled_by_streaming:
                await self.send_voice_response(user, response_text)

        except Exception as e:
            self.logger.error(f"Error processing voice message for {user.display_name}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if user_id in self.user_sessions:
                self.user_sessions[user_id]['processing'] = False
                self.logger.info(f"🏁 Finished processing for user {user.display_name}")

    async def send_voice_response(self, user, response_text):
        """Send voice response to connected voice channel"""
        try:
            voice_client, voice_channel = self._get_active_voice_client()
            
            if not voice_client:
                self.logger.warning("No active voice client found")
                return
            
            self.logger.info(f"Sending voice response to {voice_channel.name}")
            
            # Generate and play speech
            audio_file = await self.tts.generate_speech(response_text)
            if audio_file:
                await self.play_audio_non_blocking(voice_client, audio_file)

            await self._send_voice_text_message(user, voice_channel, response_text)

        except Exception as e:
            self.logger.error(f"Error sending voice response: {e}")

    async def play_audio_non_blocking(self, voice_client, wav_file):
        """Play audio without blocking other users"""
        try:
            if not voice_client or not voice_client.is_connected():
                return
            
            # Check if already playing - if so, queue or skip
            if voice_client.is_playing():
                self.logger.info("Voice client busy, skipping audio")
                try:
                    os.remove(wav_file)
                except:
                    pass
                return
                
            source = discord.FFmpegPCMAudio(wav_file)
            voice_client.play(source, after=lambda e: self._audio_finished(wav_file, e))
            self.logger.info(f"Started playing audio: {wav_file}")
            
            # Wait for audio to finish playing
            while voice_client.is_playing():
                await asyncio.sleep(0.1)
            self.logger.info("Audio playback completed")
            
        except Exception as e:
            self.logger.error(f"Error playing audio: {e}")
            try:
                os.remove(wav_file)
            except:
                pass

    def _audio_finished(self, wav_file, error):
        """Cleanup after audio finishes"""
        if error:
            self.logger.error(f"Audio playback error: {error}")
        try:
            os.remove(wav_file)
        except Exception as e:
            self.logger.error(f"Error removing audio file: {e}")

    async def handle_text(self, user, text: str):
        """Legacy method - redirects to new multi-user processing"""
        await self.process_voice_message(user, text)

    async def play_audio(self, voice_client, wav_file):
        try:
            if not voice_client or not voice_client.is_connected():
                return
                
            source = discord.FFmpegPCMAudio(wav_file)
            voice_client.play(source, after=lambda e: print(f'Player error: {e}') if e else None)
            
            while voice_client.is_playing():
                await asyncio.sleep(0.1)
                
            try:
                os.remove(wav_file)
            except Exception as e:
                self.logger.error(f"Error removing audio file: {e}")
                
        except Exception as e:
            self.logger.error(f"Error playing audio: {e}")
            import traceback
            traceback.print_exc()

    @commands.command()
    async def vc(self, ctx):
        if not ctx.author.voice:
            await ctx.send("You need to be in a voice channel!")
            return

        try:
            self.logger.info(f"Joining channel: {ctx.author.voice.channel}")
            
            await self.tts.initialize()
            self.logger.info(f"{self.tts.__class__.__name__} initialized")
            
            # Disconnect any existing connection first
            if ctx.voice_client:
                await ctx.voice_client.disconnect()
                await asyncio.sleep(1)
            
            # Connect with new discord.py version (includes 4006 fix)
            vc = await ctx.author.voice.channel.connect(cls=voice_recv.VoiceRecvClient)
            self.logger.info("Connected to voice channel")
            
            # Track voice client for multi-user management
            guild_id = ctx.guild.id
            self.active_voice_clients[guild_id] = vc

            sink = VoiceSink(self, self.bot)
            self.logger.info("Created multi-user voice sink")
            
            vc.listen(sink)
            self.logger.info("Started listening for multiple users")
            
            await ctx.send("🎙️ Multi-user voice chat ready! Everyone can speak!")
            
        except Exception as e:
            self.logger.error(f"Error joining voice: {e}")
            if ctx.voice_client:
                await ctx.voice_client.disconnect()
            await ctx.send("❌ Failed to join voice channel")

    @commands.command()
    async def stop(self, ctx):
        guild_id = ctx.guild.id
        if ctx.voice_client:
            # Clean up voice client tracking
            if guild_id in self.active_voice_clients:
                del self.active_voice_clients[guild_id]
            
            # Clear user sessions for this guild
            users_to_remove = [user_id for user_id, session in self.user_sessions.items() 
                             if session.get('voice_channel') and session['voice_channel'].guild.id == guild_id]
            for user_id in users_to_remove:
                del self.user_sessions[user_id]
            
            await ctx.voice_client.disconnect()
            await ctx.send("👋 Multi-user voice chat stopped!")
        else:
            await ctx.send("❌ Not in a voice channel!")

    @commands.command()
    async def die(self, ctx):
        if ctx.voice_client:
            ctx.voice_client.stop()
        await ctx.send("💤 Shutting down...")
        await ctx.bot.close()

class InworldTTS:
    supports_streaming_input = True

    def __init__(
        self,
        api_key: Optional[str] = None,
        voice_id: str = 'Dennis',
        model_id: str = 'inworld-tts-1.5-max',
        endpoint: str = 'wss://api.inworld.ai/tts/v1/voice:streamBidirectional',
        output_dir: str = 'output'
    ):
        self.api_key = api_key or os.getenv('INWORLD_TTS_API_KEY')
        self.voice_id = os.getenv('INWORLD_TTS_VOICE_ID', voice_id)
        self.model_id = os.getenv('INWORLD_TTS_MODEL_ID', model_id)
        self.endpoint = os.getenv('INWORLD_TTS_ENDPOINT', endpoint)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.sample_rate_hz = env_int('INWORLD_TTS_SAMPLE_RATE_HZ', 48000)
        self.buffer_char_threshold = env_int('INWORLD_TTS_BUFFER_CHAR_THRESHOLD', 40)
        self.auto_mode = env_bool('INWORLD_TTS_AUTO_MODE', True)
        self.timeout_seconds = env_int('INWORLD_TTS_TIMEOUT_SECONDS', 90)
        self.timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        self.initialized = False
        self.logger = logging.getLogger('InworldTTS')

    async def initialize(self) -> bool:
        if self.initialized:
            return True

        if not self.api_key:
            self.logger.error('Missing INWORLD_TTS_API_KEY')
            return False

        self.initialized = True
        return True

    @property
    def headers(self) -> Dict[str, str]:
        return {
            'Authorization': f'Basic {self.api_key}'
        }

    async def _emit_audio_chunk(self, callback, chunk: bytes):
        if not callback or not chunk:
            return

        result = callback(chunk)
        if asyncio.iscoroutine(result):
            await result

    def _decode_audio_chunk(self, audio_content: str) -> bytes:
        if not audio_content:
            return b''

        chunk = base64.b64decode(audio_content)
        if len(chunk) <= 44:
            return b''

        if chunk.startswith(b'RIFF'):
            return chunk[44:]

        return chunk

    async def _iter_chunks(self, chunks) -> AsyncIterator[str]:
        if hasattr(chunks, '__aiter__'):
            async for chunk in chunks:
                yield chunk
            return

        for chunk in chunks:
            yield chunk

    async def stream_text_chunks(self, chunks, on_audio_chunk=None) -> bytes:
        if not await self.initialize():
            return b''

        context_id = f'ctx-{uuid.uuid4().hex[:8]}'
        pending_flushes = 0
        completed_flushes = 0
        raw_audio = bytearray()
        receiver_error = None
        context_ready = asyncio.Event()

        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            async with session.ws_connect(
                self.endpoint,
                headers=self.headers,
                heartbeat=30
            ) as ws:
                async def receiver():
                    nonlocal completed_flushes, receiver_error
                    try:
                        while True:
                            message = await ws.receive()

                            if message.type == aiohttp.WSMsgType.TEXT:
                                payload = json.loads(message.data)
                                result = payload.get('result') or {}
                                status = result.get('status') or {}

                                if status.get('code', 0) not in (0, None):
                                    raise RuntimeError(f"Inworld TTS error: {status}")

                                if 'contextCreated' in result:
                                    context_ready.set()

                                if 'audioChunk' in result:
                                    pcm_chunk = self._decode_audio_chunk(
                                        result['audioChunk'].get('audioContent', '')
                                    )
                                    if pcm_chunk:
                                        raw_audio.extend(pcm_chunk)
                                        await self._emit_audio_chunk(on_audio_chunk, pcm_chunk)

                                if 'flushCompleted' in result:
                                    completed_flushes += 1

                                if 'contextClosed' in result:
                                    break

                            elif message.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED):
                                break
                            elif message.type == aiohttp.WSMsgType.ERROR:
                                raise RuntimeError('Inworld websocket returned an error frame')
                    except Exception as e:
                        receiver_error = e
                        context_ready.set()

                receiver_task = asyncio.create_task(receiver())

                await ws.send_json({
                    'create': {
                        'voiceId': self.voice_id,
                        'modelId': self.model_id,
                        'audioConfig': {
                            'audioEncoding': 'LINEAR16',
                            'sampleRateHertz': self.sample_rate_hz
                        },
                        'bufferCharThreshold': self.buffer_char_threshold,
                        'autoMode': self.auto_mode,
                        'timestampType': 'WORD',
                        'timestampTransportStrategy': 'ASYNC'
                    },
                    'contextId': context_id
                })

                await asyncio.wait_for(context_ready.wait(), timeout=10)
                if receiver_error:
                    raise receiver_error

                async for chunk in self._iter_chunks(chunks):
                    chunk = (chunk or '').strip()
                    if not chunk:
                        continue

                    pending_flushes += 1
                    await ws.send_json({
                        'send_text': {
                            'text': chunk,
                            'flush_context': {}
                        },
                        'contextId': context_id
                    })

                while completed_flushes < pending_flushes:
                    if receiver_error:
                        raise receiver_error
                    await asyncio.sleep(0.02)

                await ws.send_json({
                    'close_context': {},
                    'contextId': context_id
                })

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

            output_file = self.output_dir / f'output_{int(time.time() * 1000)}.wav'
            with wave.open(str(output_file), 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.sample_rate_hz)
                wav_file.writeframes(pcm_audio)

            return str(output_file)
        except Exception as e:
            self.logger.error(f'Error generating Inworld speech: {e}')
            return None

    async def cleanup(self):
        self.initialized = False


class PiperTTS:
    supports_streaming_input = False

    def __init__(self, 
                 piper_path: str = 'piper/piper.exe',
                 model_path: str = 'piper/en_US-ryari-high.onnx',
                 model_config: str = 'piper/en_US-ryari-high.onnx.json',
                 output_dir: str = 'output',
                 timeout: int = 10):
        self.piper_path = Path(piper_path)
        self.model_path = Path(model_path)
        self.model_config = Path(model_config)
        self.output_dir = Path(output_dir)
        self.timeout = timeout
        
        self.process: Optional[subprocess.Popen] = None
        self.initialized = False
        self.lock = asyncio.Lock()
        self.current_generation: Optional[asyncio.Task] = None
        
        self.logger = logging.getLogger('PiperTTS')
        self.logger.setLevel(logging.INFO)
        
        self.output_dir.mkdir(exist_ok=True)
        
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_generation_time': 0
        }

    async def verify_files(self) -> bool:
        required_files = [
            (self.piper_path, "Piper executable"),
            (self.model_path, "Model file"),
            (self.model_config, "Model configuration")
        ]
        
        for file_path, description in required_files:
            if not file_path.exists():
                self.logger.error(f"Missing {description} at {file_path}")
                return False
        return True

    async def initialize(self) -> bool:
        if self.initialized:
            return True
            
        try:
            async with self.lock:
                if not await self.verify_files():
                    return False
                
                if self.process and self.process.poll() is None:
                    self.process.terminate()
                    await asyncio.sleep(0.1)
                    
                self.process = subprocess.Popen(
                    [
                        str(self.piper_path),
                        '-m', str(self.model_path),
                        '-c', str(self.model_config),
                        '--json-input'
                    ],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                await asyncio.sleep(0.1)
                if self.process.poll() is not None:
                    stderr = self.process.stderr.read()
                    self.logger.error(f"Piper failed to start: {stderr}")
                    return False
                
                self.initialized = True
                self.logger.info("Piper initialized successfully")
                return True
                
        except Exception as e:
            self.logger.error(f"Initialization failed: {str(e)}")
            return False

    async def generate_speech(self, text: str) -> Optional[str]:
        start_time = time.time()
        self.stats['total_requests'] += 1
        
        try:
            if not self.initialized and not await self.initialize():
                return None
                
            async with self.lock:
                output_file = str(self.output_dir / f'output_{int(time.time() * 1000)}.wav')
                input_json = {
                    'text': text,
                    'output_file': output_file,
                    'length_scale': 1.0,
                    'noise_scale': 0.667,
                    'noise_w': 0.8
                }
                
                process = await asyncio.create_subprocess_exec(
                    str(self.piper_path),
                    '-m', str(self.model_path),
                    '-c', str(self.model_config),
                    '--json-input',
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                try:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(json.dumps(input_json).encode() + b'\n'),
                        timeout=self.timeout
                    )
                    
                    if process.returncode != 0:
                        self.logger.error(f"Piper process failed: {stderr.decode()}")
                        self.stats['failed_requests'] += 1
                        return None
                    
                    if not os.path.exists(output_file):
                        self.logger.error("Output file was not created")
                        self.stats['failed_requests'] += 1
                        return None
                        
                    self.stats['successful_requests'] += 1
                    generation_time = time.time() - start_time
                    self.stats['average_generation_time'] = (
                        (self.stats['average_generation_time'] * 
                         (self.stats['successful_requests'] - 1) +
                         generation_time) / self.stats['successful_requests']
                    )
                    
                    self.logger.info(f"Generated speech in {generation_time:.2f}s: {output_file}")
                    return output_file
                    
                except asyncio.TimeoutError:
                    self.logger.error("Piper process timed out")
                    process.kill()
                    self.stats['failed_requests'] += 1
                    return None
                    
                except Exception as e:
                    self.logger.error(f"Generation error: {str(e)}")
                    self.stats['failed_requests'] += 1
                    return None
                    
        except Exception as e:
            self.logger.error(f"Unexpected error in generate_speech: {str(e)}")
            self.stats['failed_requests'] += 1
            return None

    async def cleanup(self):
        try:
            if self.process and self.process.poll() is None:
                self.process.terminate()
                try:
                    await asyncio.wait_for(
                        asyncio.create_task(asyncio.sleep(0)),
                        timeout=2
                    )
                except asyncio.TimeoutError:
                    self.process.kill()
                    
            for file in self.output_dir.glob('*.wav'):
                try:
                    os.remove(file)
                except Exception as e:
                    self.logger.error(f"Failed to remove file {file}: {str(e)}")
                    
        except Exception as e:
            self.logger.error(f"Cleanup error: {str(e)}")
        finally:
            self.initialized = False
            self.process = None


def build_tts_provider():
    if TTS_PROVIDER == 'inworld':
        return InworldTTS()

    if TTS_PROVIDER == 'piper':
        return PiperTTS()

    raise ValueError(f'Unsupported TTS provider: {TTS_PROVIDER}')

class AudioProcessor:
    @staticmethod
    def get_audio_info(file_path):
        try:
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                file_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            data = json.loads(result.stdout)
            
            duration = float(data['format']['duration'])
            stream = next((s for s in data['streams'] if s['codec_type'] == 'audio'), None)
            sample_rate = int(stream['sample_rate']) if stream else 48000
            channels = int(stream['channels']) if stream else 1
            
            return duration, sample_rate, channels
        except Exception as e:
            print(f"Error getting audio info: {e}")
            return 5.0, 48000, 1

    @staticmethod
    def fast_convert_and_analyze(input_file, output_file):
        try:
            duration, sample_rate, channels = AudioProcessor.get_audio_info(input_file)

            cmd = [
                'ffmpeg',
                '-i', input_file,
                '-vn',
                '-ar', '48000',
                '-ac', '1',
                '-f', 's16le',
                '-acodec', 'pcm_s16le',
                'pipe:1',
                '-y'
            ]
            
            process = subprocess.run(cmd, capture_output=True)
            audio_data = np.frombuffer(process.stdout, dtype=np.int16)
            
            amplitudes = np.abs(audio_data)
            segments = np.array_split(amplitudes, 256)
            waveform = np.array([np.max(segment) if len(segment) > 0 else 0 for segment in segments])
            
            max_val = np.max(waveform)
            if max_val > 0:
                waveform = (waveform / max_val * 255).astype(np.uint8)
                window_size = 3
                smoothed = np.convolve(waveform, np.ones(window_size)/window_size, mode='same')
                waveform = smoothed.astype(np.uint8)
                waveform = np.maximum(waveform, 10)
            else:
                waveform = np.full(256, 128, dtype=np.uint8)
            
            waveform_base64 = base64.b64encode(bytes(waveform.tolist())).decode('utf-8')

            subprocess.run([
                'ffmpeg',
                '-i', input_file,
                '-c:a', 'libopus',
                '-b:a', '64k',
                output_file
            ], check=True)

            return waveform_base64, duration

        except Exception as e:
            print(f"Error in fast convert and analyze: {e}")
            return base64.b64encode(bytes([128] * 256)).decode('utf-8'), 5.0

class VoiceSink(voice_recv.AudioSink):
    def __init__(self, cog, bot):
        super().__init__()
        self.cog = cog
        self.bot = bot
        self.decode = True
        
        # Initialize Faster-Whisper (CPU mode for stability)
        try:
            # Force CPU mode until PyTorch CUDA issues are resolved
            self.whisper_model = WhisperModel(
                "base",  # Good balance of speed and accuracy
                device="cpu",
                compute_type="int8"
            )
            print(f"🎤 Faster-Whisper initialized on CPU (stable mode)")
        except Exception as e:
            print(f"❌ Faster-Whisper failed: {e}")
            self.whisper_model = None
            
        # Multi-user recording state
        self.user_recordings = {}  # user_id -> recording data
        self.output_dir = Path('recordings')
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger('VoiceSink')
        
        # Voice Activity Detection settings
        self.SILENCE_THRESHOLD_MS = 1500  # Stop recording after 1.5s silence
        self.MIN_RECORDING_MS = 500       # Minimum recording duration
        self.AMPLITUDE_THRESHOLD = 100    # Minimum amplitude to detect speech
        
        print("Multi-user VoiceSink initialized")

    def wants_opus(self) -> bool:
        return False

    def cleanup(self):
        self.logger.info("VoiceSink cleanup")
        self.decode = False
        self.user_recordings = {}

    def is_speaking(self, audio_data):
        """Detect if user is speaking based on amplitude"""
        max_amplitude = np.max(np.abs(audio_data))
        return max_amplitude > self.AMPLITUDE_THRESHOLD

    async def finalize_recording(self, user_id):
        """Save and queue recording for transcription"""
        if user_id not in self.user_recordings:
            return
            
        recording_data = self.user_recordings[user_id]
        buffer = recording_data['buffer']
        
        # Check minimum duration
        duration_ms = len(buffer) * 20  # 20ms per chunk
        if duration_ms < self.MIN_RECORDING_MS:
            self.logger.debug(f"Recording too short for user {user_id}: {duration_ms}ms")
            self.user_recordings[user_id] = self._init_user_recording()
            return
        
        # Save audio file
        filename = f"temp_{user_id}_{int(time.time() * 1000)}.wav"
        filepath = self.output_dir / filename
        
        try:
            # Combine all audio chunks
            combined_audio = b''.join(buffer)
            audio_np = np.frombuffer(combined_audio, dtype=np.int16)
            
            # Save as WAV file
            with wave.open(str(filepath), 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(48000)  # 48kHz
                wav_file.writeframes(combined_audio)
            
            self.logger.info(f"Saved recording for user {user_id}: {filename} ({duration_ms}ms)")
            
            # Queue for async transcription (non-blocking)
            asyncio.run_coroutine_threadsafe(
                self.transcribe_and_cleanup(str(filepath), user_id),
                self.bot.loop
            )
            
        except Exception as e:
            self.logger.error(f"Error saving recording for user {user_id}: {e}")
        
        # Reset user recording state
        self.user_recordings[user_id] = self._init_user_recording()

    def _init_user_recording(self):
        """Initialize recording state for a user"""
        return {
            'buffer': [],
            'last_activity': time.time(),
            'recording': False,
            'start_time': None
        }

    async def transcribe_and_cleanup(self, filepath, user_id):
        """Transcribe audio file and clean up (async)"""
        try:
            if not self.whisper_model:
                self.logger.error("Whisper model not available")
                return
            
            # Transcribe audio
            segments, info = self.whisper_model.transcribe(filepath, beam_size=5)
            transcription = " ".join([segment.text for segment in segments]).strip()
            
            if transcription:
                self.logger.info(f"Transcription for user {user_id}: {transcription}")
                
                # Get user object - try cache first, then API
                user = self.bot.get_user(user_id)
                if user:
                    self.logger.info(f"Found user {user.display_name} in cache, processing message...")
                else:
                    self.logger.info(f"User {user_id} not in cache, fetching from Discord API...")
                    try:
                        user = await self.bot.fetch_user(user_id)
                        if user:
                            self.logger.info(f"Fetched user {user.display_name} from API, processing message...")
                        else:
                            self.logger.error(f"Could not fetch user {user_id} from Discord API")
                            return
                    except Exception as fetch_error:
                        self.logger.error(f"Error fetching user {user_id}: {fetch_error}")
                        return
                
                # Process the message (only called once now)
                if user:
                    await self.cog.process_voice_message(user, transcription)
            
        except Exception as e:
            self.logger.error(f"Transcription error for user {user_id}: {e}")
        
        finally:
            # Clean up file
            try:
                os.remove(filepath)
                self.logger.debug(f"Cleaned up file: {filepath}")
            except Exception as e:
                self.logger.error(f"Error removing file {filepath}: {e}")

    def write(self, user, data: voice_recv.VoiceData):
        try:
            if user is None or data.pcm is None:
                return
            
            user_id = user.id
            
            # Initialize user recording if not exists
            if user_id not in self.user_recordings:
                self.user_recordings[user_id] = self._init_user_recording()
            
            recording = self.user_recordings[user_id]
            
            # Only record if we're in recording state (set by Discord speaking events)
            if recording['recording']:
                try:
                    audio_np = np.frombuffer(data.pcm, dtype=np.int16)
                    audio_stereo = audio_np.reshape(-1, 2)
                    audio_mono = audio_stereo.mean(axis=1).astype(np.int16)
                    
                    recording['buffer'].append(audio_mono.tobytes())
                    recording['last_activity'] = time.time()
                    
                except Exception as e:
                    self.logger.error(f"Error processing audio: {e}")
                
        except Exception as e:
            self.logger.error(f"Error in write: {e}")

    @voice_recv.AudioSink.listener()
    def on_voice_member_speaking_start(self, member):
        """Discord detected user started speaking (push-to-talk pressed)"""
        try:
            user_id = member.id
            if user_id not in self.user_recordings:
                self.user_recordings[user_id] = self._init_user_recording()
            
            recording = self.user_recordings[user_id]
            if not recording['recording']:
                recording['recording'] = True
                recording['start_time'] = time.time()
                recording['buffer'] = []  # Clear any old data
                self.logger.info(f"🎤 Started recording for {member.display_name}")
                
        except Exception as e:
            self.logger.error(f"Error in speaking start: {e}")

    @voice_recv.AudioSink.listener()
    def on_voice_member_speaking_stop(self, member):
        """Discord detected user stopped speaking (push-to-talk released)"""
        try:
            user_id = member.id
            if user_id in self.user_recordings and self.user_recordings[user_id]['recording']:
                self.logger.info(f"🎤 Stopped recording for {member.display_name}")
                # Immediately finalize recording when user releases push-to-talk
                asyncio.run_coroutine_threadsafe(
                    self.finalize_recording(user_id),
                    self.bot.loop
                )
                
        except Exception as e:
            self.logger.error(f"Error in speaking stop: {e}")


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('discord_bot.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    # Fix Windows Unicode issues
    import sys
    if sys.platform == 'win32':
        import os
        os.environ['PYTHONIOENCODING'] = 'utf-8'

def main():
    setup_logging()
    apply_voice_recv_compat_patches()
    logger = logging.getLogger('main')
    
    # Pre-load models before Discord connection
    logger.info("🔄 Pre-loading AI models...")
    
    # Initialize memory store (loads SentenceTransformer on GPU)
    logger.info("📚 Loading memory system...")
    memory_store = EnhancedMemoryStore()
    
    # Pre-load Faster-Whisper (CPU mode for stability)
    logger.info("🎤 Loading Faster-Whisper...")
    try:
        # Use CPU mode until PyTorch CUDA issues are resolved
        whisper_model = WhisperModel(
            "base",
            device="cpu",
            compute_type="int8"
        )
        logger.info("✅ Faster-Whisper loaded on CPU (stable mode)")
    except Exception as e:
        logger.error(f"❌ Faster-Whisper loading failed: {e}")
    
    llm_client = build_llm_client()
    tts = build_tts_provider()

    # Pre-warm LLM backend
    logger.info(f"🤖 Pre-warming {llm_client.__class__.__name__}...")
    try:
        asyncio.run(llm_client.warmup())
        logger.info(f"✅ {llm_client.__class__.__name__} warmed up")
    except Exception as e:
        logger.warning(f"⚠️ LLM warmup failed: {e}")
    
    logger.info("✅ All models loaded! Starting Discord bot...")
    
    intents = discord.Intents.default()
    intents.message_content = True
    intents.voice_states = True
    intents.guilds = True

    bot = Bot(
        command_prefix='!',
        intents=intents,
        memory_store=memory_store,
        llm_client=llm_client,
        tts=tts
    )

    @bot.event
    async def on_message(message):
        if message.author == bot.user:
            return

        if message.channel.id not in bot.channel_contexts:
            bot.channel_contexts[message.channel.id] = ChannelContext()

        context = bot.channel_contexts[message.channel.id].get_context()

        # Scrub the bot's username from the message content
        scrubbed_content = bot.scrub_bot_username(message.content)

        is_direct = (bot.user in message.mentions or (
            message.reference and 
            message.reference.resolved and 
            message.reference.resolved.author == bot.user
        ))

        # Always check for commands first
        await bot.process_commands(message)
        
        # Then handle direct mentions/replies if it's not a command
        if is_direct and not message.content.startswith('!'):
            logger.info("\nProcessing direct mention...")
            response = await bot.conversation_handler.process_interaction(
                user_id=str(message.author),
                guild_id=message.guild.id,
                message_content=scrubbed_content,
                interaction_type="direct_mention",
                context=context
            )
            if response['response']:
                await bot.process_message(message, response['response'], use_tts=True)

    async def setup_hook():
        await bot.add_cog(Testing(bot, bot.tts))

    bot.setup_hook = setup_hook

    try:
        logger.info("Starting bot...")
        logger.info("Make sure:")
        logger.info(f"1. TTS provider is configured: {TTS_PROVIDER}")
        logger.info("2. 'recordings' and 'output' directories exist")
        logger.info(f"3. LLM provider is available: {LLM_PROVIDER}")
        logger.info("\nStarting bot now...")
        bot.run(DISCORD_BOT_TOKEN)
    except Exception as e:
        logger.error(f"Error running bot: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
