import discord
import json
import ollama
import datetime
import subprocess
import time
import aiohttp
import base64
import os
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
from sentence_transformers import SentenceTransformer
import faiss
from typing import Optional, Dict, Any, List
import logging

# Discord Bot Token
DISCORD_BOT_TOKEN = os.getenv('DISCORD_BOT_TOKEN')  # Load from environment variable

# System Prompt
SYSTEM_PROMPT = """You are Hikari-chan, a lively and engaging AI Discord bot inspired by Hinata from Naruto. You combine Hinata’s kindness and modesty with a playful, sharp-witted, and occasionally unpredictable personality, making conversations engaging, fun, and dynamic.

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

class EnhancedMemoryStore:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
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
        print(f"📊 Using CPU FAISS (optimal for small datasets), GPU for embeddings")
        
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
        self.logger = logging.getLogger('MemoryStore')
        
        self.load_memories()

    def get_conversation_context(self, user_id: str, current_message: str, 
                               guild_id: Optional[int] = None,
                               max_context: int = 3) -> str:
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
        return self.encoder.encode(text)

    def save_memories(self):
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
    def __init__(self, memory_store: EnhancedMemoryStore):
        self.memory_store = memory_store
        self.logger = logging.getLogger('UnifiedConversationHandler')

    async def process_interaction(
        self,
        user_id: str,
        guild_id: int,
        message_content: str,
        interaction_type: str = "chat",
        context: Optional[dict] = None
    ):
        try:
            # Get historical context from memory store
            conversation_context = self.memory_store.get_conversation_context(
                user_id=user_id,
                current_message=message_content,
                guild_id=guild_id
            )

            # Combine memory context with recent messages
            recent_context = ""
            if context and 'recent_messages' in context:
                recent_context = "\n".join([
                    f"{msg['author']}: {msg['content']}"
                    for msg in context['recent_messages'][-5:]
                ])

            # Prepare the complete context
            full_context = f"{SYSTEM_PROMPT}\n\nMemory context:\n{conversation_context}\nRecent conversation:\n{recent_context}"

            # Always generate response for direct mentions, replies, or pings
            messages = [
                {'role': 'system', 'content': full_context},
                {'role': 'user', 'content': f"{user_id}: {message_content}"}
            ]
            response = await self._generate_response(messages)

            if response:
                # Store the interaction in memory
                self.memory_store.add_conversation_turn(
                    user_id=user_id,
                    timestamp=datetime.datetime.now(),
                    user_message=message_content,
                    assistant_message=response,
                    guild_id=guild_id
                )

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

    async def _generate_response(self, messages: List[Dict[str, str]]) -> str:
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: ollama.chat(
                    model='hf.co/subsectmusic/qwriko3-4b-instruct-2507:Q4_K_M',
                    messages=messages,
                    options={
                        'num_predict': 2048,        # Maximum number of tokens to generate
                        'temperature': 0.8,         # Higher temperature (0-1) increases creativity/randomness
                        'top_k': 40,               # Limit vocabulary to top K options per token
                        'top_p': 0.9,              # Nucleus sampling threshold
                        'repeat_penalty': 1.1,     # Penalize repetition (>1.0 reduces repetition)
                        'presence_penalty': 0.2,    # Penalize tokens based on presence in context
                        'frequency_penalty': 0.2    # Penalize tokens based on frequency in context
                    }
                )
            )
            return response['message']['content']
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return None


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
    def __init__(self, command_prefix, intents, memory_store=None):
        super().__init__(command_prefix=commands.when_mentioned_or('!'), intents=intents)
        self.channel_contexts = {}
        self.piper = PiperTTS()
        self.audio_processor = AudioProcessor()
        # Use pre-loaded memory store if provided, otherwise create new one
        self.memory_store = memory_store if memory_store else EnhancedMemoryStore()
        self.conversation_handler = UnifiedConversationHandler(self.memory_store)
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
            result = await self.conversation_handler.process_interaction(
                user_id=str(message.author),
                guild_id=message.guild.id,
                message_content=message.content,
                interaction_type="direct_mention" if use_tts else "chat",
                context=self.channel_contexts[message.channel.id].get_context()
            )
            
            if result['should_respond'] and result['response']:
                response_content = result['response']
                response_content = response_content.replace(f"<@{self.user.id}>", "").replace(f"<@!{self.user.id}>", "")
                
                async with message.channel.typing():
                    if use_tts:
                        wav_file = await self.piper.generate_speech(response_content)
                        if wav_file:
                            await self.send_voice_message(message.channel, wav_file, response_content)
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
            ogg_file = wav_file.replace('.wav', '.ogg')
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
    def __init__(self, bot, piper):
        self.bot = bot
        self.piper = piper
        self.conversation_handler = UnifiedConversationHandler(bot.memory_store)
        self.logger = logging.getLogger('VoiceCog')
        print("Voice cog initialized")

    async def handle_text(self, user, text: str):
        try:
            self.logger.info(f"Got text from {user}: {text}")
            
            guild_id = user.guild.id if hasattr(user, 'guild') else 0
            
            # Process the voice interaction
            result = await self.conversation_handler.process_interaction(
                user_id=str(user),
                guild_id=guild_id,
                message_content=text,
                interaction_type="voice"
            )
            
            if result['should_respond'] and result['response']:
                response = result['response']
                if hasattr(user, 'voice') and user.voice and user.voice.channel:
                    wav_file = await self.piper.generate_speech(response)
                    if wav_file:
                        voice_client = user.voice.channel.guild.voice_client
                        if voice_client:
                            await self.play_audio(voice_client, wav_file)
                    await user.voice.channel.send(f"🎤 {response}")

        except Exception as e:
            self.logger.error(f"Error handling text: {e}")
            import traceback
            traceback.print_exc()

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
            
            await self.piper.initialize()
            self.logger.info("Piper initialized")
            
            # Disconnect any existing connection first
            if ctx.voice_client:
                await ctx.voice_client.disconnect()
                await asyncio.sleep(1)
            
            # Connect with new discord.py version (includes 4006 fix)
            vc = await ctx.author.voice.channel.connect(cls=voice_recv.VoiceRecvClient)
            self.logger.info("Connected to voice channel")
            
            # Wait for connection to stabilize
            await asyncio.sleep(2)
            
            # Verify connection before proceeding
            if not vc.is_connected():
                raise Exception("Voice connection failed to establish properly")
            
            sink = VoiceSink(self, self.bot)
            self.logger.info("Created voice sink")
            
            vc.listen(sink)
            self.logger.info("Started listening")
            
            await ctx.send("🎙️ Ready to chat!")
            
        except Exception as e:
            self.logger.error(f"Error joining voice: {e}")
            if ctx.voice_client:
                await ctx.voice_client.disconnect()
            await ctx.send("❌ Failed to join voice channel")

    @commands.command()
    async def stop(self, ctx):
        if ctx.voice_client:
            if ctx.guild.id in self.bot.conversations:
                del self.bot.conversations[ctx.guild.id]
            await ctx.voice_client.disconnect()
            await ctx.send("👋 Bye!")
        else:
            await ctx.send("❌ Not in a voice channel!")

    @commands.command()
    async def die(self, ctx):
        if ctx.voice_client:
            ctx.voice_client.stop()
        await ctx.send("💤 Shutting down...")
        await ctx.bot.close()

class PiperTTS:
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
            
        self.audio_buffer = []
        self.recording = False
        self.current_speaker = None
        self.output_dir = Path('recordings')
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger('VoiceSink')
        print("VoiceSink initialized")

    def wants_opus(self) -> bool:
        return False

    def cleanup(self):
        self.logger.info("VoiceSink cleanup")
        self.decode = False
        self.recording = False
        self.audio_buffer = []
        self.current_speaker = None

    def write(self, user, data: voice_recv.VoiceData):
        try:
            if user is None or data.pcm is None or not self.recording:
                return
            
            if user != self.current_speaker:
                return
                
            try:
                audio_np = np.frombuffer(data.pcm, dtype=np.int16)
                audio_stereo = audio_np.reshape(-1, 2)
                audio_mono = audio_stereo.mean(axis=1).astype(np.int16)
                
                self.audio_buffer.append(audio_mono.tobytes())
                
                max_amplitude = np.max(np.abs(audio_mono))
                if max_amplitude > 100:
                    self.logger.debug(f"Audio levels - Max amplitude: {max_amplitude}")
                    
            except Exception as e:
                self.logger.error(f"Error processing audio: {e}")
                import traceback
                traceback.print_exc()
                
        except Exception as e:
            self.logger.error(f"Error in write: {e}")
            import traceback
            traceback.print_exc()

    def save_and_process_audio(self):
        if not self.audio_buffer:
            return

        try:
            timestamp = int(time.time() * 1000)
            filename = self.output_dir / f"{self.current_speaker}_{timestamp}.wav"
            
            audio_data = b''.join(self.audio_buffer)
            
            with wave.open(str(filename), 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(48000)
                wav_file.writeframes(audio_data)
            
            self.logger.info(f"Saved audio: {filename}")
            
            try:
                if self.whisper_model:
                    # Use Faster-Whisper for transcription
                    segments, info = self.whisper_model.transcribe(str(filename))
                    text = " ".join([segment.text.strip() for segment in segments])
                    
                    if text and text.strip():
                        self.logger.info(f"Faster-Whisper recognized: {text}")
                        asyncio.run_coroutine_threadsafe(
                            self.cog.handle_text(self.current_speaker, text.strip()),
                            self.bot.loop
                        )
                    else:
                        self.logger.info("No speech detected by Faster-Whisper")
                else:
                    self.logger.warning("Whisper model not available")
                    
            except Exception as e:
                self.logger.error(f"Faster-Whisper error: {e}")
                # Could add Google Speech Recognition fallback here if needed
                
        except Exception as e:
            self.logger.error(f"Error saving/processing audio: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.audio_buffer = []

    @voice_recv.AudioSink.listener()
    def on_voice_member_speaking_start(self, member):
            self.logger.info(f"Speaking start: {member}")
            if not self.recording or self.current_speaker != member:
                self.recording = True
                self.current_speaker = member
                self.audio_buffer = []

    @voice_recv.AudioSink.listener()
    def on_voice_member_speaking_stop(self, member):
            self.logger.info(f"Speaking stop: {member}")
            if member == self.current_speaker:
                self.recording = False
                self.save_and_process_audio()
                self.current_speaker = None

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
    
    # Pre-warm Ollama model
    logger.info("🤖 Pre-warming Ollama model...")
    try:
        import asyncio
        async def warm_ollama():
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: ollama.chat(
                    model='hf.co/subsectmusic/qwriko3-4b-instruct-2507:Q4_K_M',
                    messages=[{'role': 'user', 'content': 'warmup'}],
                    options={'num_predict': 1}
                )
            )
        asyncio.run(warm_ollama())
        logger.info("✅ Ollama model warmed up")
    except Exception as e:
        logger.warning(f"⚠️ Ollama warmup failed: {e}")
    
    logger.info("✅ All models loaded! Starting Discord bot...")
    
    intents = discord.Intents.default()
    intents.message_content = True
    intents.voice_states = True
    intents.guilds = True

    bot = Bot(command_prefix='!', intents=intents, memory_store=memory_store)

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
        await bot.add_cog(Testing(bot, bot.piper))

    bot.setup_hook = setup_hook

    try:
        logger.info("Starting bot...")
        logger.info("Make sure:")
        logger.info("1. Piper files are in 'piper' directory")
        logger.info("2. 'recordings' and 'output' directories exist")
        logger.info("3. Ollama is running with: ollama run llama3.1:8b")
        logger.info("\nStarting bot now...")
        bot.run(DISCORD_BOT_TOKEN)
    except Exception as e:
        logger.error(f"Error running bot: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()