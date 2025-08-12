# bot_local_qwen.py
# Локальный Discord voice-ассистент "Кентик" (Python 3.12.8)
# Функции: приветствие, проверка слышимости, горячее слово "Кентик",
# музыка (yt_dlp), веб-поиск + краткое резюме (Ollama), TTS (Piper), ASR (faster-whisper)
# Реалтайм-логика: короткие окна записи (~1.2с) без гонок, русское ASR large-v3.
# Исправления: русский ASR, проверка наличия .onnx.json, корректные коллбеки записи,
# одна запись за раз (нет "Already recording"), стабильный цикл слушания, гарантия подключения к голосовому.

import os
import io
import re
import asyncio
import tempfile
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Deque, Dict, List

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

import discord
from discord.ext import commands
from discord.sinks import WaveSink
import yt_dlp

from faster_whisper import WhisperModel

# ========= ENV =========
load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

# ASR (Whisper)
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "/whisper/faster-whisper-large-v3")
ASR_LANGUAGE = (os.getenv("ASR_LANGUAGE", "ru") or "ru").strip().lower()
_whisper_model: Optional[WhisperModel] = None
_asr_warmed: bool = False

# TTS (Piper)
PIPER_VOICE = os.getenv("PIPER_VOICE")  # путь к .onnx
ESPEAKNG_DATA = os.getenv("ESPEAKNG_DATA")  # путь к espeak-ng-data (можно не задавать, если системный)

# LLM (Ollama)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")  # без /api
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "120"))

# ========= Discord intents =========
INTENTS = discord.Intents.default()
INTENTS.message_content = True   # включи Message Content Intent в Dev Portal
INTENTS.voice_states = True

bot = commands.Bot(command_prefix="!", intents=INTENTS)

# ========= Музыка =========
@dataclass
class Track:
    query: str
    requested_by: Optional[str] = None
    title: Optional[str] = None

@dataclass
class MusicState:
    queue: Deque[Track] = field(default_factory=deque)
    now_playing: Optional[Track] = None

guild_music: Dict[int, MusicState] = {}
assistant_running: Dict[int, bool] = {}

def get_state(guild_id: int) -> MusicState:
    if guild_id not in guild_music:
        guild_music[guild_id] = MusicState()
    return guild_music[guild_id]

YTDLP_OPTS = {
    "format": "bestaudio/best",
    "noplaylist": True,
    "quiet": True,
    "extract_flat": False,
    "default_search": "ytsearch",
    "source_address": "0.0.0.0",
}

def ytdlp_get_source(query: str) -> dict:
    with yt_dlp.YoutubeDL(YTDLP_OPTS) as ydl:
        info = ydl.extract_info(query, download=False)
        if "entries" in info:
            info = info["entries"][0]
        return {"url": info["url"], "title": info.get("title", query)}

async def play_next(ctx: commands.Context):
    state = get_state(ctx.guild.id)
    if not state.queue:
        state.now_playing = None
        return
    track = state.queue.popleft()
    src = ytdlp_get_source(track.query)
    track.title = src.get("title") or track.query
    state.now_playing = track

    vc = ctx.voice_client

    def after_play(_):
        fut = asyncio.run_coroutine_threadsafe(play_next(ctx), bot.loop)
        try:
            fut.result()
        except Exception:
            pass

    ffmpeg_opts = "-nostdin -reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5"
    vc.play(discord.FFmpegPCMAudio(src["url"], before_options=ffmpeg_opts), after=after_play)
    await ctx.send(f"▶️ **Играет:** {track.title}")

# ========= Вспомогательные: Voice connect, TTS, ASR, web, LLM =========
async def ensure_voice_connected(ctx: commands.Context) -> Optional[discord.VoiceClient]:
    if ctx.author.voice is None or ctx.author.voice.channel is None:
        await ctx.send("Зайди в голосовой канал, потом набери команду.")
        return None

    vc = ctx.voice_client
    try:
        if vc is None:
            vc = await ctx.author.voice.channel.connect(reconnect=True)
        else:
            if vc.channel != ctx.author.voice.channel:
                await vc.move_to(ctx.author.voice.channel)
    except Exception as e:
        await ctx.send(f"Не удалось подключиться к голосовому каналу: {e}")
        return None

    # ждём полной готовности соединения (до ~6 сек)
    for _ in range(30):
        if vc and vc.is_connected():
            break
        await asyncio.sleep(0.2)

    if not vc or not vc.is_connected():
        await ctx.send("Голосовое подключение не установлено. Попробуй ещё раз (!join).")
        return None

    return vc


async def speak(vc: discord.VoiceClient, text: str):
    """Озвучить текст через Piper и проиграть в голосовом канале."""
    if vc is None or not vc.is_connected():
        raise RuntimeError("Нет голосового подключения для воспроизведения TTS.")

    if not PIPER_VOICE or not os.path.exists(PIPER_VOICE):
        raise RuntimeError(f"PIPER_VOICE не найден: {PIPER_VOICE}")
    cfg = PIPER_VOICE + ".json"
    if not os.path.exists(cfg):
        raise RuntimeError(f"Отсутствует конфиг модели Piper: {cfg}")

    out_path = tempfile.mktemp(suffix=".wav")

    cmd = ["piper", "--model", PIPER_VOICE, "--output_file", out_path]
    if ESPEAKNG_DATA and os.path.isdir(ESPEAKNG_DATA):
        cmd += ["--espeak-ng-data", ESPEAKNG_DATA]

    # генерируем речь
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, err = await proc.communicate(input=text.encode("utf-8"))
    if proc.returncode != 0 or not os.path.exists(out_path):
        raise RuntimeError(f"Piper error: {err.decode(errors='ignore')}")

    # ждём, пока освободится плеер, и проигрываем
    finished = asyncio.Event()
    def _after(_):
        bot.loop.call_soon_threadsafe(finished.set)

    vc.play(discord.FFmpegPCMAudio(out_path), after=_after)
    await finished.wait()

async def _remux_to_clean_wav(raw_bytes: bytes, rate=16000, ch=1) -> bytes:
    """Нормализуем входное аудио в валидный WAV PCM s16le через ffmpeg."""
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", "pipe:0",
        "-f", "wav",
        "-ar", str(rate),
        "-ac", str(ch),
        "-acodec", "pcm_s16le",
        "pipe:1",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate(input=raw_bytes)
    if proc.returncode != 0 or not out:
        raise RuntimeError(err.decode("utf-8", "ignore") or "ffmpeg remux failed")
    return out

def _ensure_asr_loaded():
    global _whisper_model, _asr_warmed
    if _whisper_model is None:
        # CPU-friendly: int8; если есть GPU, можно сменить на "float16"
        _whisper_model = WhisperModel(WHISPER_MODEL_NAME, device="auto", compute_type="int8")

    # Прогрев модели один раз (убирает первый «фриз»)
    if not _asr_warmed:
        import numpy as np
        import wave
        # 0.5 c тишины @16kHz mono
        tmp = tempfile.mktemp(suffix=".wav")
        with wave.open(tmp, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # s16le
            wf.setframerate(16000)
            wf.writeframes(np.zeros(16000//2, dtype=np.int16).tobytes())
        try:
            list(_whisper_model.transcribe(
                tmp,
                language=ASR_LANGUAGE,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 200, "speech_pad_ms": 200},
                beam_size=1,
                condition_on_previous_text=False,
                temperature=0.0,
            )[0])
        except Exception:
            pass
        _asr_warmed = True

async def asr_transcribe(wav_bytes: bytes) -> str:
    """Распознавание через faster-whisper с безопасной нормализацией WAV, язык — русский."""
    if len(wav_bytes) < 3000:
        return ""

    _ensure_asr_loaded()

    try:
        clean = await _remux_to_clean_wav(wav_bytes, 16000, 1)
    except Exception:
        return ""

    tmp = tempfile.mktemp(suffix=".wav")
    with open(tmp, "wb") as f:
        f.write(clean)

    segments, _ = _whisper_model.transcribe(
        tmp,
        language=ASR_LANGUAGE,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 200, "speech_pad_ms": 200},
        beam_size=1,  # быстрее, почти realtime-режим
        condition_on_previous_text=False,
        temperature=0.0,
        suppress_tokens="-1",
    )
    return " ".join(s.text.strip() for s in segments).strip()

def ddg_search(query: str, n: int = 3) -> List[dict]:
    url = "https://duckduckgo.com/html/"
    r = requests.post(url, data={"q": query}, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    out = []
    for res in soup.select(".result")[:n]:
        a = res.select_one(".result__a")
        if not a:
            continue
        href = a.get("href")
        title = a.get_text(" ", strip=True)
        sn = res.select_one(".result__snippet")
        out.append({"title": title, "url": href, "snippet": sn.get_text(" ", strip=True) if sn else ""})
    return out

def fetch_and_skim(url: str, max_chars: int = 2500) -> str:
    try:
        html = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"}).text
        soup = BeautifulSoup(html, "html.parser")
        for t in soup(["script", "style", "noscript"]):
            t.decompose()
        text = " ".join(soup.get_text(" ").split())
        return text[:max_chars]
    except Exception:
        return ""

def ollama_generate(prompt: str, system: str = "Отвечай кратко и по-русски.", temperature: float = 0.3) -> str:
    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": f"{system}\n\n{prompt}".strip(),
                "stream": False,
                "options": {"temperature": temperature},
            },
            timeout=OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        return f"(LLM ошибка: {e})"

def summarize_with_llm(text: str, limit: int = 700) -> str:
    prompt = (
        "Сделай фактическое резюме на русском, без воды.\n"
        f"До {limit} символов. Сохраняй даты/цифры.\n\n{text[:6000]}"
    )
    return ollama_generate(prompt)

# ========= Запись аудио (утилиты) =========
async def start_recording(vc: discord.VoiceClient) -> WaveSink:
    """Безопасный старт записи: если уже идёт запись — останавливаем и ждём сброса флага."""
    if vc is None or not vc.is_connected():
        raise RuntimeError("VoiceClient не подключён к голосовому каналу.")

    # Если по какой-то причине ещё «записывает» — мягко останавливаем и ждём.
    if getattr(vc, "recording", False):
        try:
            vc.stop_recording()
        except Exception:
            pass
        for _ in range(20):
            if not getattr(vc, "recording", False):
                break
            await asyncio.sleep(0.05)

    sink = WaveSink()

    # ВАЖНО: коллбек — корутина
    async def on_stopped(_sink: WaveSink, *_):
        return

    try:
        vc.start_recording(sink, on_stopped)
    except discord.sinks.errors.RecordingException:
        # Ещё раз страховочно стопаем и пробуем повторно
        try:
            vc.stop_recording()
        except Exception:
            pass
        await asyncio.sleep(0.1)
        vc.start_recording(sink, on_stopped)

    return sink


async def record_for(vc: discord.VoiceClient, seconds: float) -> Dict[discord.Member, io.BytesIO]:
    """Записать голос на N секунд и вернуть сырые WAV-буферы по пользователям."""
    sink = await start_recording(vc)
    try:
        await asyncio.sleep(seconds)
    finally:
        try:
            vc.stop_recording()
        except Exception:
            pass
        await asyncio.sleep(0.05)

    outputs: Dict[discord.Member, io.BytesIO] = {}
    for user, audio in list(sink.audio_data.items()):
        if getattr(audio, "file", None):
            outputs[user] = io.BytesIO(audio.file.getvalue())
    return outputs

# ========= NLU (ключевое слово + простые правила) =========
KEYWORDS = ["кент", "кентик", "кентек", "кентюк"]

def extract_after_keyword(text: str) -> str:
    """Возвращает текст после слова-триггера."""
    t = text.lower()
    for kw in KEYWORDS:
        idx = t.find(kw)
        if idx != -1:
            return text[idx + len(kw):].strip(" ,.:;—-")
    return ""

def detect_and_handle_intent(text: str) -> dict:
    """Определяет простые интенты, ожидая, что text уже содержит ключевое слово."""
    low = text.lower()
    if "включи" in low or "поставь" in low or "сыграй" in low:
        q = extract_after_keyword(text)
        return {"name": "play_music", "query": q}
    if any(w in low for w in ["пауза", "останови"]):
        return {"name": "pause"}
    if any(w in low for w in ["продолжай", "продолжить", "resume"]):
        return {"name": "resume"}
    if any(w in low for w in ["следующая", "скип", "пропусти", "next"]):
        return {"name": "skip"}
    if low.startswith("что такое") or "найди" in low:
        q = extract_after_keyword(text) or (low.split("найди", 1)[-1].strip() if "найди" in low else "")
        return {"name": "web_search", "query": q}
    if any(w in low for w in ["скажи", "озвучь", "привет"]):
        msg = extract_after_keyword(text) or "Привет! Я на связи."
        return {"name": "say", "text": msg}
    return {"name": "smalltalk", "text": extract_after_keyword(text) or text}

# ========= Основной цикл ассистента =========
async def assistant_loop(ctx: commands.Context):
    guild_id = ctx.guild.id
    if assistant_running.get(guild_id):
        await ctx.send("Ассистент уже запущен.")
        return
    assistant_running[guild_id] = True

    # гарантируем голосовое подключение
    vc = await ensure_voice_connected(ctx)
    if vc is None:
        assistant_running[guild_id] = False
        return

    # 1) Приветствие
    try:
        await speak(vc, "Всем привет!")
    except Exception as e:
        await ctx.send(f"(TTS приветствия не удалось: {e})")

    # 2) Быстрая проверка слышимости (2 сек)
    try:
        await ctx.send("🎧 Проверяю, слышу ли я...")
        samples = await record_for(vc, 2.0)
        heard: List[str] = []
        for user, bio in samples.items():
            text = await asr_transcribe(bio.getvalue())
            if text:
                heard.append(f"{user.display_name}: {text}")
        if heard:
            await ctx.send("Слышу:\n- " + "\n- ".join(heard[:5]))
        else:
            await ctx.send("Похоже, никто ничего не сказал, но продолжу слушать.")
    except Exception as e:
        await ctx.send(f"(Проверка слышимости не удалась: {e})")

    await ctx.send("🎙 Слушаю. Говорите **«Кентик ...»** и затем команду.")

    # 3) «Почти realtime»: короткие окна записи ~1.2с
    try:
        while assistant_running.get(guild_id) and vc and vc.is_connected():
            # Не слушаем, когда бот сам что-то озвучивает, чтобы не ловить собственный голос
            if vc.is_playing():
                await asyncio.sleep(0.2)
                continue

            samples = await record_for(vc, 1.2)  # окно 1.0–1.5с даёт быструю реакцию
            for user, bio in samples.items():
                wav_bytes = bio.getvalue()
                if len(wav_bytes) < 3000:
                    continue

                text = await asr_transcribe(wav_bytes)
                if not text:
                    continue

                await ctx.send(f"🗣 **{user.display_name}:** {text}")

                # ключевое слово
                low = text.lower()
                if not any(kw in low for kw in KEYWORDS):
                    continue

                intent = detect_and_handle_intent(text)
                name = intent.get("name")

                try:
                    if name == "play_music":
                        q = intent.get("query") or ""
                        get_state(guild_id).queue.append(Track(query=q, requested_by=user.display_name))
                        await ctx.send(f"Добавил в очередь: **{q}**")
                        if not vc.is_playing() and not vc.is_paused():
                            await play_next(ctx)

                    elif name == "pause":
                        if vc.is_playing():
                            vc.pause(); await ctx.send("⏸ Пауза")

                    elif name == "resume":
                        if vc.is_paused():
                            vc.resume(); await ctx.send("▶️ Продолжил")

                    elif name == "skip":
                        if vc.is_playing() or vc.is_paused():
                            vc.stop(); await ctx.send("⏭ Пропустил")

                    elif name == "web_search":
                        q = (intent.get("query") or "").strip()
                        if not q:
                            await ctx.send("Что именно ищем?")
                            continue
                        await ctx.send(f"🔎 Ищу: **{q}**")
                        results = ddg_search(q, n=3)
                        if not results:
                            await ctx.send("Ничего не нашёл.")
                            continue
                        blocks = []
                        for r in results:
                            skim = fetch_and_skim(r["url"], max_chars=2000) if r["url"] else r["snippet"]
                            blocks.append(f"{r['title']}\n{r['url']}\n{skim}")
                        summary = summarize_with_llm("\n\n---\n\n".join(blocks), limit=700)
                        await ctx.send(summary[:1900])
                        try:
                            await speak(vc, summary[:300])
                        except Exception:
                            pass

                    elif name == "say":
                        msg = intent.get("text") or "Да?"
                        await speak(vc, msg)

                    else:  # smalltalk
                        reply = ollama_generate(
                            f"Ключевое слово уже произнесено. Пользователь сказал: {intent.get('text','')}. Ответь коротко и по делу."
                        )
                        await ctx.send(reply[:1900] if reply else intent.get("text",""))
                        try:
                            await speak(vc, (reply or "Хорошо.")[:300])
                        except Exception:
                            pass

                except Exception as e:
                    await ctx.send(f"(Ошибка выполнения: {e})")

            await asyncio.sleep(0.02)

    finally:
        try:
            if ctx.voice_client:
                await stop_recording(ctx.voice_client)
        except Exception:
            pass
        assistant_running[guild_id] = False
        await ctx.send("Ассистент остановлен.")

# ========= Команды =========
@bot.command(name="join")
async def join(ctx: commands.Context):
    vc = await ensure_voice_connected(ctx)
    if vc is None:
        return
    await ctx.send(f"Подключился к: **{vc.channel.name}**. Запускаю ассистента…")
    await assistant_loop(ctx)

@bot.command(name="leave")
async def leave(ctx: commands.Context):
    assistant_running[ctx.guild.id] = False
    if ctx.voice_client:
        await stop_recording(ctx.voice_client)
        await ctx.voice_client.disconnect(force=True)
    await ctx.send("Отключился.")

@bot.command(name="assistant")
async def assistant(ctx: commands.Context):
    vc = await ensure_voice_connected(ctx)
    if vc is None:
        return
    await assistant_loop(ctx)

@bot.command(name="play")
async def play(ctx: commands.Context, *, query: str):
    if ctx.voice_client is None or not ctx.voice_client.is_connected():
        await ctx.send("Сначала !join.")
        return
    get_state(ctx.guild.id).queue.append(Track(query=query, requested_by=str(ctx.author)))
    await ctx.send(f"Добавил в очередь: **{query}**")
    if not ctx.voice_client.is_playing() and not ctx.voice_client.is_paused():
        await play_next(ctx)

@bot.command(name="pause")
async def pause(ctx: commands.Context):
    vc = ctx.voice_client
    if vc and vc.is_playing():
        vc.pause(); await ctx.send("⏸ Пауза")

@bot.command(name="resume")
async def resume(ctx: commands.Context):
    vc = ctx.voice_client
    if vc and vc.is_paused():
        vc.resume(); await ctx.send("▶️ Продолжил")

@bot.command(name="skip")
async def skip(ctx: commands.Context):
    vc = ctx.voice_client
    if vc and (vc.is_playing() or vc.is_paused()):
        vc.stop(); await ctx.send("⏭ Пропустил")

@bot.command()
async def recordtest(ctx):
    """Записывает 5 секунд из голосового, сохраняет WAV и отправляет в чат."""
    if ctx.author.voice is None:
        await ctx.send("Зайди в голосовой канал.")
        return

    vc = ctx.voice_client
    if vc is None:
        vc = await ctx.author.voice.channel.connect()

    sink = discord.sinks.WaveSink()

    async def finished_callback(sink: WaveSink, *args):
        for user, audio in sink.audio_data.items():
            wav_path = f"/tmp/{user}.wav"
            with open(wav_path, "wb") as f:
                f.write(audio.file.getvalue())
            await ctx.send(f"Сохранил {user}.wav", file=discord.File(wav_path))
        # НЕ отключаемся автоматически

    vc.start_recording(sink, finished_callback)  # coroutine callback — ок
    await ctx.send("Запись 5 секунд... Говори что-нибудь!")
    await asyncio.sleep(5)
    vc.stop_recording()

@bot.command(name="say")
async def say(ctx: commands.Context, *, text: str):
    vc = await ensure_voice_connected(ctx)
    if vc is None:
        return
    try:
        await speak(vc, text)
        await ctx.send("(Озвучил)")
    except Exception as e:
        await ctx.send(f"(TTS ошибка: {e})")

@bot.event
async def on_ready():
    try:
        requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3).raise_for_status()
        print(f"Ollama OK ({OLLAMA_MODEL}).")
    except Exception:
        print("⚠️ Ollama недоступен. Запусти `ollama serve` или проверь OLLAMA_HOST.")
    print(f"Logged in as {bot.user} (id={bot.user.id})")

if __name__ == "__main__":
    if not DISCORD_TOKEN:
        raise SystemExit("Нет DISCORD_TOKEN в .env")
    bot.run(DISCORD_TOKEN)
