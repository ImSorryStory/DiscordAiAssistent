FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# -- Системные зависимости --
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    ca-certificates \
    espeak-ng-data \
    libsodium23 \
    libopus0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "py-cord[voice]" \
    yt-dlp \
    requests \
    beautifulsoup4 \
    python-dotenv \
    faster-whisper \
    piper-tts

# -- Директории для моделей --
RUN mkdir -p /whisper/faster-whisper-large-v3 \
    && mkdir -p /voices/ru_RU

# -- Whisper large-v3 (многоязычная) --
WORKDIR /whisper/faster-whisper-large-v3
RUN set -eux; \
    curl -fL --retry 3 -o model.bin       https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/model.bin       && test -s model.bin; \
    curl -fL --retry 3 -o config.json     https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/config.json     && test -s config.json; \
    curl -fL --retry 3 -o tokenizer.json  https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/tokenizer.json  && test -s tokenizer.json; \
    curl -fL --retry 3 -o vocabulary.json https://huggingface.co/Systran/faster-whisper-large-v3/resolve/main/vocabulary.json && test -s vocabulary.json

# -- Piper RU голос (Denis, medium) + .json --
WORKDIR /voices/ru_RU
RUN set -eux; \
    curl -fL --retry 3 -o ru_RU-denis-medium.onnx \
      https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/denis/medium/ru_RU-denis-medium.onnx \
      && test -s ru_RU-denis-medium.onnx; \
    curl -fL --retry 3 -o ru_RU-denis-medium.onnx.json \
      https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/denis/medium/ru_RU-denis-medium.onnx.json \
      && test -s ru_RU-denis-medium.onnx.json

# -- Приложение --
WORKDIR /app
COPY . .

# -- ENV для кода бота --
ENV WHISPER_MODEL=/whisper/faster-whisper-large-v3 \
    PIPER_VOICE=/voices/ru_RU/ru_RU-denis-medium.onnx \
    ESPEAKNG_DATA=/usr/share/espeak-ng-data \
    ASR_LANGUAGE=ru

# -- Запуск --
CMD ["python", "-u", "bot_local_qwen.py"]
