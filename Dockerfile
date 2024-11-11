FROM ghcr.io/astral-sh/uv:python3.12-bookworm

ENV UV_COMPILE_BYTECODE=1

RUN apt-get update && \
    apt-get install -y git && \
    apt-get clean

# RUN git clone https://github.com/vffuunnyy/ai_hack.git /app
COPY . /app
WORKDIR /app

RUN sh install.sh

CMD ["uv run", "train.py"]
