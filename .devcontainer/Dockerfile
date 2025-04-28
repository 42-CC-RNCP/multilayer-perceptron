FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    git curl sudo zsh build-essential libgl1-mesa-glx tree \
    && apt-get clean

RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"
