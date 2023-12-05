FROM mambaorg/micromamba:1.5.1

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    openssh-client \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*
USER $MAMBA_USER
