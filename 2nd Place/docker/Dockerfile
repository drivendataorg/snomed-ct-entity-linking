FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04
ENV TZ=Asia/Yerevan
ARG PDM_VERSION=2.12.4
ARG DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/.local/bin:${PATH}"

RUN apt-get update && \
    apt-get install -y --option=Dpkg::Options::=--force-confdef \
        make \
        g++ \
        curl \
        python3.10-venv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://raw.githubusercontent.com/pdm-project/pdm/main/install-pdm.py | python3 - -v $PDM_VERSION && \
    echo "export PATH=/root/.local/bin:${PATH}" >> $HOME/.profile
