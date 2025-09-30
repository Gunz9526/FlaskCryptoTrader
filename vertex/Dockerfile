FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    git \
    cmake \
    && rm -rf /var/lib/apt/lists/*

RUN cd /tmp \
    && git clone --branch v0.6.4 https://github.com/TA-Lib/ta-lib.git \
    && cd ta-lib \
    && mkdir build && cd build \
    && cmake .. \
    && make \
    && make install \
    && rm -rf /tmp/ta-lib

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129 && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]

CMD ["--tune_mode", "deep"]