ARG TARGETPLATFORM

FROM --platform=$TARGETPLATFORM python:3.13-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y fonts-nanum* && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /tmp

RUN wget https://github.com/ta-lib/ta-lib/releases/download/v0.6.4/ta-lib-0.6.4-src.tar.gz && \
    tar -xvzf ta-lib-0.6.4-src.tar.gz && \
    cd ta-lib-0.6.4/ && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd / && rm -rf /tmp/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd --create-home --shell /bin/bash appuser
RUN mkdir -p /var/run/celery && chown -R appuser:appuser /var/run/celery

RUN mkdir -p /app/logs /app/migrations
RUN chown -R appuser:appuser /app
USER appuser

ENV FLASK_APP=run.py
ENV FLASK_ENV=production
ENV PYTHONPATH=/app

EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "120", "run:app"]