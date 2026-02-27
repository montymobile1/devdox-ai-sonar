FROM python:3.12-alpine

RUN apk add --no-cache \
    bash \
    git \
    gcc \
    musl-dev \
    linux-headers \
    build-base \
    rust \
    cargo

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ENVIRONMENT=production

# Install from local source
COPY pyproject.toml .
COPY src ./src

RUN pip install --no-cache-dir .

# Project mount point
WORKDIR /github/workspace


ENTRYPOINT ["python", "-m", "devdox_ai_sonar.cli"]