# News Engine: one image, two roles (ingest service / dashboard) selected by the compose command.
FROM python:3.12-slim

ARG NEWS_GPU=0
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 TZ=UTC

WORKDIR /app
COPY deploy/requirements-news.txt /tmp/requirements-news.txt
COPY deploy/requirements-news-gpu.txt /tmp/requirements-news-gpu.txt
RUN pip install -r /tmp/requirements-news.txt \
    && if [ "$NEWS_GPU" = "1" ]; then pip install -r /tmp/requirements-news-gpu.txt; fi

COPY scripts/lib /app/scripts/lib
COPY scripts/news_service.py /app/scripts/news_service.py
COPY scripts/20_news_run.py /app/scripts/20_news_run.py
COPY UI/news_api.py UI/news_server.py UI/trending.html UI/news.html /app/UI/
COPY UI/assets /app/UI/assets
# geography (Natural Earth, offline), publisher registry and taxonomy are mounted read-only by compose (see docker-compose.yml)

RUN useradd --system --uid 10001 --home /state engine && mkdir -p /state /app/data && chown -R engine /state
USER engine
ENV NEWS_STATE_DIR=/state
VOLUME ["/state"]
CMD ["python", "scripts/news_service.py", "run"]
