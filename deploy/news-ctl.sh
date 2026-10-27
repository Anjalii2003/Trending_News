#!/usr/bin/env bash
# News Engine control: start | stop | restart | status | logs [svc] | health | summary | once | build | down
set -euo pipefail
cd "$(dirname "$0")"
DC="docker compose -f docker-compose.yml"
cmd="${1:-status}"; shift || true
case "$cmd" in
  build)   $DC build ;;
  start)   $DC up -d --build ;;
  stop)    $DC stop ;;                         # graceful: the running cycle finishes (up to 5 min), state is kept
  restart) $DC restart "$@" ;;
  down)    $DC down ;;                         # removes containers, KEEPS the data volume
  status)  $DC ps; echo; docker inspect --format '{{.Name}}  health={{.State.Health.Status}}  restarts={{.RestartCount}}' news-ingest news-dashboard 2>/dev/null || true ;;
  logs)    $DC logs --tail=100 -f "${@:-news-ingest}" ;;
  health)  $DC exec -T news-ingest python scripts/news_service.py health; curl -fsS "http://127.0.0.1:${NEWS_DASHBOARD_PORT:-8880}/api/news/health"; echo ;;
  summary) curl -fsS "http://127.0.0.1:${NEWS_DASHBOARD_PORT:-8880}/api/news/live" | python3 -m json.tool ;;
  once)    $DC exec news-ingest python scripts/news_service.py once ;;   # run one extra cycle now (refused while one is running)
  *) echo "usage: $0 {build|start|stop|restart [svc]|down|status|logs [svc]|health|summary|once}"; exit 2 ;;
esac
