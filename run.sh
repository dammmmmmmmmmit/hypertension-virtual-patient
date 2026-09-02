#!/usr/bin/env bash
# Starts the full Virtual Patient Drug-Response Simulator stack:
# Postgres/Redis/Qdrant (Docker) -> Ollama -> FastAPI backend -> Next.js
# frontend. Safe to re-run — skips anything already up. Creates the
# Docker containers on first run only (see PROJECT_CONTEXT.md §11 for
# the exact commands this mirrors). Does NOT do first-time data/model
# setup (schema init, ChEMBL/PubChem/SIDER ingestion, model training) —
# that's a slow, external-API-dependent one-time step, kept separate on
# purpose so this script stays fast and safe to run repeatedly.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# Frontend pinned to 3001, not 3000 — this machine permanently runs an
# unrelated project's Docker container (acs-ai-dev-langgraph-api-1) bound
# to host port 3000. A plain "does something respond on :3000" check
# false-positives against that container (it answers real HTTP requests
# with its own JSON), which silently made this script think the frontend
# was already up when it was actually hitting the wrong app entirely —
# found live, not hypothetical. is_content_up() below guards against the
# same mistake happening again on whatever port ends up in use.
FRONTEND_PORT=3001

is_up() { curl -s -o /dev/null --max-time 2 "$1"; }
# Verifies the response body actually contains something unique to THIS
# app, not just that some server answered — see the port-3000 note above.
is_content_up() { curl -s --max-time 2 "$1" 2>/dev/null | grep -q "$2"; }

echo "==> Docker containers (Postgres :5433, Redis :6379, Qdrant :6333)"
for c in vps-pg vps-redis vps-qdrant; do
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$c"; then
    echo "    $c: already up"
  elif docker ps -a --format '{{.Names}}' 2>/dev/null | grep -qx "$c"; then
    docker start "$c" >/dev/null && echo "    $c: started"
  else
    case "$c" in
      vps-pg)     docker run -d -p 5433:5432 -e POSTGRES_PASSWORD=postgres --name vps-pg postgres:16 >/dev/null ;;
      vps-redis)  docker run -d -p 6379:6379 --name vps-redis redis:7 >/dev/null ;;
      vps-qdrant) docker run -d -p 6333:6333 --name vps-qdrant qdrant/qdrant >/dev/null ;;
    esac
    echo "    $c: created + started (first run)"
  fi
done

echo "==> Ollama"
if pgrep -f "ollama serve" >/dev/null; then
  echo "    already running"
else
  nohup ollama serve >/tmp/vps-ollama.log 2>&1 &
  disown
  sleep 2
  echo "    started (log: /tmp/vps-ollama.log)"
fi

echo "==> Backend (FastAPI, :8000)"
if is_up http://localhost:8000/docs; then
  echo "    already running"
else
  if [ ! -f .env ]; then
    cp .env.example .env
    echo "    created .env from .env.example — edit DATABASE_URL/etc. if needed"
  fi
  nohup uv run uvicorn app.api.main:app --port 8000 >/tmp/vps-backend.log 2>&1 &
  disown
  echo "    starting... (log: /tmp/vps-backend.log)"
fi

echo "==> Frontend (Next.js, :$FRONTEND_PORT)"
if is_content_up "http://localhost:$FRONTEND_PORT" "Virtual Patient Drug-Response Simulator"; then
  echo "    already running"
else
  if [ ! -d frontend/node_modules ]; then
    echo "    node_modules missing, running npm install first (one-time)..."
    (cd frontend && npm install)
  fi
  (cd frontend && nohup npm run dev >/tmp/vps-frontend.log 2>&1 & disown)
  echo "    starting... (log: /tmp/vps-frontend.log)"
fi

echo ""
echo "==> Waiting for backend + frontend to come up (up to 40s)..."
back_ok=0
front_ok=0
for _ in $(seq 1 40); do
  is_up http://localhost:8000/docs && back_ok=1
  is_content_up "http://localhost:$FRONTEND_PORT" "Virtual Patient Drug-Response Simulator" && front_ok=1
  [ "$back_ok" = 1 ] && [ "$front_ok" = 1 ] && break
  sleep 1
done

echo ""
if [ "$back_ok" = 1 ]; then echo "Backend:  http://localhost:8000/docs        [OK]"
else echo "Backend:  http://localhost:8000/docs        [NOT UP YET — check /tmp/vps-backend.log]"; fi
if [ "$front_ok" = 1 ]; then echo "Frontend: http://localhost:$FRONTEND_PORT        [OK]"
else echo "Frontend: http://localhost:$FRONTEND_PORT        [NOT UP YET — check /tmp/vps-frontend.log]"; fi
echo ""
echo "Run ./stop.sh to stop the backend + frontend (Docker containers and Ollama are left running)."
