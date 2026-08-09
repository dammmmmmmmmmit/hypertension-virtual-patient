#!/usr/bin/env bash
# Stops the backend + frontend dev servers started by run.sh. Leaves
# Docker containers and Ollama running (cheap to keep warm; stop them
# manually with `docker stop vps-pg vps-redis vps-qdrant` / `pkill ollama`
# if you actually want everything down).
set -uo pipefail

echo "==> Stopping backend (uvicorn)"
pkill -f "uvicorn app.api.main" && echo "    stopped" || echo "    was not running"

echo "==> Stopping frontend (next dev)"
pkill -f "next dev" && echo "    stopped" || echo "    was not running"
