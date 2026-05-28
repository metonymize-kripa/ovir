#!/usr/bin/env bash
# start.sh — Boot the full onet-falkor-playground stack.
#
# Usage:
#   ./start.sh          # normal start (skips loader if graph already populated)
#   ./start.sh --fresh  # drop + reload the graph from scratch
#
# Prerequisites: docker, uv, node >= 18
# Runs from: onet-falkor-playground/

set -euo pipefail

FRESH=${1:-}
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HERE="$(cd "$(dirname "$0")" && pwd)"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${CYAN}[start]${NC} $*"; }
success() { echo -e "${GREEN}[start]${NC} $*"; }
warn()    { echo -e "${YELLOW}[start]${NC} $*"; }
die()     { echo -e "${RED}[start]${NC} $*" >&2; exit 1; }

# ─── 0. Preflight ─────────────────────────────────────────────────────────────
command -v docker &>/dev/null || die "docker not found"
command -v uv     &>/dev/null || die "uv not found (https://github.com/astral-sh/uv)"
command -v node   &>/dev/null || die "node not found"
command -v npm    &>/dev/null || die "npm not found"

# ─── 1. FalkorDB ──────────────────────────────────────────────────────────────
info "Starting FalkorDB..."
(cd "$REPO_ROOT" && docker compose up falkordb -d --quiet-pull)

info "Waiting for FalkorDB to be ready..."
for i in $(seq 1 30); do
  if docker compose -f "$REPO_ROOT/docker-compose.yml" exec -T falkordb redis-cli ping 2>/dev/null | grep -q PONG; then
    success "FalkorDB ready."
    break
  fi
  if [ "$i" -eq 30 ]; then
    die "FalkorDB did not respond after 30s. Check: docker compose logs falkordb"
  fi
  sleep 1
done

# ─── 2. O*NET Loader ──────────────────────────────────────────────────────────
LOADER_DIR="$HERE/loader"
GRAPH_POPULATED=false

if [ "$FRESH" != "--fresh" ]; then
  # Check if graph already has occupations loaded
  OCC_COUNT=$(docker compose -f "$REPO_ROOT/docker-compose.yml" exec -T falkordb \
    redis-cli GRAPH.QUERY onet "MATCH (o:Occupation) RETURN count(o)" 2>/dev/null \
    | grep -E '^[0-9]+$' | head -1 || echo "0")
  if [ "${OCC_COUNT:-0}" -gt 100 ] 2>/dev/null; then
    success "Graph already populated ($OCC_COUNT occupations). Skipping loader. Pass --fresh to reload."
    GRAPH_POPULATED=true
  fi
fi

if [ "$GRAPH_POPULATED" = false ]; then
  info "Loading O*NET data into FalkorDB (this takes 2–4 min)..."
  LOADER_ARGS="--data-dir $REPO_ROOT/data/db_30_0_text"
  [ "$FRESH" = "--fresh" ] && LOADER_ARGS="$LOADER_ARGS --fresh"
  (cd "$LOADER_DIR" && uv run load_onet.py $LOADER_ARGS)
  success "Graph loaded."
fi

# ─── 3. Backend ───────────────────────────────────────────────────────────────
BACKEND_DIR="$HERE/backend"
info "Starting FastAPI backend on :8000..."
(cd "$BACKEND_DIR" && uv run uvicorn main:app --port 8000 --log-level warning) &
BACKEND_PID=$!

# Wait for backend to accept connections
for i in $(seq 1 20); do
  if curl -sf http://localhost:8000/health &>/dev/null; then
    success "Backend ready at http://localhost:8000"
    break
  fi
  if [ "$i" -eq 20 ]; then
    die "Backend did not come up. Check output above."
  fi
  sleep 1
done

# ─── 4. Frontend ──────────────────────────────────────────────────────────────
FRONTEND_DIR="$HERE/frontend"
if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
  info "Installing frontend dependencies..."
  (cd "$FRONTEND_DIR" && npm install --silent)
  success "Dependencies installed."
fi

info "Starting Next.js frontend on :3001..."
(cd "$FRONTEND_DIR" && npm run dev) &
FRONTEND_PID=$!

# ─── 5. Open browser ──────────────────────────────────────────────────────────
sleep 3
success "Stack is up:"
echo -e "  ${GREEN}Frontend${NC}        http://localhost:3001"
echo -e "  ${GREEN}Backend API${NC}     http://localhost:8000/docs"
echo -e "  ${GREEN}FalkorDB UI${NC}     http://localhost:3000"
echo ""

# Try to open browser (mac/linux)
if command -v open &>/dev/null; then
  open http://localhost:3001
elif command -v xdg-open &>/dev/null; then
  xdg-open http://localhost:3001
fi

# ─── 6. Cleanup on exit ───────────────────────────────────────────────────────
cleanup() {
  echo ""
  info "Shutting down backend and frontend..."
  kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
  wait "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
  info "Done. FalkorDB container left running — stop it with: docker compose stop falkordb"
}
trap cleanup EXIT INT TERM

# Block until Ctrl-C
wait
