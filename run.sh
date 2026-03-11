#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

# Build frontend
echo "Building frontend..."
cd app/frontend
npm install --silent
npm run build
cd ../..

# Start server
echo "Starting server on http://localhost:8000"
uv run uvicorn app.backend.main:app --port 8000
