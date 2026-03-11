# Stage 1: Build frontend
FROM node:22-slim AS frontend-build
WORKDIR /build
COPY app/frontend/package.json app/frontend/package-lock.json ./
RUN npm ci
COPY app/frontend/ .
RUN npm run build

# Stage 2: Python runtime
FROM python:3.12-slim AS runtime

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install Python dependencies (CPU-only PyTorch)
COPY pyproject.toml uv.lock ./
COPY high_society/ high_society/
RUN uv sync --frozen --no-dev

# Copy backend
COPY app/ app/

# Copy trained weights
COPY experiments/results/pool/ experiments/results/pool/

# Copy frontend build
COPY --from=frontend-build /build/dist app/frontend/dist

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
