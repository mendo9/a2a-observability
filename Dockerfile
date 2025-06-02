FROM python:3.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy project files
COPY pyproject.toml uv.lock* ./
COPY src/ ./src/

# Install dependencies
RUN uv sync --frozen

# Install debugpy and watchdog for remote debugging and hot reload
RUN uv add debugpy watchdog

# Copy environment file
COPY .env* ./

# Expose application port and debug port
EXPOSE 8000 5678

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/.well-known/agent.json || exit 1

# Create development startup script with debug capability but no wait
RUN echo '#!/bin/bash\n\
echo "🚀 Starting A2A Server with Phoenix observability and debug capability..."\n\
echo "🔧 Debug server available on 0.0.0.0:5678 (attach anytime)"\n\
echo "📊 Phoenix UI: http://localhost:6006"\n\
uv run python -Xfrozen_modules=off -m debugpy --listen 0.0.0.0:5678 src/server.py\n\
' > /app/dev_start.sh && chmod +x /app/dev_start.sh

# Run the server in development mode
CMD ["/app/dev_start.sh"] 