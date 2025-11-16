FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    build-essential \
    cmake \
    pkg-config \
    libopenblas-dev \
    lsof \
    procps \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p models

COPY model_installation.sh .
RUN chmod +x model_installation.sh

RUN if [ ! -f "models/Phi-3.5-mini-instruct-Q4_K_M.gguf" ]; then \
        echo "Downloading model..." && \
        cd models && \
        wget -q --show-progress https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q4_K_M.gguf; \
    else \
        echo "Model already exists, skipping download"; \
    fi

RUN if [ -f "models/Phi-3.5-mini-instruct-Q4_K_M.gguf" ]; then \
        MODEL_SIZE=$(stat -c%s "models/Phi-3.5-mini-instruct-Q4_K_M.gguf"); \
        echo "Model size: $MODEL_SIZE bytes"; \
        if [ "$MODEL_SIZE" -lt 1000000 ]; then \
            echo "Error: Model file appears to be corrupted or incomplete"; \
            exit 1; \
        fi; \
        echo "Model verification successful"; \
    else \
        echo "Error: Model file not found after download"; \
        exit 1; \
    fi

COPY api/ ./api/
COPY starter.sh .
RUN chmod +x starter.sh
RUN cat > entrypoint.sh << 'EOF'
set -e

echo "🚀 Starting moeJson Docker Container"
echo "======================================"

if [ ! -f "models/Phi-3.5-mini-instruct-Q4_K_M.gguf" ]; then
    echo "❌ Model file not found!"
    exit 1
fi

echo "✅ Model file verified"

# Check if required scripts exist
if [ ! -f "api/model_server.py" ]; then
    echo "❌ Model server script not found!"
    exit 1
fi

if [ ! -f "api/app.py" ]; then
    echo "❌ App script not found!"
    exit 1
fi

echo "✅ All required files present"

RUN chmod +x starter.sh
echo "🎯 Launching services..."
exec ./starter.sh
EOF

EXPOSE 9000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:9000/health || exit 1

# Set entrypoint
ENTRYPOINT ["./starter.sh"]